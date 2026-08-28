from __future__ import annotations

import argparse
import base64
import fnmatch
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import textwrap
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .region_pools import region_is_in_pool
from .specs import (
    DEFAULT_RUNNER_BUCKET_LOCATION,
    JOB_PRIORITY_CLASSES,
    JobSpec,
    job_specs_from_dict,
    load_fleet_spec,
    load_job_specs,
    load_yaml_file,
    region_from_zone,
    slugify,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOYMENT_PATH = Path("deployment.yaml")
DEFAULT_JOB_PATH = Path("job.yaml")
CONTROLLER_RECONCILE_SECONDS = 30
CONTROLLER_LEASE_SECONDS = 900
CONTROLLER_LEASE_RENEW_SECONDS = 60
GOOGLE_API_TIMEOUT_SECONDS = 90
WATCH_POLL_SECONDS = 10
TERMINAL_JOB_STATES = {"succeeded", "failed", "deactivated"}
IGNORED_BUNDLE_NAMES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "__pycache__",
}


class ControllerLeaseRenewer:
    """Keep a controller lease live while reconciliation blocks on remote I/O."""

    def __init__(
        self,
        *,
        store,
        name: str,
        owner: str,
        epoch: str,
        ttl_seconds: int,
        interval_seconds: int,
    ) -> None:
        if interval_seconds <= 0 or interval_seconds >= ttl_seconds:
            raise ValueError("lease renewal interval must be positive and below its TTL")
        self.store = store
        self.name = name
        self.owner = owner
        self.epoch = epoch
        self.ttl_seconds = ttl_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._renew_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("controller lease renewer is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="tpu-runner-controller-lease",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                if not self.renew():
                    return
            except Exception as exc:
                # A synchronous controller heartbeat remains authoritative and
                # will either recover the renewal or stop reconciliation. Keep
                # retrying here so one transient Firestore error does not let a
                # healthy lease age all the way to its TTL.
                print(
                    f"controller lease background renewal failed; retrying: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def renew(self) -> bool:
        if self._lost.is_set() or self._stop.is_set():
            return False
        with self._renew_lock:
            if self._lost.is_set() or self._stop.is_set():
                return False
            acquired = self.store.acquire_lease(
                self.name,
                self.owner,
                self.ttl_seconds,
                epoch=self.epoch,
            )
            if not acquired:
                self._lost.set()
            return bool(acquired)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Do not release the lease until an in-flight background renewal is
            # finished; otherwise it could reacquire immediately after release.
            self._thread.join()
            self._thread = None


def main(argv: list[str] | None = None) -> int:
    """Run the public CLI with concise errors for expected user failures."""

    try:
        return _main(argv)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"tpu-runner: error: {exc}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tpu-runner",
        description=(
            "Orchestrate queued jobs across regional Google Cloud TPU capacity, "
            "including Spot, on-demand, and adopted TPUs."
        ),
        epilog=(
            "Run 'tpu-runner COMMAND --help' for command behavior and an example.\n"
            "State and control events use JSON; logs remain readable text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def public_parser(
        name: str,
        *,
        summary: str,
        description: str,
        example: str,
    ) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            help=summary,
            description=textwrap.fill(description),
            epilog=f"Example:\n  {example}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    def internal_parser(name: str, description: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            description=f"Internal deployment command. {description}",
        )

    fleet_parser = public_parser(
        "validate-fleet",
        summary="validate a deployment file without cloud changes",
        description=(
            "Parse and validate the complete fleet configuration, then print its "
            "normalized bucket and TPU entries. This performs no cloud mutations."
        ),
        example="tpu-runner validate-fleet deployment.yaml",
    )
    fleet_parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DEPLOYMENT_PATH),
        help="deployment YAML path (default: deployment.yaml)",
    )

    jobs_parser = public_parser(
        "validate-jobs",
        summary="validate a job manifest without submitting it",
        description=(
            "Parse and validate every job in a manifest and print the normalized "
            "job specifications. This uploads nothing and creates no queue records."
        ),
        example="tpu-runner validate-jobs job.yaml",
    )
    jobs_parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_JOB_PATH),
        help="job manifest YAML path (default: job.yaml)",
    )

    init_parser = public_parser(
        "init",
        summary="write an example deployment file in the current directory",
        description=(
            "Copy the packaged deployment template to PATH. Existing files are "
            "never overwritten."
        ),
        example="tpu-runner init deployment.yaml",
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DEPLOYMENT_PATH),
        help="output path (default: deployment.yaml)",
    )

    submit_parser = public_parser(
        "submit",
        summary="submit jobs from a manifest",
        description=(
            "Validate the selected jobs, archive and upload their local bundles and "
            "regional specifications, atomically create the queue records, and "
            "trigger the controller. JSON events include each generated job ID and "
            "artifact location. Capacity racing never creates duplicate executions."
        ),
        example="tpu-runner submit job.yaml --job-id train-1",
    )
    submit_parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_JOB_PATH),
        help="job manifest YAML path (default: job.yaml)",
    )
    submit_parser.add_argument(
        "--job-id",
        action="append",
        dest="job_ids",
        help="submit only this exact manifest job id; may be repeated",
    )

    watch_parser = public_parser(
        "watch",
        summary="watch one job until it is terminal",
        description=(
            "Poll one job, stream available Cloud Logging output, and emit JSON state, "
            "assignment, and artifact events until the job succeeds, fails, or is "
            "deactivated. Assignment events identify the exact TPU and attempt. The "
            "exit status is zero only when the job succeeds."
        ),
        example="tpu-runner watch JOB_ID",
    )
    watch_parser.add_argument("job_id", help="job ID to watch")

    logs_parser = public_parser(
        "logs",
        summary="show logs and diagnostics for one job",
        description=(
            "Print durable per-worker logs and diagnostics from the job bucket, then "
            "show recent Cloud Logging entries. This is a read-only snapshot, not a "
            "continuous follow operation."
        ),
        example="tpu-runner logs JOB_ID",
    )
    logs_parser.add_argument("job_id", help="job ID whose logs should be shown")

    cancel_parser = public_parser(
        "cancel",
        summary="cancel one or more jobs",
        description=(
            "Deactivate pending jobs or request exact-attempt cancellation for active "
            "jobs. One JSON result is printed per job. Use --if-pending when assignment "
            "must lose rather than race this request; conflicts return a nonzero status."
        ),
        example="tpu-runner cancel JOB_ID --if-pending",
    )
    cancel_parser.add_argument("job_ids", nargs="+", help="job IDs to cancel")
    cancel_parser.add_argument(
        "--if-pending",
        action="store_true",
        help=(
            "deactivate only if the job is still unassigned and pending; "
            "fail without mutation if assignment has started"
        ),
    )

    priority_parser = public_parser(
        "set-priority",
        summary="atomically reprioritize one pending, unassigned job",
        description=(
            "Change queue priority only if the exact job is still pending and "
            "unassigned. If assignment has started, the command reports a conflict "
            "without changing the job and returns a nonzero status."
        ),
        example="tpu-runner set-priority JOB_ID high",
    )
    priority_parser.add_argument("job_id", help="pending job ID")
    priority_parser.add_argument(
        "priority",
        choices=JOB_PRIORITY_CLASSES,
        help="new queue priority",
    )

    interrupt_parser = public_parser(
        "interrupt-spot",
        summary="request deletion of one exact runner-managed Spot attempt",
        description=(
            "Request controlled deletion of one exact active attempt on a runner-managed "
            "Spot TPU. All three identities are revalidated transactionally; adopted and "
            "on-demand resources are rejected."
        ),
        example=(
            "tpu-runner interrupt-spot RESOURCE_ID --job-id JOB_ID "
            "--attempt-id ATTEMPT_ID"
        ),
    )
    interrupt_parser.add_argument("resource_id", help="managed TPU resource ID")
    interrupt_parser.add_argument("--job-id", required=True, help="owning job ID")
    interrupt_parser.add_argument(
        "--attempt-id", required=True, help="active attempt ID"
    )

    probe_parser = public_parser(
        "probe-adopted-device-owners",
        summary="read TPU device owners on an adopted resource without mutation",
        description=(
            "Inspect root-visible TPU device owners on every worker of one adopted TPU. "
            "The probe is read-only and does not stop processes or change runner state."
        ),
        example="tpu-runner probe-adopted-device-owners --resource-id RESOURCE_ID",
    )
    probe_parser.add_argument(
        "--resource-id", required=True, help="adopted TPU resource ID"
    )

    status_parser = public_parser(
        "status",
        summary="emit read-only active fleet state as JSON",
        description=(
            "Read Firestore and print schema-versioned fleet, job, attempt, resource, "
            "and interruption-request state. Counts and full item records are included; "
            "no controller or cloud resource is changed."
        ),
        example="tpu-runner status --deployment deployment.yaml --pretty",
    )
    status_parser.add_argument(
        "--deployment",
        default=str(DEFAULT_DEPLOYMENT_PATH),
        help="deployment YAML path (default: deployment.yaml)",
    )
    status_parser.add_argument(
        "--pretty", action="store_true", help="pretty-print the JSON output"
    )

    internal_parser(
        "controller",
        "Run the Cloud Run reconciliation loop for the deployed fleet.",
    )
    bootstrap_parser = internal_parser(
        "bootstrap-ready",
        "Wait for the deployed worker startup artifact to become available.",
    )
    bootstrap_parser.add_argument(
        "--deployment",
        default=str(DEFAULT_DEPLOYMENT_PATH),
        help="deployment YAML path",
    )
    bootstrap_parser.add_argument(
        "--startup", required=True, help="rendered startup script path"
    )

    deploy_parser = public_parser(
        "deploy",
        summary="create or update the declared runner deployment",
        description=(
            "Validate the deployment, enable required APIs, configure the runner bucket, "
            "Firestore, service accounts, IAM and SSH identity, build the controller "
            "image, and roll out and start the Cloud Run controller job."
        ),
        example="tpu-runner deploy deployment.yaml",
    )
    deploy_parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DEPLOYMENT_PATH),
        help="deployment YAML path (default: deployment.yaml)",
    )

    lease_parser = internal_parser(
        "release-controller-lease",
        "Release the controller lease only when it is owned by the given execution.",
    )
    lease_parser.add_argument(
        "--deployment", required=True, help="deployment YAML path"
    )
    lease_parser.add_argument(
        "--owner", required=True, help="expected controller lease owner"
    )

    wait_lease_parser = internal_parser(
        "wait-controller-release",
        "Wait until the previous controller lease is absent or expired.",
    )
    wait_lease_parser.add_argument(
        "--deployment", required=True, help="deployment YAML path"
    )
    wait_lease_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
        help="maximum wait in seconds (default: 1200)",
    )
    wait_lease_parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="poll interval in seconds (default: 5)",
    )

    epoch_parser = internal_parser(
        "set-controller-epoch",
        "Fence older controller deployments with a new exact epoch.",
    )
    epoch_parser.add_argument(
        "--deployment", required=True, help="deployment YAML path"
    )
    epoch_parser.add_argument(
        "--epoch", required=True, help="new controller deployment epoch"
    )

    args = parser.parse_args(argv)
    if args.command == "validate-fleet":
        spec = load_fleet_spec(materialize_path(args.path))
        print(json.dumps({"bucket": spec.bucket, "tpus": [entry.__dict__ for entry in spec.tpus]}, indent=2))
        return 0
    if args.command == "validate-jobs":
        specs = load_job_specs(materialize_path(args.path))
        jobs = []
        for job in specs:
            item = asdict(job)
            for internal in (
                "bucket_regions",
                "regional_bundles",
                "region",
                "storage_region",
                "bucket",
            ):
                item.pop(internal)
            jobs.append(item)
        print(json.dumps({"jobs": jobs}, indent=2, default=list))
        return 0
    if args.command == "init":
        output = Path(args.path)
        if output.exists():
            raise ValueError(f"refusing to overwrite existing deployment file: {output}")
        output.write_text(
            (PACKAGE_DIR / "deployment.example.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"wrote {output}")
        return 0
    if args.command == "submit":
        return submit_jobs(Path(args.path), job_ids=args.job_ids)
    if args.command == "watch":
        return watch_job(args.job_id)
    if args.command == "logs":
        return show_logs(args.job_id)
    if args.command == "cancel":
        return cancel_jobs(args.job_ids, if_pending=args.if_pending)
    if args.command == "set-priority":
        return set_job_priority(
            args.job_id,
            priority=args.priority,
        )
    if args.command == "interrupt-spot":
        return request_controlled_interruption(
            resource_id=args.resource_id,
            job_id=args.job_id,
            attempt_id=args.attempt_id,
        )
    if args.command == "probe-adopted-device-owners":
        return probe_adopted_device_owners(args.resource_id)
    if args.command == "status":
        payload = fleet_status(Path(args.deployment))
        print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    if args.command == "controller":
        from .controller import Controller
        from .gcp import ConcurrentInventoryGCPClient
        from .runtime import FirestoreStateStore

        require_controller_ssh_identity(os.environ)
        fleet, _, store = runner_context()
        owner = os.environ.get("CLOUD_RUN_EXECUTION") or os.environ.get("HOSTNAME") or f"local-{uuid.uuid4()}"
        epoch = os.environ.get("TPU_RUNNER_CONTROLLER_EPOCH", "")
        lease_name = "controller"
        if not store.acquire_lease(
            lease_name,
            owner,
            CONTROLLER_LEASE_SECONDS,
            epoch=epoch,
        ):
            store.record_event("controller_skipped", {"reason": "lease_held", "owner": owner})
            print(json.dumps({"status": "skipped", "reason": "lease_held"}))
            store.close()
            return 0
        lease_renewer = ControllerLeaseRenewer(
            store=store,
            name=lease_name,
            owner=owner,
            epoch=epoch,
            ttl_seconds=CONTROLLER_LEASE_SECONDS,
            interval_seconds=CONTROLLER_LEASE_RENEW_SECONDS,
        )
        lease_renewer.start()
        controller = Controller(
            fleet=fleet,
            store=store,
            gcp=ConcurrentInventoryGCPClient(fleet=fleet),
            renew_lease=lease_renewer.renew,
        )
        try:
            while True:
                if not lease_renewer.renew():
                    raise RuntimeError("controller lost its leader lease")
                try:
                    controller.reconcile_once()
                except subprocess.SubprocessError as exc:
                    store.record_event("controller_gcp_error", {"error": str(exc)})
                    print(f"controller GCP operation failed; retrying: {exc}", file=os.sys.stderr, flush=True)
                    time.sleep(CONTROLLER_RECONCILE_SECONDS)
                    continue
                active_jobs = store.has_jobs_with_statuses(
                    {"pending", "running", "cancelling"}
                )
                if active_jobs or controller.managed_capacity_exists():
                    time.sleep(CONTROLLER_RECONCILE_SECONDS)
                    continue

                # Release before the final check so a concurrent submission can
                # either be observed here or acquire the lease itself.
                lease_renewer.stop()
                store.release_lease(lease_name, owner)
                if not store.has_jobs_with_statuses(
                    {"pending", "running", "cancelling"}
                ):
                    print(json.dumps({"status": "idle"}))
                    return 0
                if not store.acquire_lease(
                    lease_name,
                    owner,
                    CONTROLLER_LEASE_SECONDS,
                    epoch=epoch,
                ):
                    print(json.dumps({"status": "handed_off"}))
                    return 0
                lease_renewer = ControllerLeaseRenewer(
                    store=store,
                    name=lease_name,
                    owner=owner,
                    epoch=epoch,
                    ttl_seconds=CONTROLLER_LEASE_SECONDS,
                    interval_seconds=CONTROLLER_LEASE_RENEW_SECONDS,
                )
                lease_renewer.start()
                controller.renew_lease = lease_renewer.renew
        finally:
            lease_renewer.stop()
            try:
                store.release_lease(lease_name, owner)
            finally:
                store.close()
    if args.command == "bootstrap-ready":
        return bootstrap_ready(Path(args.deployment), Path(args.startup))
    if args.command == "deploy":
        deployment_path = Path(args.path).resolve()
        fleet = load_fleet_spec(deployment_path)
        env = dict(os.environ)
        env["TPU_RUNNER_PYTHON"] = sys.executable
        env["TPU_RUNNER_DEPLOYMENT_PATH"] = str(deployment_path)
        env["TPU_RUNNER_NAME"] = fleet.name
        env["TPU_RUNNER_PROJECT"] = fleet.project
        env["TPU_RUNNER_BUCKET"] = fleet.bucket
        env["TPU_RUNNER_BUCKET_LOCATION"] = DEFAULT_RUNNER_BUCKET_LOCATION
        env["TPU_RUNNER_CONTROLLER_REGION"] = fleet.controller_region
        env["TPU_RUNNER_CONTROLLER_TIMEOUT"] = fleet.controller_timeout
        env["TPU_RUNNER_CONTROLLER_MEMORY"] = fleet.controller_memory
        env["TPU_RUNNER_CONTROLLER_MAX_RETRIES"] = str(
            fleet.controller_max_retries
        )
        env["TPU_RUNNER_SSH_TRANSPORT"] = fleet.ssh_transport
        env["TPU_RUNNER_CONTROLLER_EPOCH"] = uuid.uuid4().hex
        env["TPU_RUNNER_FIRESTORE_LOCATION"] = fleet.firestore_location
        env["TPU_RUNNER_NETWORK"] = fleet.network
        env["TPU_RUNNER_WORKER_SECRETS"] = "\n".join(fleet.worker_secrets)
        return subprocess.run(["bash", str(PACKAGE_DIR / "deploy.sh")], env=env, check=False).returncode
    if args.command == "release-controller-lease":
        from .runtime import FirestoreStateStore

        fleet = load_fleet_spec(materialize_path(args.deployment))
        store = FirestoreStateStore(
            collection_prefix=fleet.name.replace("-", "_"),
            project=fleet.project,
        )
        try:
            store.release_lease("controller", args.owner)
        finally:
            store.close()
        print(json.dumps({"status": "released_if_owned", "owner": args.owner}))
        return 0
    if args.command == "wait-controller-release":
        from .runtime import FirestoreStateStore, lease_is_live

        if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
            raise ValueError("controller release timeout and poll interval must be positive")
        fleet = load_fleet_spec(materialize_path(args.deployment))
        store = FirestoreStateStore(
            collection_prefix=fleet.name.replace("-", "_"),
            project=fleet.project,
        )
        deadline = time.monotonic() + args.timeout_seconds
        try:
            while True:
                lease = store.get_lease("controller")
                if not lease_is_live(lease, now=datetime.now(timezone.utc)):
                    print(json.dumps({"status": "released", "lease": lease}, default=str, sort_keys=True))
                    return 0
                if time.monotonic() >= deadline:
                    print(
                        json.dumps({"status": "timeout", "lease": lease}, default=str, sort_keys=True),
                        file=sys.stderr,
                    )
                    return 1
                time.sleep(args.poll_seconds)
        finally:
            store.close()
    if args.command == "set-controller-epoch":
        from .runtime import FirestoreStateStore

        fleet = load_fleet_spec(materialize_path(args.deployment))
        store = FirestoreStateStore(
            collection_prefix=fleet.name.replace("-", "_"),
            project=fleet.project,
        )
        try:
            store.set_lease_epoch("controller", args.epoch)
        finally:
            store.close()
        print(json.dumps({"status": "updated", "epoch": args.epoch}))
        return 0
    raise AssertionError(args.command)


def require_controller_ssh_identity(env: Mapping[str, str]) -> None:
    """Fail before leasing when a controller cannot act as the runner user."""

    ssh_user = env.get("TPU_RUNNER_SSH_USER", "").strip()
    ssh_private_key = env.get("TPU_RUNNER_SSH_PRIVATE_KEY", "").strip()
    if not ssh_user or not ssh_private_key:
        raise ValueError(
            "controller requires TPU_RUNNER_SSH_USER and "
            "TPU_RUNNER_SSH_PRIVATE_KEY; use the deployed controller or inject "
            "the exact runner identity"
        )


def materialize_path(path: str) -> Path:
    if not path.startswith("gs://"):
        return Path(path)
    tmp = Path(tempfile.mkdtemp(prefix="tpu-runner-spec-")) / Path(path).name
    subprocess.run(["gcloud", "storage", "cp", path, str(tmp)], check=True)
    return tmp


def bootstrap_ready(deployment_path: Path, startup_path: Path) -> int:
    from .gcp import ConcurrentInventoryGCPClient, resource_matches_entry

    fleet = load_fleet_spec(deployment_path)
    ssh_transport = "direct" if fleet.ssh_transport == "direct" else "iap"
    payload = base64.b64encode(startup_path.read_bytes()).decode()
    remote_command = f"printf %s {shlex.quote(payload)} | base64 -d | sudo bash"
    tpus, _ = ConcurrentInventoryGCPClient(fleet=fleet).list_inventory(
        project=fleet.project
    )
    ready_tpus = sorted(
        (
            tpu
            for tpu in tpus
            if tpu.ready_for_ssh(ssh_transport)
            and any(resource_matches_entry(tpu, entry) for entry in fleet.tpus)
        ),
        key=lambda tpu: (tpu.zone, tpu.name),
    )
    for tpu in ready_tpus:
        command = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh",
            tpu.name,
            f"--project={fleet.project}",
            f"--zone={tpu.zone}",
            "--worker=all",
            "--quiet",
            f"--command={remote_command}",
        ]
        if ssh_transport == "iap":
            command.insert(-2, "--tunnel-through-iap")
        subprocess.run(command, check=True)
        print(f"bootstrapped ready TPU {tpu.name}")
    return 0


def runner_fleet():
    deployment_path = os.environ.get("TPU_RUNNER_DEPLOYMENT", str(DEFAULT_DEPLOYMENT_PATH))
    fleet = load_fleet_spec(materialize_path(deployment_path))
    return fleet, fleet.project


def runner_context():
    from .runtime import FirestoreStateStore

    fleet, project = runner_fleet()
    prefix = fleet.name.replace("-", "_")
    return fleet, project, FirestoreStateStore(project=project, collection_prefix=prefix)


def google_authorized_session(project: str, *, scopes: tuple[str, ...]):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(scopes=scopes)
    if project and hasattr(credentials, "with_quota_project"):
        credentials = credentials.with_quota_project(project)
    return AuthorizedSession(credentials)


def probe_adopted_device_owners(resource_id: str) -> int:
    from .distributed import DistributedTPURunner
    from .runtime import FirestoreStateStore

    fleet, project, store = runner_context()
    try:
        resource = store.get_resource(resource_id)
        if resource is None:
            raise ValueError(f"resource not found: {resource_id!r}")
        if not resource.adopted:
            raise ValueError(f"resource {resource_id!r} is not adopted")
        if resource.status != "idle" or resource.current_job_id or resource.current_attempt_id:
            raise ValueError(f"resource {resource_id!r} is not exactly idle and unassigned")
    finally:
        store.close()
    workers = DistributedTPURunner(
        name=fleet.name,
        project=project,
        ssh_transport=fleet.ssh_transport,
    ).probe_device_owners(resource=resource)
    clear = len(workers) == max(1, resource.worker_count) and all(
        bool(worker["clear"]) for worker in workers.values()
    )
    payload = {
        "resource_id": resource.id,
        "tpu_name": resource.tpu_name,
        "zone": resource.zone,
        "clear": clear,
        "workers": workers,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    event_store = FirestoreStateStore(
        project=project,
        collection_prefix=fleet.name.replace("-", "_"),
    )
    try:
        event_store.record_event("adopted_tpu_device_owner_probe", payload)
    finally:
        event_store.close()
    print(json.dumps(payload, sort_keys=True))
    return 0 if clear else 75


def fleet_status(deployment_path: Path) -> dict:
    """Read active runner state without loading terminal job history."""
    from .controller import resource_is_inventory_owned_for_fleet
    from .runtime import FirestoreStateStore

    fleet = load_fleet_spec(materialize_path(str(deployment_path)))
    store = FirestoreStateStore(
        project=fleet.project,
        collection_prefix=fleet.name.replace("-", "_"),
    )
    try:
        jobs = store.list_jobs_with_statuses({"pending", "running", "cancelling"})
        attempts = [
            attempt
            for job in jobs
            if job.current_attempt_id
            for attempt in store.list_attempts_for_job(job.spec.id)
        ]
        resources = [
            resource
            for resource in store.list_resources_excluding_statuses(
                {"deleted", "preempted"}
            )
            if resource_is_inventory_owned_for_fleet(resource, fleet)
        ]
        interruption_requests = store.list_interruption_requests_with_statuses(
            {"requested", "processing"}
        )
    finally:
        store.close()
    return build_status_payload(
        fleet=fleet,
        jobs=jobs,
        attempts=attempts,
        resources=resources,
        interruption_requests=interruption_requests,
    )


def build_status_payload(
    *,
    fleet,
    jobs,
    attempts,
    resources,
    interruption_requests,
) -> dict:
    """Build the small JSON contract used by operational verifiers."""
    generated_at = datetime.now(timezone.utc)
    from .gcp import generated_resource_names

    attempts_by_job: dict[str, list] = {}
    from .controller import desired_managed_capacity_counts

    attempts_by_id = {attempt.id: attempt for attempt in attempts}
    for attempt in attempts:
        attempts_by_job.setdefault(attempt.job_id, []).append(attempt)
    resources_by_id = {resource.id: resource for resource in resources}
    job_items: list[dict] = []
    for job in sorted(jobs, key=lambda candidate: candidate.spec.id):
        job_attempts = attempts_by_job.get(job.spec.id, [])
        current = attempts_by_id.get(job.current_attempt_id or "")
        resource = resources_by_id.get(current.resource_id) if current else None
        compute_region = region_from_zone(resource.zone) if resource else ""
        if not job.spec.storage_region:
            placement_status = "racing"
        elif compute_region and (
            not region_is_in_pool(compute_region, job.spec.region)
            or compute_region != job.spec.storage_region
        ):
            placement_status = "compute_region_mismatch"
        else:
            placement_status = "pinned"
        requested_tpu = list(job.spec.tpu) if isinstance(job.spec.tpu, tuple) else job.spec.tpu
        job_items.append(
            {
                "job_id": job.spec.id,
                "status": job.status,
                "submitted_at": job.submitted_at,
                "priority": job.spec.priority,
                "tpu": requested_tpu,
                "tpu_name": resource.tpu_name if resource else job.spec.tpu_name,
                "zone": resource.zone if resource else job.spec.zone,
                "buckets": list(job.spec.buckets),
                "selected_bucket": job.spec.bucket,
                "candidate_regions": [
                    region for region, _ in job.spec.bucket_regions
                ],
                "region": job.spec.region,
                "storage_region": job.spec.storage_region,
                "compute_region": compute_region,
                "placement_status": placement_status,
                "resource_id": current.resource_id if current else job.assigned_resource_id,
                "current_attempt_id": job.current_attempt_id,
                "current_attempt_status": current.status if current else None,
                "attempt_count": len(job_attempts),
                "interruption_count": sum(
                    attempt.status == "interrupted" for attempt in job_attempts
                ),
            }
        )

    attempt_items = sorted((asdict(attempt) for attempt in attempts), key=lambda item: item["id"])
    resource_items = sorted((asdict(resource) for resource in resources), key=lambda item: item["id"])
    interruption_items = sorted(
        (asdict(request) for request in interruption_requests), key=lambda item: item["id"]
    )
    desired_counts = desired_managed_capacity_counts(
        jobs,
        fleet=fleet,
        resources=tuple(resources),
    )
    fleet_entries = []
    for entry in fleet.tpus:
        declared = (
            [{"tpu_name": entry.existing, "queued_resource": None}]
            if entry.adopted
            else [
                {"queued_resource": queued_name, "tpu_name": node_id}
                for queued_name, node_id in (
                    generated_resource_names(entry, ordinal)
                    for ordinal in range(1, entry.count + 1)
                )
            ]
        )
        fleet_entries.append(
            {
                "id": entry.id,
                "type": entry.type,
                "zone": entry.zone,
                "provisioning_model": entry.provisioning_model,
                "adopted": entry.adopted,
                "ceiling_count": 1 if entry.adopted else entry.count,
                "desired_count": (
                    1 if entry.adopted else desired_counts.get(entry.id, 0)
                ),
                "keep_warm_count": entry.keep_warm_count,
                "declared": declared,
            }
        )

    def counts(items: list[dict]) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown")
            result[status] = result.get(status, 0) + 1
        return dict(sorted(result.items()))

    return {
        "schema_version": 2,
        "generated_at": generated_at.isoformat(),
        "fleet": {
            "name": fleet.name,
            "project": fleet.project,
            "idle_timeout_seconds": fleet.idle_timeout_seconds,
            "entries": fleet_entries,
        },
        "jobs": {"counts": counts(job_items), "items": job_items},
        "attempts": {"counts": counts(attempt_items), "items": attempt_items},
        "resources": {"counts": counts(resource_items), "items": resource_items},
        "interruption_requests": {
            "counts": counts(interruption_items),
            "items": interruption_items,
        },
    }


def select_job_specs(
    raw_specs: tuple[JobSpec, ...], requested_job_ids: Sequence[str] | None
) -> tuple[JobSpec, ...]:
    """Return an exact manifest-ordered selection before any external work."""
    if requested_job_ids is None:
        return raw_specs
    requested = tuple(str(job_id) for job_id in requested_job_ids)
    if not requested or any(not job_id for job_id in requested):
        raise ValueError("job selection must contain at least one non-empty job id")
    duplicates = sorted(
        job_id for job_id in set(requested) if requested.count(job_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate requested job id(s): {', '.join(duplicates)}")
    known_ids = {spec.id for spec in raw_specs}
    missing = sorted(set(requested) - known_ids)
    if missing:
        raise ValueError(f"requested job id(s) not found in manifest: {', '.join(missing)}")
    selected_ids = set(requested)
    selected = tuple(spec for spec in raw_specs if spec.id in selected_ids)
    if not selected:
        raise ValueError("job selection is empty")
    return selected


def submit_jobs(path: Path, *, job_ids: Sequence[str] | None = None) -> int:
    from .gcp import generated_resource_names
    from .placement import (
        gcloud_bucket_location,
        resolve_regional_buckets,
        validate_job_gcs_dependencies,
    )
    from .runtime import GCSArtifactClient, JobRecord, job_record_to_dict

    source_path = materialize_path(str(path))
    data = load_yaml_file(source_path)
    jobs_raw = data.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("job spec field 'jobs' must be a non-empty list")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for index, raw in enumerate(jobs_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each job entry must be a mapping")
        if not raw.get("id"):
            env = raw.get("env") if isinstance(raw.get("env"), dict) else {}
            base = slugify(str(env.get("RUN_NAME") or f"job-{index}"))[:40]
            raw["id"] = f"{base}-{stamp}-{uuid.uuid4().hex[:6]}"
    raw_specs = select_job_specs(job_specs_from_dict(data), job_ids)

    fleet, project = runner_fleet()
    known_tpus = {
        entry.existing: entry
        for entry in fleet.tpus
        if entry.existing
    }
    for entry in fleet.tpus:
        if entry.adopted:
            continue
        for ordinal in range(1, entry.count + 1):
            _, tpu_name = generated_resource_names(entry, ordinal)
            known_tpus[tpu_name] = entry
    for raw_spec in raw_specs:
        if raw_spec.tpu_name:
            target_entry = known_tpus.get(raw_spec.tpu_name)
            if target_entry is None:
                raise ValueError(f"tpu_name is not declared in the fleet: {raw_spec.tpu_name}")
            if not raw_spec.accepts_tpu_type(target_entry.type):
                raise ValueError(
                    f"tpu_name {raw_spec.tpu_name} has type {target_entry.type}, "
                    f"which is not allowed by tpu={raw_spec.tpu}"
                )
            if raw_spec.zone and raw_spec.zone != target_entry.zone:
                raise ValueError(
                    f"tpu_name {raw_spec.tpu_name} is in zone {target_entry.zone}, "
                    f"not requested zone {raw_spec.zone}"
                )
        elif raw_spec.zone and not any(
            raw_spec.accepts_tpu_type(entry.type)
            and raw_spec.zone == entry.zone
            for entry in fleet.tpus
        ):
            raise ValueError(
                f"zone {raw_spec.zone} has no declared fleet capacity allowed by "
                f"tpu={raw_spec.tpu}"
            )

    regional_specs: list[JobSpec] = []
    bucket_locations: dict[str, str] = {}
    for raw_spec in raw_specs:
        target_entry = known_tpus.get(raw_spec.tpu_name) if raw_spec.tpu_name else None

        def resolve(storage_bucket: str) -> str:
            if storage_bucket not in bucket_locations:
                bucket_locations[storage_bucket] = gcloud_bucket_location(
                    storage_bucket,
                    project=project,
                )
            return bucket_locations[storage_bucket]

        bucket_regions = resolve_regional_buckets(
            raw_spec.buckets,
            resolve_bucket_location=resolve,
        )
        validate_job_gcs_dependencies(
            raw_spec,
            bucket_regions=bucket_regions,
            resolve_bucket_location=resolve,
        )
        required_region = ""
        if target_entry is not None:
            required_region = region_from_zone(target_entry.zone)
        elif raw_spec.zone:
            required_region = region_from_zone(raw_spec.zone)
        available_regions = {region for region, _ in bucket_regions}
        if required_region and required_region not in available_regions:
            raise ValueError(
                f"job {raw_spec.id!r} has no bucket in required region "
                f"{required_region!r}"
            )
        compatible_regions = {
            region_from_zone(entry.zone)
            for entry in fleet.tpus
            if raw_spec.accepts_tpu_type(entry.type)
            and (not raw_spec.zone or raw_spec.zone == entry.zone)
        }
        if not available_regions & compatible_regions:
            raise ValueError(
                f"job {raw_spec.id!r} has no compatible fleet capacity in its "
                "bucket regions"
            )
        regional_specs.append(
            replace(
                raw_spec,
                bucket_regions=bucket_regions,
            )
        )

    artifacts = GCSArtifactClient()
    prepared: list[tuple[JobRecord, tuple[str, ...]]] = []
    submission_token = uuid.uuid4().hex
    for raw_spec in regional_specs:
        print(
            json.dumps(
                {
                    "event": "preparing",
                    "job_id": raw_spec.id,
                    "priority": raw_spec.priority,
                    "regions": [region for region, _ in raw_spec.bucket_regions],
                    "tpu": raw_spec.tpu,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        regional_bundles = publish_regional_bundles(
            raw_spec.bundle,
            base_dir=source_path.parent,
            bucket_regions=raw_spec.bucket_regions,
            project=project,
        )
        spec = replace(
            raw_spec,
            bundle="",
            regional_bundles=regional_bundles,
        )
        submitted_at = datetime.now(timezone.utc).isoformat()
        record = JobRecord(spec=spec, submitted_at=submitted_at)
        spec_uris = tuple(
            f"{bucket}/jobs/{spec.id}/spec-{submission_token}.json"
            for _, bucket in spec.bucket_regions
        )
        with tempfile.TemporaryDirectory(prefix="tpu-runner-job-") as directory:
            spec_path = Path(directory) / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "submitted_at": submitted_at,
                        "job": job_record_to_dict(record),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            for spec_uri in spec_uris:
                artifacts.upload(spec_path, spec_uri)
        prepared.append((record, spec_uris))

    from .runtime import FirestoreStateStore

    records = [record for record, _ in prepared]
    store = FirestoreStateStore(
        project=project,
        collection_prefix=fleet.name.replace("-", "_"),
    )
    records_created = False
    try:
        store.create_jobs(records)
        records_created = True
        for record, spec_uris in prepared:
            spec = record.spec
            submitted_at = record.submitted_at
            store.record_event(
                "job_submitted",
                {
                    "job_id": spec.id,
                    "regional_bundles": dict(spec.regional_bundles),
                    "spec_uris": list(spec_uris),
                    "submitted_at": submitted_at,
                    "tpu_name": spec.tpu_name,
                    "zone": spec.zone,
                    "regions": [region for region, _ in spec.bucket_regions],
                    "priority": spec.priority,
                },
                emit=False,
            )
            output = {
                "bundles": dict(spec.regional_bundles),
                "event": "submitted",
                "job_id": spec.id,
                "priority": spec.priority,
                "regions": [region for region, _ in spec.bucket_regions],
                "specs": dict(
                    zip(
                        (region for region, _ in spec.bucket_regions),
                        spec_uris,
                        strict=True,
                    )
                ),
                "submitted_at": submitted_at,
                "tpu": spec.tpu,
            }
            if spec.tpu_name:
                output["tpu_name"] = spec.tpu_name
            if spec.zone:
                output["zone"] = spec.zone
            print(json.dumps(output, sort_keys=True), flush=True)
    finally:
        store.close()
        if records_created:
            trigger_controller(
                fleet,
                job_ids=tuple(record.spec.id for record in records),
            )
    return 0


def publish_regional_bundles(
    bundle: str,
    *,
    base_dir: Path,
    bucket_regions: tuple[tuple[str, str], ...],
    project: str,
) -> tuple[tuple[str, str], ...]:
    if bundle.startswith("gs://"):
        if len(bucket_regions) != 1:
            raise ValueError("multi-region jobs require a local source bundle")
        return ((bucket_regions[0][0], bundle),)
    source = (base_dir / bundle).resolve()
    temporary_archive = source.is_dir()
    if source.is_dir():
        archive, digest = create_content_bundle(source)
    elif source.is_file() and tarfile.is_tarfile(source):
        archive = source
        digest = sha256_file(source)
    else:
        raise ValueError(f"bundle must be a directory, tar archive, or GCS URI: {source}")
    try:
        regional_bundles: list[tuple[str, str]] = []
        for region, bucket in bucket_regions:
            uri = f"{bucket}/bundles/{digest}.tar.gz"
            command = [
                "gcloud",
                "storage",
                "cp",
                "--no-clobber",
                str(archive),
                uri,
                f"--project={project}",
            ]
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    command,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            regional_bundles.append((region, uri))
        return tuple(regional_bundles)
    finally:
        if temporary_archive:
            archive.unlink(missing_ok=True)
            archive.parent.rmdir()


def create_content_bundle(root: Path) -> tuple[Path, str]:
    paths = bundle_paths(root)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(f"{path.lstat().st_mode & 0o777:o}".encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"L\0")
            digest.update(os.readlink(path).encode())
            digest.update(b"\0")
        elif path.is_file():
            digest.update(b"F\0")
            content_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content_digest.update(chunk)
            digest.update(content_digest.digest())
        else:
            digest.update(b"D\0")
    hexdigest = digest.hexdigest()
    archive = Path(tempfile.mkdtemp(prefix="tpu-runner-bundle-")) / f"{hexdigest}.tar.gz"
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in paths:
                    tar.add(
                        path,
                        arcname=path.relative_to(root).as_posix(),
                        recursive=False,
                        filter=normalized_tar_info,
                    )
    return archive, hexdigest


def bundle_paths(root: Path) -> list[Path]:
    ignore_patterns = bundle_ignore_patterns(root)
    paths: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            path = current_path / name
            if ignored_bundle_path(path.relative_to(root), ignore_patterns):
                continue
            paths.append(path)
            if not path.is_symlink():
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(files):
            path = current_path / name
            if not ignored_bundle_path(path.relative_to(root), ignore_patterns):
                paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def bundle_ignore_patterns(root: Path) -> tuple[str, ...]:
    path = root / ".tpu-runnerignore"
    if not path.is_file():
        return ()
    return tuple(
        line
        for raw_line in path.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )


def ignored_bundle_path(relative: Path, patterns: tuple[str, ...] = ()) -> bool:
    if any(part in IGNORED_BUNDLE_NAMES for part in relative.parts) or relative.suffix == ".pyc":
        return True
    candidate = relative.as_posix()
    for raw_pattern in patterns:
        pattern = raw_pattern.strip("/")
        if not pattern:
            continue
        if "/" in pattern:
            if fnmatch.fnmatchcase(candidate, pattern):
                return True
        elif any(fnmatch.fnmatchcase(part, pattern) for part in relative.parts):
            return True
    return False


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trigger_controller(fleet, *, job_ids: Sequence[str] = ()) -> None:
    job = f"{fleet.name}-controller"
    region = fleet.controller_region
    session = google_authorized_session(
        fleet.project,
        scopes=("https://www.googleapis.com/auth/cloud-platform",),
    )
    try:
        response = session.post(
            "https://run.googleapis.com/v2/"
            f"projects/{fleet.project}/locations/{region}/jobs/{job}:run",
            json={},
            timeout=GOOGLE_API_TIMEOUT_SECONDS,
        )
        if not response.ok:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise RuntimeError(f"controller trigger failed: {detail}")
        try:
            operation = response.json()
        except ValueError as exc:
            raise RuntimeError("controller trigger returned invalid JSON") from exc
    finally:
        session.close()

    output: dict[str, object] = {
        "controller": job,
        "event": "controller_triggered",
        "region": region,
    }
    operation_name = operation.get("name") if isinstance(operation, dict) else None
    if operation_name:
        output["operation"] = operation_name
    if job_ids:
        output["job_ids"] = list(job_ids)
    print(json.dumps(output, sort_keys=True), flush=True)


def cancel_jobs(job_ids: Sequence[str], *, if_pending: bool = False) -> int:
    fleet, _, store = runner_context()
    exit_code = 0
    changed = False
    try:
        for job_id in job_ids:
            status = (
                store.cancel_job(job_id, if_pending=True)
                if if_pending
                else store.cancel_job(job_id)
            )
            if status is None:
                result = {"job_id": job_id, "status": "unknown"}
                exit_code = max(exit_code, 1)
            elif status == "conflict":
                result = {
                    "job_id": job_id,
                    "status": "conflict",
                    "reason": "job is no longer unassigned and pending",
                }
                exit_code = 2
            else:
                result = {"job_id": job_id, "status": status}
                changed = changed or status in {"cancelling", "deactivated"}
            print(json.dumps(result, sort_keys=True))
    finally:
        store.close()
    if changed:
        trigger_controller(fleet)
    return exit_code


def set_job_priority(
    job_id: str,
    *,
    priority: str,
) -> int:
    """Atomically reprioritize exactly one pending, unassigned job."""

    fleet, _, store = runner_context()
    try:
        status = store.reprioritize_pending_job(
            job_id,
            priority=priority,
        )
    finally:
        store.close()
    if status is None:
        result = {"job_id": job_id, "status": "unknown"}
        exit_code = 1
    elif status == "conflict":
        result = {
            "job_id": job_id,
            "status": "conflict",
            "reason": "job is no longer unassigned and pending",
        }
        exit_code = 2
    else:
        result = {
            "job_id": job_id,
            "status": status,
            "priority": priority,
        }
        exit_code = 0
    print(json.dumps(result, sort_keys=True))
    if status == "reprioritized":
        trigger_controller(fleet)
    return exit_code


def request_controlled_interruption(
    *, resource_id: str, job_id: str, attempt_id: str
) -> int:
    fleet, _, store = runner_context()
    eligible_fleet_entry_ids = {
        entry.id
        for entry in fleet.tpus
        if not entry.adopted and entry.provisioning_model == "spot"
    }
    try:
        try:
            request = store.create_interruption_request(
                resource_id=resource_id,
                job_id=job_id,
                attempt_id=attempt_id,
                eligible_fleet_entry_ids=eligible_fleet_entry_ids,
            )
        except (KeyError, ValueError) as exc:
            print(f"interruption request rejected: {exc}", file=os.sys.stderr)
            return 1
    finally:
        store.close()
    print(
        json.dumps(
            {
                "request_id": request.id,
                "resource_id": request.resource_id,
                "job_id": request.job_id,
                "attempt_id": request.attempt_id,
                "status": request.status,
            },
            sort_keys=True,
        )
    )
    trigger_controller(fleet)
    return 0


def watch_job(job_id: str) -> int:
    fleet, project, store = runner_context()
    seen_logs: set[str] = set()
    previous_state = ""
    previous_log_error = ""
    previous_assignment: tuple[str, str] | None = None
    entries_by_id = {entry.id: entry for entry in fleet.tpus}
    logging_session = google_authorized_session(
        project,
        scopes=("https://www.googleapis.com/auth/logging.read",),
    )
    try:
        while True:
            job = store.get_job(job_id)
            if job is None:
                print(
                    json.dumps(
                        {
                            "error": "unknown job",
                            "event": "error",
                            "job_id": job_id,
                        },
                        sort_keys=True,
                    ),
                    file=os.sys.stderr,
                )
                return 1
            attempt = store.get_attempt(job.current_attempt_id) if job.current_attempt_id else None
            resource_id = job.assigned_resource_id or (attempt.resource_id if attempt else "")
            assignment = (
                (resource_id, attempt.id if attempt else "")
                if resource_id
                else None
            )
            if assignment is not None and assignment != previous_assignment:
                resource = store.get_resource(resource_id)
                assigned: dict[str, object] = {
                    "attempt_id": attempt.id if attempt else "",
                    "event": "assigned",
                    "job_id": job_id,
                    "resource": resource_id,
                    "tpu_name": resource.tpu_name if resource else resource_id,
                }
                if resource is not None:
                    assigned.update(
                        {
                            "adopted": resource.adopted,
                            "fleet_entry": resource.fleet_entry_id or "",
                            "tpu_type": resource.tpu_type,
                            "worker_count": resource.worker_count,
                            "zone": resource.zone,
                        }
                    )
                    entry = entries_by_id.get(resource.fleet_entry_id or "")
                    if entry is not None:
                        assigned["provisioning_model"] = entry.provisioning_model
                print(json.dumps(assigned, sort_keys=True), flush=True)
            previous_assignment = assignment

            state_fields: dict[str, object] = {
                "attempt": attempt.status if attempt else "",
                "attempt_id": attempt.id if attempt else "",
                "job": job.status,
                "job_id": job_id,
                "priority": job.spec.priority,
                "resource": resource_id,
                "tpu": job.spec.tpu,
            }
            if attempt is not None:
                if attempt.exit_code is not None:
                    state_fields["exit_code"] = attempt.exit_code
                if attempt.error_summary:
                    state_fields["error"] = attempt.error_summary
                if attempt.end_reason:
                    state_fields["end_reason"] = attempt.end_reason
            if job.spec.storage_region:
                state_fields["region"] = job.spec.storage_region
            if job.spec.bucket:
                state_fields["bucket"] = job.spec.bucket
            state = json.dumps(state_fields, sort_keys=True)
            output = dict(state_fields)
            output["checked_at"] = datetime.now(timezone.utc).isoformat()
            output["event"] = "state" if state != previous_state else "poll"
            print(json.dumps(output, sort_keys=True), flush=True)
            previous_state = state
            try:
                print_cloud_logs(
                    project,
                    fleet.name,
                    job_id,
                    seen_logs,
                    session=logging_session,
                )
                if previous_log_error:
                    print(
                        json.dumps(
                            {
                                "event": "logs_recovered",
                                "job_id": job_id,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    previous_log_error = ""
            except RuntimeError as exc:
                error = str(exc)
                if error != previous_log_error:
                    print(
                        json.dumps(
                            {
                                "error": error,
                                "event": "log_warning",
                                "job_id": job_id,
                            },
                            sort_keys=True,
                        ),
                        file=os.sys.stderr,
                        flush=True,
                    )
                previous_log_error = error
            if job.status in TERMINAL_JOB_STATES:
                if job.spec.bucket:
                    print(
                        json.dumps(
                            {
                                "event": "artifacts",
                                "job_id": job_id,
                                "uri": f"{job.spec.bucket}/jobs/{job_id}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                return 0 if job.status == "succeeded" else 1
            time.sleep(WATCH_POLL_SECONDS)
    except KeyboardInterrupt:
        return 130
    finally:
        logging_session.close()
        store.close()


def show_logs(job_id: str) -> int:
    fleet, project, store = runner_context()
    try:
        job = store.get_job(job_id)
        attempts = store.list_attempts_for_job(job_id) if job is not None else []
    finally:
        store.close()
    if job is None:
        print(f"unknown job: {job_id}", file=os.sys.stderr)
        return 1
    found = False
    errors: list[str] = []
    for attempt in attempts:
        outcome = f"status={attempt.status}"
        if attempt.exit_code is not None:
            outcome += f" exit_code={attempt.exit_code}"
        if attempt.error_summary:
            outcome += f" error={attempt.error_summary}"
        print(f"[{attempt.id}] {outcome}")
        for artifact_dir, label in (("logs", "log"), ("diagnostics", "diagnostic")):
            pattern = (
                f"{job.spec.bucket}/jobs/{job_id}/attempts/{attempt.id}/"
                f"{artifact_dir}/*.log"
            )
            listing = subprocess.run(
                ["gcloud", "storage", "ls", pattern, f"--project={project}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if listing.returncode != 0:
                output = (listing.stdout + listing.stderr).strip()
                if "matched no objects" not in output.lower():
                    errors.append(output or f"could not list {pattern}")
                continue
            for uri in listing.stdout.splitlines():
                if not uri.strip():
                    continue
                found = True
                print(f"\n[{attempt.id} {label}] {uri}")
                read = subprocess.run(
                    ["gcloud", "storage", "cat", uri, f"--project={project}"],
                    text=True,
                    check=False,
                )
                if read.returncode != 0:
                    errors.append(f"could not read {uri}")
    try:
        print_cloud_logs(project, fleet.name, job_id, set())
    except RuntimeError as exc:
        errors.append(str(exc))
    if not found:
        if any(attempt.status in {"failed", "failed_setup"} for attempt in attempts):
            message = "job failed; no durable worker logs or diagnostics were uploaded"
        else:
            message = "no durable worker logs or diagnostics uploaded yet"
        print(message, file=os.sys.stderr)
    for error in errors:
        print(f"log retrieval failed: {error}", file=os.sys.stderr)
    return 1 if errors else 0


def cloud_log_entries(
    project: str,
    runner_name: str,
    job_id: str,
    *,
    session=None,
) -> list[dict]:
    query = (
        "logName="
        + json.dumps(f"projects/{project}/logs/{runner_name}-worker")
        + " AND jsonPayload.job_id="
        + json.dumps(job_id)
    )
    owns_session = session is None
    if session is None:
        session = google_authorized_session(
            project,
            scopes=("https://www.googleapis.com/auth/logging.read",),
        )
    try:
        body: dict[str, object] = {
            "filter": query,
            "orderBy": "timestamp desc",
            "pageSize": 200,
            "resourceNames": [f"projects/{project}"],
        }
        entries: list[dict] = []
        for _ in range(5):
            response = session.post(
                "https://logging.googleapis.com/v2/entries:list",
                json=body,
                timeout=GOOGLE_API_TIMEOUT_SECONDS,
            )
            if not response.ok:
                detail = response.text.strip() or f"HTTP {response.status_code}"
                raise RuntimeError(f"Cloud Logging query failed: {detail}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("Cloud Logging returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("Cloud Logging returned an unexpected response")
            page_entries = payload.get("entries", [])
            if not isinstance(page_entries, list):
                raise RuntimeError("Cloud Logging returned invalid entries")
            entries.extend(item for item in page_entries if isinstance(item, dict))
            page_token = payload.get("nextPageToken")
            if entries or not isinstance(page_token, str) or not page_token:
                break
            body["pageToken"] = page_token
        return list(reversed(entries[:200]))
    finally:
        if owns_session:
            session.close()


def print_cloud_logs(
    project: str,
    runner_name: str,
    job_id: str,
    seen: set[str],
    *,
    session=None,
) -> None:
    for entry in cloud_log_entries(
        project,
        runner_name,
        job_id,
        session=session,
    ):
        payload = entry.get("jsonPayload") or {}
        key = str(entry.get("insertId") or (entry.get("timestamp"), payload.get("worker"), payload.get("message")))
        if key in seen:
            continue
        seen.add(key)
        timestamp = entry.get("timestamp", "")
        worker = payload.get("worker", "worker")
        attempt_id = payload.get("attempt_id", "")
        message = str(payload.get("message", "")).rstrip()
        header = f"{timestamp} job_id={job_id} worker={worker}"
        if attempt_id:
            header += f" attempt_id={attempt_id}"
        print(f"{header}\n{message}", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
