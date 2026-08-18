from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from .specs import FleetSpec, TPUEntry, slugify, stable_id

GCLOUD_LIST_TIMEOUT_SECONDS = 90
GCLOUD_MUTATION_TIMEOUT_SECONDS = 300


PENDING_QUEUED_RESOURCE_STATES = {
    "ACCEPTED",
    "WAITING_FOR_RESOURCES",
    "PROVISIONING",
    "CREATING",
    "ACTIVE",
}

TERMINAL_QUEUED_RESOURCE_STATES = {
    "FAILED",
    "DELETING",
    "SUSPENDING",
    "SUSPENDED",
    "EXPIRED",
}

TERMINAL_TPU_VM_STATES = {
    "PREEMPTED",
    "SUSPENDING",
    "SUSPENDED",
    "TERMINATED",
    "STOPPED",
    "FAILED",
    "DELETING",
}


@dataclass(frozen=True)
class TPUVM:
    name: str
    zone: str
    accelerator_type: str
    state: str = "READY"
    health: str = "HEALTHY"
    labels: dict[str, str] = field(default_factory=dict)
    spot: bool = True
    queued_resource: str | None = None
    worker_count: int = 1
    has_public_ip: bool = False

    @property
    def operationally_ready(self) -> bool:
        return (
            self.state.upper() in {"READY", "HEALTHY"}
            and self.health.upper() == "HEALTHY"
        )

    def supports_ssh_transport(self, ssh_transport: str) -> bool:
        if ssh_transport == "iap":
            return not self.has_public_ip
        if ssh_transport == "direct":
            return self.has_public_ip
        raise ValueError(f"unsupported SSH transport: {ssh_transport!r}")

    def ready_for_ssh(self, ssh_transport: str) -> bool:
        return self.operationally_ready and self.supports_ssh_transport(ssh_transport)

    @property
    def terminal(self) -> bool:
        return self.state.upper() in TERMINAL_TPU_VM_STATES


@dataclass(frozen=True)
class QueuedResource:
    name: str
    zone: str
    accelerator_type: str
    state: str
    labels: dict[str, str] = field(default_factory=dict)
    node_id: str | None = None
    provisioning_model: str = "SPOT"

    @property
    def pending_or_active(self) -> bool:
        return self.state.upper() in PENDING_QUEUED_RESOURCE_STATES

    @property
    def terminal(self) -> bool:
        return self.state.upper() in TERMINAL_QUEUED_RESOURCE_STATES

    @property
    def deleting(self) -> bool:
        return self.state.upper() in {"DELETING", "SUSPENDING"}


class SubprocessGCPClient:
    def __init__(self, *, fleet: FleetSpec):
        self.fleet = fleet

    def list_tpus(
        self,
        project: str | None = None,
        *,
        additional_targets: tuple[tuple[str, str], ...] = (),
    ) -> list[TPUVM]:
        targets = merge_inventory_targets(
            declared_tpu_targets(self.fleet), additional_targets
        )
        payloads = describe_inventory_targets(
            targets,
            resource_args=("tpu-vm",),
            project=project,
        )
        return [parse_tpu_vm(payload) for payload in payloads]

    def list_queued_resources(
        self,
        project: str | None = None,
        *,
        additional_targets: tuple[tuple[str, str], ...] = (),
    ) -> list[QueuedResource]:
        targets = merge_inventory_targets(
            declared_queued_resource_targets(self.fleet), additional_targets
        )
        payloads = describe_inventory_targets(
            targets,
            resource_args=("queued-resources",),
            project=project,
        )
        return [parse_queued_resource(payload) for payload in payloads]

    def create_queued_resource(
        self,
        *,
        fleet: FleetSpec,
        entry: TPUEntry,
        queued_resource_name: str,
        node_id: str,
    ) -> None:
        subprocess.run(
            build_create_queued_resource_command(
                fleet=fleet,
                entry=entry,
                queued_resource_name=queued_resource_name,
                node_id=node_id,
            ),
            check=True,
            timeout=GCLOUD_MUTATION_TIMEOUT_SECONDS,
        )

    def delete_queued_resource(self, *, name: str, zone: str, project: str | None = None) -> None:
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            name,
            f"--zone={zone}",
            "--quiet",
            "--force",
            "--async",
        ]
        if project:
            command.append(f"--project={project}")
        run_idempotent_delete(command)

    def delete_tpu_vm(self, *, name: str, zone: str, project: str | None = None) -> None:
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--zone={zone}",
            "--quiet",
            "--async",
        ]
        if project:
            command.append(f"--project={project}")
        run_idempotent_delete(command)


def run_idempotent_delete(command: list[str]) -> None:
    """Run one exact GCP deletion, accepting an already-absent target.

    Controller reconciliation operates on a point-in-time inventory. A prior
    deletion can complete before a later phase consumes that same snapshot;
    Cloud TPU then returns NOT_FOUND for the repeated exact deletion. Treating
    that terminal state as success keeps the rest of the reconciliation cycle
    moving while preserving failures such as permission or identity errors.
    """

    try:
        subprocess.run(
            command,
            check=True,
            timeout=GCLOUD_MUTATION_TIMEOUT_SECONDS,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        if "NOT_FOUND:" in output and "was not found" in output:
            return
        raise


def declared_tpu_targets(fleet: FleetSpec) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    for entry in fleet.tpus:
        if entry.adopted:
            targets.append((str(entry.existing), entry.zone))
            continue
        targets.extend(
            (generated_resource_names(entry, ordinal)[1], entry.zone)
            for ordinal in range(1, entry.count + 1)
        )
    return tuple(targets)


def declared_queued_resource_targets(
    fleet: FleetSpec,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (generated_resource_names(entry, ordinal)[0], entry.zone)
        for entry in fleet.tpus
        if not entry.adopted
        for ordinal in range(1, entry.count + 1)
    )


def merge_inventory_targets(
    *target_groups: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                (str(name).strip(), str(zone).strip())
                for targets in target_groups
                for name, zone in targets
                if str(name).strip() and str(zone).strip()
            },
            key=lambda item: (item[1], item[0]),
        )
    )


def describe_inventory_targets(
    targets: tuple[tuple[str, str], ...],
    *,
    resource_args: tuple[str, ...],
    project: str | None,
) -> list[dict]:
    """Describe only exact declared names, in parallel, ignoring absence."""

    def describe(target: tuple[str, str]) -> dict | None:
        name, zone = target
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            *resource_args,
            "describe",
            name,
            f"--zone={zone}",
            "--format=json",
        ]
        if project:
            command.append(f"--project={project}")
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=GCLOUD_LIST_TIMEOUT_SECONDS,
        )
        if process.returncode == 0:
            return json.loads(process.stdout)
        output = "\n".join(
            part for part in (process.stdout, process.stderr) if part
        )
        if is_not_found_output(output):
            return None
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=process.stdout,
            stderr=process.stderr,
        )

    # Keep subprocess creation on the controller's main thread. Firestore uses
    # a background gRPC poller, and forking gcloud children from worker threads
    # can inherit its file descriptors and destabilize the long-lived process.
    return [payload for target in targets if (payload := describe(target))]


def is_not_found_output(output: str) -> bool:
    lowered = output.lower()
    return (
        "not_found" in lowered
        or "was not found" in lowered
        or "could not fetch resource" in lowered
    )


def build_create_queued_resource_command(
    *,
    fleet: FleetSpec,
    entry: TPUEntry,
    queued_resource_name: str,
    node_id: str,
) -> list[str]:
    if entry.adopted:
        raise ValueError("adopted TPU entries are never created")
    if not entry.runtime:
        raise ValueError("created TPU entries require a runtime")
    labels = {
        "managed-by": fleet.name,
        "fleet-entry": entry.id,
    }
    command = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        queued_resource_name,
        f"--zone={entry.zone}",
        f"--accelerator-type={entry.type}",
        f"--runtime-version={entry.runtime}",
        f"--node-id={node_id}",
        "--provisioning-model=spot",
        "--spot",
        "--async",
        f"--network={fleet.network}",
        f"--subnetwork={fleet.subnetwork}",
        "--labels=" + ",".join(f"{key}={value}" for key, value in labels.items()),
        f"--metadata=startup-script-url={fleet.bucket}/artifacts/startup.sh",
    ]
    if fleet.ssh_transport == "iap":
        command.append("--internal-ips")
    if fleet.project:
        command.append(f"--project={fleet.project}")
        command.extend(
            [
                f"--service-account={fleet.name}-worker@{fleet.project}.iam.gserviceaccount.com",
                "--scopes=https://www.googleapis.com/auth/cloud-platform",
            ]
        )
    return command


def generated_resource_names(entry: TPUEntry, ordinal: int) -> tuple[str, str]:
    payload = {"entry": entry.id, "zone": entry.zone, "type": entry.type, "ordinal": ordinal}
    suffix = stable_id("r", payload).removeprefix("r-")[:8]
    base = slugify(f"{entry.runner_name}-{entry.id}-{ordinal}-{suffix}")[:50]
    return f"qr-{base}", base


def resource_matches_entry(resource: TPUVM | QueuedResource, entry: TPUEntry) -> bool:
    if resource.zone != entry.zone or resource.accelerator_type != entry.type:
        return False
    if isinstance(resource, TPUVM):
        provisioning_model = "spot" if resource.spot else "on-demand"
    else:
        provisioning_model = resource.provisioning_model.lower()
    if provisioning_model != entry.provisioning_model:
        return False
    if resource.labels.get("managed-by") != entry.runner_name:
        return False
    return resource.labels.get("fleet-entry") == entry.id


def parse_tpu_vm(item: dict) -> TPUVM:
    labels = item.get("labels") or {}
    name = short_zone(item.get("name", ""))
    zone = short_zone(item.get("zone", "")) or location_from_resource_name(item.get("name", ""))
    accelerator = item.get("acceleratorType") or item.get("accelerator_type") or ""
    if "/" in accelerator:
        accelerator = accelerator.rsplit("/", 1)[-1]
    scheduling = item.get("schedulingConfig") or {}
    endpoints = item.get("networkEndpoints") or []
    return TPUVM(
        name=name,
        zone=zone,
        accelerator_type=accelerator,
        state=item.get("state", item.get("health", "READY")),
        health=item.get("health", "HEALTHY"),
        labels={str(key): str(value) for key, value in labels.items()},
        spot=bool(
            scheduling.get("spot")
            or scheduling.get("preemptible", False)
            or str(item.get("provisioningModel", "")).upper() == "SPOT"
        ),
        queued_resource=short_zone(item.get("queuedResource", "")) or None,
        worker_count=max(1, len(endpoints)),
        has_public_ip=any(bool(endpoint.get("accessConfig")) for endpoint in endpoints),
    )


def parse_queued_resource(item: dict) -> QueuedResource:
    tpu = item.get("tpu") or {}
    node_spec = {}
    if isinstance(tpu.get("nodeSpec"), list) and tpu["nodeSpec"]:
        node_spec = tpu["nodeSpec"][0]
    elif isinstance(tpu.get("nodeSpec"), dict):
        node_spec = tpu["nodeSpec"]
    node = node_spec.get("node") or {}
    labels = item.get("labels") or node.get("labels") or {}
    accelerator = (
        node_spec.get("acceleratorType")
        or node.get("acceleratorType")
        or item.get("acceleratorType")
        or item.get("accelerator_type")
        or ""
    )
    if "/" in accelerator:
        accelerator = accelerator.rsplit("/", 1)[-1]
    return QueuedResource(
        name=short_zone(item.get("name", "")),
        zone=short_zone(item.get("zone", "")) or location_from_resource_name(item.get("name", "")),
        accelerator_type=accelerator,
        state=queued_resource_state(item),
        labels={str(key): str(value) for key, value in labels.items()},
        node_id=node_spec.get("nodeId") or node_spec.get("node_id"),
        provisioning_model=str(item.get("provisioningModel", "SPOT")).upper(),
    )


def short_zone(zone: str) -> str:
    return zone.rsplit("/", 1)[-1] if zone else zone


def queued_resource_state(item: dict) -> str:
    state = item.get("state", "")
    if isinstance(state, dict):
        return str(state.get("state") or state.get("name") or "")
    return str(state)


def location_from_resource_name(name: str) -> str:
    parts = name.split("/")
    try:
        return parts[parts.index("locations") + 1]
    except (ValueError, IndexError):
        return ""
