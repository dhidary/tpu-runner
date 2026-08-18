from __future__ import annotations

import base64
import json
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .runtime import AttemptRecord, ResourceRecord, checkpoint_dir, job_bucket
from .specs import CacheSpec, JobSpec

RETRYABLE_INFRASTRUCTURE_EXIT_CODE = 75
RETRYABLE_INFRASTRUCTURE_FALLOUT_EXIT_CODES = frozenset({0, 1, 75, 143})


def is_retryable_infrastructure_exit_set(exit_codes: set[int]) -> bool:
    """Recognize one exact TPU interruption plus bounded sibling shutdown fallout."""
    return (
        RETRYABLE_INFRASTRUCTURE_EXIT_CODE in exit_codes
        and exit_codes <= RETRYABLE_INFRASTRUCTURE_FALLOUT_EXIT_CODES
    )


class TemporaryAccessError(RuntimeError):
    pass


SSH_TIMEOUT_SECONDS = 180
STARTUP_READY_TIMEOUT_SECONDS = 120
DEVICE_RELEASE_TIMEOUT_SECONDS = 60
ARTIFACT_UPLOAD_ATTEMPTS = 3
ARTIFACT_UPLOAD_RETRY_SECONDS = 5
_LAUNCH_ACK_RE = re.compile(
    r"^TPU_RUNNER_(?:LAUNCHED|ALREADY_RUNNING|ALREADY_COMPLETE)\s+(\S+)(?:\s+.*)?$",
    re.MULTILINE,
)
_CANCEL_ACK_RE = re.compile(
    r"^RUNNER_CANCEL_(SENT|NOT_RUNNING|ATTEMPT_MISMATCH)\s+(\S+)\s+(\S+)$",
    re.MULTILINE,
)
_DISK_RECOVERY_ACK_RE = re.compile(
    r"^TPU_RUNNER_DISK_RECOVERY\s+(\S+)\s+(SUFFICIENT|RECOVERED)"
    r"\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)$",
    re.MULTILINE,
)
_DEVICE_OWNER_PROBE_ACK_RE = re.compile(
    r"^TPU_RUNNER_DEVICE_OWNER_PROBE\s+(\S+)\s+(CLEAR|BUSY)\s+([0-9,-]+)$",
    re.MULTILINE,
)
MINIMUM_ROOT_FREE_KB = 10 * 1024 * 1024


@dataclass(frozen=True)
class DistributedResult:
    statuses: dict[str, dict]
    complete: bool
    exit_code: int | None = None
    failure_kind: str = ""
    error_summary: str = ""


@dataclass(frozen=True)
class DistributedTPURunner:
    name: str
    project: str | None = None
    ssh_transport: str = "direct"

    @property
    def work_root(self) -> str:
        return f"/tmp/{self.name}/work"

    @property
    def cache_root(self) -> str:
        return f"/tmp/{self.name}/cache"

    @property
    def shm_root(self) -> str:
        return f"/dev/shm/{self.name}"

    @property
    def process_root(self) -> str:
        return f"/tmp/{self.name}/process"

    @property
    def startup_ready_marker(self) -> str:
        return f"/var/lib/{self.name}/startup-ready"

    @property
    def log_name(self) -> str:
        return f"{self.name}-worker"

    def launch(self, *, job: JobSpec, attempt: AttemptRecord, resource: ResourceRecord) -> None:
        result = self.run_tpu_vm_ssh_all(
            resource=resource,
            script=self.launch_script(job=job, attempt=attempt, resource=resource),
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0:
            # gcloud's multi-worker SSH command can return a non-zero aggregate
            # status after every remote worker has already detached its wrapper
            # successfully (for example, when an IAP transport exits noisily).
            # Treat the launch as established only when every expected worker
            # emitted its path-specific acknowledgement.  The normal status
            # poll remains authoritative for command/setup completion.
            acknowledged_hosts = set(_LAUNCH_ACK_RE.findall(output))
            if len(acknowledged_hosts) == max(1, resource.worker_count):
                return
            if looks_like_temporary_access_error(output):
                raise TemporaryAccessError(output.strip())
            raise RuntimeError(output.strip() or f"distributed launch failed with exit {result.returncode}")

    def cancel(self, *, attempt: AttemptRecord, resource: ResourceRecord) -> str:
        result = self.run_tpu_vm_ssh_all(
            resource=resource,
            script=f"""#!/usr/bin/env bash
set -euo pipefail
PID_FILE={shlex.quote(self.process_root + '/distributed-current.pid')}
EXPECTED_ATTEMPT={shlex.quote(attempt.id)}
HOST="$(hostname)"
if [[ ! -f "$PID_FILE" ]]; then
  echo "RUNNER_CANCEL_NOT_RUNNING $HOST $EXPECTED_ATTEMPT"
  exit 0
fi
read -r current_attempt current_pid < "$PID_FILE" || true
if [[ "$current_attempt" != "$EXPECTED_ATTEMPT" || ! "$current_pid" =~ ^[0-9]+$ ]]; then
  echo "RUNNER_CANCEL_ATTEMPT_MISMATCH $HOST $EXPECTED_ATTEMPT"
  exit 0
fi
if ! kill -0 "$current_pid" 2>/dev/null; then
  echo "RUNNER_CANCEL_NOT_RUNNING $HOST $EXPECTED_ATTEMPT"
  exit 0
fi
current_pgid="$(ps -o pgid= -p "$current_pid" | tr -d ' ')"
if [[ "$current_pgid" =~ ^[0-9]+$ ]]; then
  kill -TERM -- "-$current_pgid"
  cancel_deadline=$((SECONDS + 10))
  while kill -0 "$current_pid" 2>/dev/null && (( SECONDS < cancel_deadline )); do
    sleep 1
  done
  kill -0 "$current_pid" 2>/dev/null && kill -KILL -- "-$current_pgid" 2>/dev/null || true
else
  kill -TERM "$current_pid"
fi
echo "RUNNER_CANCEL_SENT $HOST $EXPECTED_ATTEMPT"
""",
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        acknowledgements = {
            host: outcome
            for outcome, host, acknowledged_attempt in _CANCEL_ACK_RE.findall(output)
            if acknowledged_attempt == attempt.id
        }
        if len(acknowledgements) == max(1, resource.worker_count):
            outcomes = set(acknowledgements.values())
            if "ATTEMPT_MISMATCH" in outcomes:
                return "attempt_mismatch"
            if "SENT" in outcomes:
                return "sent"
            return "not_running"
        if result.returncode != 0:
            if looks_like_temporary_access_error(output):
                raise TemporaryAccessError(output.strip())
            if acknowledgements:
                raise TemporaryAccessError(
                    "distributed cancel received only "
                    f"{len(acknowledgements)}/{max(1, resource.worker_count)} exact worker "
                    "acknowledgements: "
                    + output.strip()
                )
            raise RuntimeError(output.strip() or f"distributed cancel failed with exit {result.returncode}")
        raise RuntimeError("distributed cancel returned no outcome")

    def recover_adopted_disk_pressure(
        self, *, resource: ResourceRecord
    ) -> dict[str, dict[str, int | str]]:
        """Clear only runner-owned scratch when an idle adopted TPU disk is unsafe."""
        result = self.run_tpu_vm_ssh_all(
            resource=resource,
            script=self.adopted_disk_recovery_script(),
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        acknowledgements = {
            host: {
                "outcome": outcome.lower(),
                "free_kb_before": int(free_before),
                "free_kb_after": int(free_after),
                "work_entries_removed": int(work_entries),
                "cache_dirs_removed": int(cache_dirs),
            }
            for (
                host,
                outcome,
                free_before,
                free_after,
                work_entries,
                cache_dirs,
            ) in _DISK_RECOVERY_ACK_RE.findall(output)
        }
        expected = max(1, resource.worker_count)
        if result.returncode != 0:
            if looks_like_temporary_access_error(output):
                raise TemporaryAccessError(output.strip())
            raise RuntimeError(
                output.strip()
                or f"adopted TPU disk recovery failed with exit {result.returncode}"
            )
        if len(acknowledgements) != expected:
            raise RuntimeError(
                "adopted TPU disk recovery received only "
                f"{len(acknowledgements)}/{expected} exact worker acknowledgements: "
                + output.strip()
            )
        return acknowledgements

    def probe_device_owners(
        self, *, resource: ResourceRecord
    ) -> dict[str, dict[str, object]]:
        """Read root-visible TPU device owners on every worker without mutation."""
        result = self.run_tpu_vm_ssh_all(
            resource=resource,
            script=self.device_owner_probe_script(),
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        acknowledgements = {
            host: {
                "clear": outcome == "CLEAR",
                "owner_pids": (
                    [] if owners == "-" else [int(pid) for pid in owners.split(",")]
                ),
            }
            for host, outcome, owners in _DEVICE_OWNER_PROBE_ACK_RE.findall(output)
        }
        expected = max(1, resource.worker_count)
        if result.returncode != 0:
            if looks_like_temporary_access_error(output):
                raise TemporaryAccessError(output.strip())
            raise RuntimeError(
                output.strip()
                or f"device-owner probe failed with exit {result.returncode}"
            )
        if len(acknowledgements) != expected:
            raise RuntimeError(
                "device-owner probe received only "
                f"{len(acknowledgements)}/{expected} exact worker acknowledgements: "
                + output.strip()
            )
        return acknowledgements

    @staticmethod
    def device_owner_probe_script() -> str:
        return """#!/usr/bin/env bash
set -euo pipefail
HOST="$(hostname)"
shopt -s nullglob
devices=(/dev/accel* /dev/vfio/[0-9]*)
if (( ${#devices[@]} == 0 )); then
  echo "[runner] no TPU accelerator devices found on $HOST" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "[runner] passwordless sudo is required for root-visible owner probe on $HOST" >&2
  exit 2
fi
owners="$(sudo -n /usr/bin/lsof -t "${devices[@]}" 2>/dev/null || true)"
owners="$(printf '%s\n' "$owners" | sed '/^$/d' | sort -nu | paste -sd, -)"
if [[ -n "$owners" ]]; then
  echo "TPU_RUNNER_DEVICE_OWNER_PROBE $HOST BUSY $owners"
else
  echo "TPU_RUNNER_DEVICE_OWNER_PROBE $HOST CLEAR -"
fi
"""

    def adopted_disk_recovery_script(self) -> str:
        return f"""#!/usr/bin/env bash
set -euo pipefail
HOST="$(hostname)"
WORK_ROOT={shlex.quote(self.work_root)}
CACHE_ROOT={shlex.quote(self.cache_root)}
PROCESS_ROOT={shlex.quote(self.process_root)}
PID_FILE="$PROCESS_ROOT/distributed-current.pid"
minimum_free_kb={MINIMUM_ROOT_FREE_KB}
runner_uid="$(id -u)"
for runner_root in "$WORK_ROOT" "$CACHE_ROOT" "$PROCESS_ROOT"; do
  if [[ ! -e "$runner_root" ]]; then
    continue
  fi
  if [[ -L "$runner_root" || ! -d "$runner_root" ]]; then
    echo "[runner] refusing disk recovery for unsafe runner root: $runner_root" >&2
    exit 2
  fi
  canonical_root="$(readlink -f -- "$runner_root" 2>/dev/null || true)"
  if [[ "$canonical_root" != "$runner_root" ]]; then
    echo "[runner] refusing disk recovery for non-canonical runner root: $runner_root" >&2
    exit 2
  fi
  root_owner="$(stat -c %u "$runner_root" 2>/dev/null || true)"
  if [[ "$root_owner" != "$runner_uid" ]]; then
    echo "[runner] refusing disk recovery for runner root with wrong owner: $runner_root" >&2
    exit 2
  fi
done
if [[ -L "$PID_FILE" ]]; then
  echo "[runner] refusing disk recovery for unsafe runner PID file: $PID_FILE" >&2
  exit 2
fi
if [[ -f "$PID_FILE" ]]; then
  read -r remote_attempt remote_pid < "$PID_FILE" || true
  if [[ "${{remote_pid:-}}" =~ ^[0-9]+$ ]] && kill -0 "$remote_pid" 2>/dev/null; then
    echo "[runner] refusing disk recovery while runner process is active: attempt=${{remote_attempt:-unknown}} pid=$remote_pid" >&2
    exit 2
  fi
fi
free_kb_before="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo invalid)"
if [[ ! "$free_kb_before" =~ ^[0-9]+$ ]]; then
  echo "[runner] could not measure root disk before adopted TPU recovery" >&2
  exit 2
fi
if (( free_kb_before >= minimum_free_kb )); then
  echo "TPU_RUNNER_DISK_RECOVERY $HOST SUFFICIENT $free_kb_before $free_kb_before 0 0"
  exit 0
fi
work_entries_removed=0
if [[ -d "$WORK_ROOT" ]]; then
  work_entries_removed="$(find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -printf '.\\n' | wc -l | tr -d ' ')"
  find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
fi
cache_dirs_removed=0
free_kb="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo invalid)"
while [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < minimum_free_kb )); do
  [[ -d "$CACHE_ROOT" ]] || break
  victim="$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\\n' | sort -n | head -n 1 | cut -d' ' -f2-)"
  [[ -n "$victim" ]] || break
  case "$victim" in
    "$CACHE_ROOT"/*) ;;
    *)
      echo "[runner] refusing disk recovery for cache path outside runner root: $victim" >&2
      exit 2
      ;;
  esac
  rm -rf -- "$victim"
  cache_dirs_removed=$((cache_dirs_removed + 1))
  free_kb="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo invalid)"
done
free_kb_after="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo invalid)"
if [[ ! "$free_kb_after" =~ ^[0-9]+$ ]] || (( free_kb_after < minimum_free_kb )); then
  echo "[runner] adopted TPU disk recovery could not restore root floor: free_kb=$free_kb_after required_kb=$minimum_free_kb" >&2
  exit 2
fi
echo "TPU_RUNNER_DISK_RECOVERY $HOST RECOVERED $free_kb_before $free_kb_after $work_entries_removed $cache_dirs_removed"
"""

    def poll(self, *, job: JobSpec, attempt: AttemptRecord, resource: ResourceRecord) -> DistributedResult:
        statuses = self.read_gcs_statuses(job=job, attempt=attempt)
        expected = max(1, resource.worker_count)
        if len(statuses) < expected:
            partial_artifact_errors = [
                f"{host}: {status['artifact_upload_error']}"
                for host, status in statuses.items()
                if status.get("artifact_upload_error")
            ]
            if partial_artifact_errors:
                return DistributedResult(
                    statuses=statuses,
                    complete=True,
                    exit_code=2,
                    failure_kind="artifact_upload_failed",
                    error_summary=(
                        "artifact upload failed before all worker statuses arrived: "
                        + "; ".join(partial_artifact_errors)
                    )[-4000:],
                )
            partial_exit_codes = {
                int(status.get("command_exit_code", status.get("exit_code", 1)))
                for status in statuses.values()
            }
            if is_retryable_infrastructure_exit_set(partial_exit_codes):
                command_errors = [
                    f"{host}: command exited "
                    f"{status.get('command_exit_code', status.get('exit_code', 1))}"
                    for host, status in statuses.items()
                    if int(status.get("command_exit_code", status.get("exit_code", 1))) != 0
                ]
                command_errors.append(
                    f"{expected - len(statuses)} worker status object(s) missing after exact "
                    "retryable infrastructure exit"
                )
                return DistributedResult(
                    statuses=statuses,
                    complete=True,
                    exit_code=RETRYABLE_INFRASTRUCTURE_EXIT_CODE,
                    failure_kind="retryable_infrastructure",
                    error_summary="; ".join(command_errors),
                )
            if partial_exit_codes and partial_exit_codes != {0}:
                states = {str(status.get("state", "failed")) for status in statuses.values()}
                command_errors = [
                    f"{host}: command exited "
                    f"{status.get('command_exit_code', status.get('exit_code', 1))}"
                    for host, status in statuses.items()
                    if int(status.get("command_exit_code", status.get("exit_code", 1))) != 0
                ]
                command_errors.append(
                    f"{expected - len(statuses)} worker status object(s) missing after "
                    "distributed command failure"
                )
                return DistributedResult(
                    statuses=statuses,
                    complete=True,
                    exit_code=max(partial_exit_codes),
                    failure_kind="failed_setup" if "failed_setup" in states else "command_failed",
                    error_summary="; ".join(command_errors),
                )
            return DistributedResult(statuses=statuses, complete=False)
        exit_code = max(int(status.get("exit_code", 1)) for status in statuses.values())
        states = {str(status.get("state", "failed")) for status in statuses.values()}
        artifact_errors = [
            f"{host}: {status['artifact_upload_error']}"
            for host, status in statuses.items()
            if status.get("artifact_upload_error")
        ]
        command_errors = [
            f"{host}: command exited {status.get('command_exit_code', status.get('exit_code', 1))}"
            for host, status in statuses.items()
            if int(status.get("command_exit_code", status.get("exit_code", 1))) != 0
        ]
        command_exit_codes = {
            int(status.get("command_exit_code", status.get("exit_code", 1)))
            for status in statuses.values()
        }
        if artifact_errors:
            failure_kind = "artifact_upload_failed"
            details = [f"artifact upload failed: {'; '.join(artifact_errors)}", *command_errors]
            error_summary = "; ".join(details)[-4000:]
        elif "failed_setup" in states:
            failure_kind = "failed_setup"
            error_summary = "failed_setup: " + "; ".join(command_errors)
        elif is_retryable_infrastructure_exit_set(command_exit_codes):
            failure_kind = "retryable_infrastructure"
            error_summary = "; ".join(command_errors)
        elif exit_code:
            failure_kind = "command_failed"
            error_summary = "; ".join(command_errors)
        else:
            failure_kind = ""
            error_summary = ""
        return DistributedResult(
            statuses=statuses,
            complete=True,
            exit_code=(
                RETRYABLE_INFRASTRUCTURE_EXIT_CODE
                if failure_kind == "retryable_infrastructure"
                else exit_code
            ),
            failure_kind=failure_kind,
            error_summary=error_summary,
        )

    def read_gcs_statuses(self, *, job: JobSpec, attempt: AttemptRecord) -> dict[str, dict]:
        bucket = job_bucket(job)
        pattern = f"{bucket}/jobs/{job.id}/attempts/{attempt.id}/status/*.json"
        command = ["gcloud", "storage", "ls", pattern]
        if self.project:
            command.append(f"--project={self.project}")
        listing = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        output = listing.stdout + listing.stderr
        if listing.returncode != 0:
            if "matched no objects" in output.lower():
                return {}
            raise TemporaryAccessError(output.strip() or "could not list worker status objects")

        statuses: dict[str, dict] = {}
        for uri in (line.strip() for line in listing.stdout.splitlines()):
            if not uri:
                continue
            cat_command = ["gcloud", "storage", "cat", uri]
            if self.project:
                cat_command.append(f"--project={self.project}")
            result = subprocess.run(
                cat_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise TemporaryAccessError(result.stderr.strip() or f"could not read {uri}")
            try:
                status = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise TemporaryAccessError(f"worker status is not valid JSON: {uri}") from exc
            host = str(status.get("host", ""))
            if host:
                statuses[host] = status
        return statuses

    def launch_script(self, *, job: JobSpec, attempt: AttemptRecord, resource: ResourceRecord) -> str:
        env = dict(job.env)
        env.update(self.worker_env(job=job, attempt=attempt, resource=resource))
        exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))
        cache_setup = self.cache_setup_script(job.caches)
        command = shlex.quote(job.command)
        bundle = shlex.quote(job.bundle)
        job_dir = shlex.quote(self.job_dir(job))
        attempt_dir = shlex.quote(self.attempt_dir(job, attempt))
        shm_dir = shlex.quote(f"{self.shm_root}/{job.id}")
        return f"""#!/usr/bin/env bash
set -euo pipefail
HOST="$(hostname)"
WORK_ROOT={shlex.quote(self.work_root)}
SHM_ROOT={shlex.quote(self.shm_root)}
CACHE_ROOT={shlex.quote(self.cache_root)}
JOB_DIR={job_dir}
ATTEMPT_DIR={attempt_dir}
SHM_DIR={shm_dir}
PROCESS_DIR={shlex.quote(self.process_root)}
PID_FILE="$PROCESS_DIR/distributed-current.pid"
CURRENT_ATTEMPT_ID={shlex.quote(attempt.id)}
{exports}
STARTUP_READY_MARKER={shlex.quote(self.startup_ready_marker)}
STARTUP_READY_TIMEOUT_SECONDS={STARTUP_READY_TIMEOUT_SECONDS}
DEVICE_RELEASE_TIMEOUT_SECONDS={DEVICE_RELEASE_TIMEOUT_SECONDS}
current_boot_id="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
startup_ready_deadline=$((SECONDS + STARTUP_READY_TIMEOUT_SECONDS))
while [[ -z "$current_boot_id" ]] || \
      [[ ! -r "$STARTUP_READY_MARKER" ]] || \
      ! grep -Fxq "boot_id=$current_boot_id" "$STARTUP_READY_MARKER"; do
  if (( SECONDS >= startup_ready_deadline )); then
    echo "[runner] startup readiness timed out after ${{STARTUP_READY_TIMEOUT_SECONDS}}s: $STARTUP_READY_MARKER" >&2
    exit 2
  fi
  sleep 2
done
gcloud_path="$(command -v gcloud 2>/dev/null || true)"
if [[ -z "$gcloud_path" ]]; then
  echo "[runner] gcloud executable is unavailable before launch" >&2
  exit 2
fi
if [[ "$gcloud_path" == /* ]]; then
  GCLOUD_BIN="$gcloud_path"
else
  gcloud_dir="$(dirname -- "$gcloud_path")"
  GCLOUD_BIN="$(cd -- "$gcloud_dir" && pwd -P)/$(basename -- "$gcloud_path")"
fi
if [[ "$GCLOUD_BIN" != /* || ! -x "$GCLOUD_BIN" ]]; then
  echo "[runner] resolved gcloud executable is invalid: $GCLOUD_BIN" >&2
  exit 2
fi
runner_uid="$(id -u)"
install -d -m 0755 "$WORK_ROOT" "$PROCESS_DIR" "$CACHE_ROOT"
install -d -m 0755 "$SHM_ROOT"
exec 9>"$PROCESS_DIR/distributed-launch.lock"
flock 9
for runner_root in "$WORK_ROOT" "$PROCESS_DIR" "$CACHE_ROOT" "$SHM_ROOT"; do
  root_owner="$(stat -c %u "$runner_root" 2>/dev/null || true)"
  if [[ "$root_owner" != "$runner_uid" ]]; then
    echo "[runner] scratch root has the wrong owner: $runner_root" >&2
    exit 2
  fi
done
if [[ -f "$ATTEMPT_DIR/status-$HOST.json" ]]; then
  until "$GCLOUD_BIN" storage cp "$ATTEMPT_DIR/status-$HOST.json" "$ATTEMPT_GCS_DIR/status/$HOST.json"; do
    echo "[runner] terminal status publication failed: $HOST" >&2
    sleep {ARTIFACT_UPLOAD_RETRY_SECONDS}
  done
  "$GCLOUD_BIN" storage cp "$ATTEMPT_DIR/command-$HOST.log" "$ATTEMPT_GCS_DIR/logs/$HOST.log" || \
    echo "[runner] command log repair upload failed: $HOST" >&2
  if [[ -f "$ATTEMPT_DIR/diagnostics-$HOST.log" ]]; then
    "$GCLOUD_BIN" storage cp "$ATTEMPT_DIR/diagnostics-$HOST.log" "$ATTEMPT_GCS_DIR/diagnostics/$HOST.log" || \
      echo "[runner] diagnostics repair upload failed: $HOST" >&2
  fi
  echo "TPU_RUNNER_ALREADY_COMPLETE $HOST $CURRENT_ATTEMPT_ID"
  exit 0
fi
if [[ -f "$PID_FILE" ]]; then
  read -r old_attempt old_pid < "$PID_FILE" || true
  if [[ -z "${{old_pid:-}}" ]]; then
    old_pid="${{old_attempt:-}}"
    old_attempt=""
  fi
  if [[ "$old_attempt" == "$CURRENT_ATTEMPT_ID" && "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "TPU_RUNNER_ALREADY_RUNNING $HOST $CURRENT_ATTEMPT_ID $old_pid"
    exit 0
  fi
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_pgid="$(ps -o pgid= -p "$old_pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ "$old_pgid" =~ ^[0-9]+$ ]]; then
      kill -TERM -- "-$old_pgid" 2>/dev/null || true
      process_release_deadline=$((SECONDS + 10))
      while kill -0 "$old_pid" 2>/dev/null && (( SECONDS < process_release_deadline )); do
        sleep 1
      done
      kill -0 "$old_pid" 2>/dev/null && kill -KILL -- "-$old_pgid" 2>/dev/null || true
    else
      kill -TERM "$old_pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
find "$SHM_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
mkdir -p "$JOB_DIR" "$ATTEMPT_DIR" "$SHM_DIR"
shm_free_kb="$(df --output=avail /dev/shm | tail -n 1 | tr -d ' ' || echo 0)"
if [[ ! "$shm_free_kb" =~ ^[0-9]+$ ]] || (( shm_free_kb < 10485760 )); then
  echo "[runner] unsafe /dev/shm: free_kb=$shm_free_kb required_kb=10485760" >&2
  exit 2
fi
"$GCLOUD_BIN" storage cp {bundle} "$ATTEMPT_DIR/bundle.tar.gz"
tar -xzf "$ATTEMPT_DIR/bundle.tar.gz" -C "$JOB_DIR"
{cache_setup}
cd "$JOB_DIR"
export TPU_WORKER_HOST="$HOST"
PREFLIGHT_LOG="$ATTEMPT_DIR/preflight-$HOST.log"
{{
  echo "host=$HOST"
  echo "time=$(date -Is)"
  df -h /
  df -h /dev/shm
  free -h || true
  command -v tpu-info >/dev/null 2>&1 && tpu-info || true
  if [[ -e /tmp/libtpu_lockfile ]]; then
    echo "libtpu_lockfile=present"
    owner_pids=""
    if command -v lsof >/dev/null 2>&1; then
      owner_pids="$(lsof -t /dev/accel0 2>/dev/null || true)"
    else
      echo "lsof=missing"
    fi
    if [[ -n "$owner_pids" ]]; then
      echo "accel0_owner_pids=$owner_pids"
      echo "libtpu_lockfile_action=kept"
    else
      echo "accel0_owner_pids=none"
      rm -f /tmp/libtpu_lockfile 2>/dev/null || true
      if [[ -e /tmp/libtpu_lockfile ]]; then
        echo "libtpu_lockfile_action=remove_failed"
      else
        echo "libtpu_lockfile_action=removed"
      fi
    fi
  else
    echo "libtpu_lockfile=absent"
  fi
}} >"$PREFLIGHT_LOG" 2>&1
"$GCLOUD_BIN" storage cp "$PREFLIGHT_LOG" "$ATTEMPT_GCS_DIR/logs/preflight-$HOST.log" >/dev/null 2>&1 || true
cat >"$ATTEMPT_DIR/run-wrapper.sh" <<'TPU_RUNNER_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
HOST="$(hostname)"
ATTEMPT_DIR="${{ATTEMPT_DIR:?}}"
ATTEMPT_GCS_DIR="${{ATTEMPT_GCS_DIR:?}}"
PID_FILE="${{PID_FILE:?}}"
COMMAND="${{TPU_RUNNER_COMMAND:?}}"
GCLOUD_BIN="${{TPU_RUNNER_GCLOUD_BIN:?}}"
DEVICE_RELEASE_TIMEOUT_SECONDS={DEVICE_RELEASE_TIMEOUT_SECONDS}
ARTIFACT_UPLOAD_ATTEMPTS={ARTIFACT_UPLOAD_ATTEMPTS}
ARTIFACT_UPLOAD_RETRY_SECONDS={ARTIFACT_UPLOAD_RETRY_SECONDS}
printf '%s %s\n' "$ATTEMPT_ID" "$$" >"$PID_FILE"
artifact_error=""
upload_artifact() {{
  local source="$1"
  local destination="$2"
  local attempt output
  for ((attempt = 1; attempt <= ARTIFACT_UPLOAD_ATTEMPTS; attempt++)); do
    if output="$("$GCLOUD_BIN" storage cp "$source" "$destination" 2>&1)"; then
      if (( attempt > 1 )); then
        echo "[runner] artifact upload recovered: $destination on attempt $attempt/$ARTIFACT_UPLOAD_ATTEMPTS" >&2
      fi
      return 0
    fi
    echo "[runner] artifact upload attempt $attempt/$ARTIFACT_UPLOAD_ATTEMPTS failed: $destination: $output" >&2
    if (( attempt < ARTIFACT_UPLOAD_ATTEMPTS )); then
      sleep "$ARTIFACT_UPLOAD_RETRY_SECONDS"
    fi
  done
  artifact_error="${{artifact_error}}${{artifact_error:+; }}${{destination}}: ${{output}}"
  echo "[runner] artifact upload exhausted $ARTIFACT_UPLOAD_ATTEMPTS attempts: $destination" >&2
  return 1
}}
(
  set +e
  started_at="$(date -Is)"
  device_owner_pids() {{
    local owners=""
    if command -v lsof >/dev/null 2>&1; then
      owners="$(lsof -t /dev/accel* /dev/vfio/[0-9]* 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
      owners="$(fuser /dev/accel* /dev/vfio/[0-9]* 2>/dev/null || true)"
    fi
    printf '%s\n' "$owners" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' '
  }}
  device_release_deadline=$((SECONDS + DEVICE_RELEASE_TIMEOUT_SECONDS))
  code=0
  while true; do
    stale_device_owner_pids="$(device_owner_pids)"
    [[ -n "$stale_device_owner_pids" ]] || break
    if (( SECONDS >= device_release_deadline )); then
      echo "[runner] TPU device or resource busy after ${{DEVICE_RELEASE_TIMEOUT_SECONDS}}s: owner_pids=$stale_device_owner_pids" \
        >"$ATTEMPT_DIR/command-$HOST.log"
      code=75
      break
    fi
    echo "[runner] waiting for prior TPU device owners: $stale_device_owner_pids" \
      >"$ATTEMPT_DIR/command-$HOST.log"
    sleep 2
  done
  if (( code == 0 )); then
    setsid bash -lc "$COMMAND" >"$ATTEMPT_DIR/command-$HOST.log" 2>&1 &
    child_pid=$!
    printf '%s %s\n' "$ATTEMPT_ID" "$child_pid" >"$PID_FILE"
    (
      offset=0
      log_path="$ATTEMPT_DIR/command-$HOST.log"
      flush_cloud_log() {{
        [[ -n "${{RUNNER_PROJECT:-}}" && -f "$log_path" ]] || return 0
        local size start count chunk payload
        size="$(stat -c %s "$log_path" 2>/dev/null || echo 0)"
        [[ "$size" =~ ^[0-9]+$ ]] || return 0
        while (( offset < size )); do
          start="$offset"
          count=$((size - start))
          (( count > 60000 )) && count=60000
          chunk="$(dd if="$log_path" bs=1 skip="$start" count="$count" status=none 2>/dev/null || true)"
          [[ -n "$chunk" ]] || return 0
          payload="$(HOST="$HOST" LOG_OFFSET="$start" LOG_CHUNK="$chunk" python3 -c 'import json, os; print(json.dumps({{"job_id": os.environ.get("JOB_ID", ""), "attempt_id": os.environ.get("ATTEMPT_ID", ""), "tpu_name": os.environ.get("TPU_NAME", ""), "worker": os.environ.get("HOST", ""), "byte_offset": int(os.environ.get("LOG_OFFSET", "0")), "message": os.environ.get("LOG_CHUNK", "")}}, separators=(",", ":")))' 2>/dev/null || true)"
          [[ -n "$payload" ]] || return 0
          if ! "$GCLOUD_BIN" logging write {shlex.quote(self.log_name)} "$payload" \
            --payload-type=json \
            --severity=INFO \
            --project="$RUNNER_PROJECT" >/dev/null 2>&1; then
            return 0
          fi
          offset=$((start + count))
        done
      }}
      while kill -0 "$child_pid" 2>/dev/null; do
        flush_cloud_log
        sleep 10
      done
      flush_cloud_log
    ) &
    log_sync_pid=$!
    wait "$child_pid"
    code=$?
    wait "$log_sync_pid" 2>/dev/null || true
  fi
  printf '%s %s\n' "$ATTEMPT_ID" "$$" >"$PID_FILE"
  finished_at="$(date -Is)"
  state="succeeded"
  if (( code != 0 )); then
    state="failed"
    (( code == 2 )) && state="failed_setup"
    {{
      echo "host=$HOST"
      echo "started_at=$started_at"
      echo "finished_at=$finished_at"
      echo "exit_code=$code"
      df -h /
      df -h /dev/shm
      free -h || true
      command -v tpu-info >/dev/null 2>&1 && tpu-info || true
      ps -eo pid,ppid,pgid,stat,etime,cmd --sort=pid | tail -n 200 || true
      journalctl -n 200 --no-pager 2>/dev/null || true
      dmesg --ctime 2>/dev/null | tail -n 200 || true
    }} >"$ATTEMPT_DIR/diagnostics-$HOST.log" 2>&1
  fi
  upload_artifact "$ATTEMPT_DIR/command-$HOST.log" "$ATTEMPT_GCS_DIR/logs/$HOST.log" || true
  if [[ -f "$ATTEMPT_DIR/diagnostics-$HOST.log" ]]; then
    upload_artifact "$ATTEMPT_DIR/diagnostics-$HOST.log" "$ATTEMPT_GCS_DIR/diagnostics/$HOST.log" || true
  fi
  final_code="$code"
  if [[ -n "$artifact_error" && "$final_code" == 0 ]]; then
    final_code=2
    state="failed_artifacts"
  fi
  STATUS_HOST="$HOST" STATUS_STATE="$state" STATUS_EXIT_CODE="$final_code" \
    STATUS_COMMAND_EXIT_CODE="$code" STATUS_STARTED_AT="$started_at" STATUS_FINISHED_AT="$finished_at" \
    STATUS_ARTIFACT_ERROR="$artifact_error" python3 -c 'import json, os; print(json.dumps({{"host": os.environ["STATUS_HOST"], "state": os.environ["STATUS_STATE"], "exit_code": int(os.environ["STATUS_EXIT_CODE"]), "command_exit_code": int(os.environ["STATUS_COMMAND_EXIT_CODE"]), "started_at": os.environ["STATUS_STARTED_AT"], "finished_at": os.environ["STATUS_FINISHED_AT"], "artifact_upload_error": os.environ["STATUS_ARTIFACT_ERROR"]}}, separators=(",", ":")))' \
    >"$ATTEMPT_DIR/status-$HOST.json"
  status_upload_attempt=0
  until "$GCLOUD_BIN" storage cp "$ATTEMPT_DIR/status-$HOST.json" "$ATTEMPT_GCS_DIR/status/$HOST.json"; do
    status_upload_attempt=$((status_upload_attempt + 1))
    echo "[runner] terminal status publication attempt $status_upload_attempt failed: $HOST" >&2
    sleep "$ARTIFACT_UPLOAD_RETRY_SECONDS"
  done
  read -r current_attempt current_pid < "$PID_FILE" || true
  if [[ "$current_attempt" == "$ATTEMPT_ID" && ( "${{child_pid:-}}" == "$current_pid" || "$current_pid" == "$$" ) ]]; then
    rm -f "$PID_FILE"
  fi
  exit "$final_code"
)
TPU_RUNNER_WRAPPER
chmod +x "$ATTEMPT_DIR/run-wrapper.sh"
export PID_FILE ATTEMPT_DIR ATTEMPT_GCS_DIR
export TPU_RUNNER_GCLOUD_BIN="$GCLOUD_BIN"
export TPU_RUNNER_COMMAND={command}
nohup setsid "$ATTEMPT_DIR/run-wrapper.sh" 9>&- </dev/null >"$ATTEMPT_DIR/wrapper-$HOST.log" 2>&1 &
wrapper_pid=$!
printf '%s %s\n' "$CURRENT_ATTEMPT_ID" "$wrapper_pid" >"$PID_FILE"
flock -u 9
exec 9>&-
echo "TPU_RUNNER_LAUNCHED $HOST $wrapper_pid"
"""

    def worker_env(self, *, job: JobSpec, attempt: AttemptRecord, resource: ResourceRecord) -> dict[str, str]:
        if resource.worker_count < 1:
            raise ValueError("resource worker_count must be positive")
        bucket = job_bucket(job)
        job_dir = self.job_dir(job)
        return {
            "JOB_ID": job.id,
            "ATTEMPT_ID": attempt.id,
            "TPU_NAME": resource.tpu_name,
            "TPU_ZONE": resource.zone,
            "TPU_TYPE": resource.tpu_type,
            "TPU_WORKER_COUNT": str(resource.worker_count),
            "JOB_DIR": job_dir,
            "JOB_BUCKET": bucket,
            "JOB_SHM_DIR": f"{self.shm_root}/{job.id}",
            "JOB_GCS_DIR": f"{bucket}/jobs/{job.id}",
            "ATTEMPT_GCS_DIR": f"{bucket}/jobs/{job.id}/attempts/{attempt.id}",
            "CHECKPOINT_GCS_DIR": checkpoint_dir(bucket, job.id),
            "CACHE_ROOT": self.cache_root,
            "RUNNER_PROJECT": self.project or "",
            "BUNDLE_SHA256": bundle_sha256(job.bundle),
        }

    def cache_setup_script(self, caches: tuple[CacheSpec, ...]) -> str:
        declared = " ".join(shlex.quote(cache.key) for cache in caches)
        prepare_lines = []
        link_lines = []
        for cache in caches:
            key = shlex.quote(cache.key)
            path = shlex.quote(cache.path)
            prepare_lines.append(
                f"""
target="$CACHE_ROOT"/{key}
if [[ -e "$target" && ! -f "$target/.ready" ]]; then
  rm -rf "$target"
fi
"""
            )
            link_lines.append(
                f"""
target="$CACHE_ROOT"/{key}
mkdir -p "$target"
link="$JOB_DIR"/{path}
mkdir -p "$(dirname "$link")"
if [[ -L "$link" || -e "$link" ]]; then
  rm -rf "$link"
fi
ln -s "$target" "$link"
"""
            )
        return f"""
CACHE_ROOT={shlex.quote(self.cache_root)}
mkdir -p "$CACHE_ROOT"
for cache_dir in "$CACHE_ROOT"/*; do
  [[ -d "$cache_dir" ]] || continue
  keep=0
  for declared_key in {declared}; do
    [[ "$(basename "$cache_dir")" == "$declared_key" ]] && keep=1
  done
  [[ "$keep" == "1" ]] || rm -rf "$cache_dir"
done
{''.join(prepare_lines)}
minimum_free_kb="${{TPU_RUNNER_MIN_ROOT_FREE_KB:-{MINIMUM_ROOT_FREE_KB}}}"
if [[ ! "$minimum_free_kb" =~ ^[0-9]+$ ]] || (( minimum_free_kb < 1048576 )); then
  echo "[runner] invalid TPU_RUNNER_MIN_ROOT_FREE_KB=$minimum_free_kb; minimum=1048576" >&2
  exit 2
fi
free_kb="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo 0)"
while [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < minimum_free_kb )); do
  victim="$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | head -n 1 | cut -d' ' -f2-)"
  [[ -n "$victim" ]] || break
  echo "[runner] evicting cache for disk pressure: $victim"
  rm -rf "$victim"
  free_kb="$(df --output=avail / | tail -n 1 | tr -d ' ' || echo 0)"
done
if [[ ! "$free_kb" =~ ^[0-9]+$ ]] || (( free_kb < minimum_free_kb )); then
  echo "[runner] unsafe root disk: free_kb=$free_kb required_kb=$minimum_free_kb" >&2
  exit 2
fi
{''.join(link_lines)}
"""

    def job_dir(self, job: JobSpec) -> str:
        return f"{self.work_root}/{job.id}"

    def attempt_dir(self, job: JobSpec, attempt: AttemptRecord) -> str:
        return f"{self.work_root}/{job.id}/attempt-{attempt.id}"

    def run_tpu_vm_ssh_all(self, *, resource: ResourceRecord, script: str) -> subprocess.CompletedProcess[str]:
        ssh_user = os.environ.get("TPU_RUNNER_SSH_USER", "")
        ssh_private_key = os.environ.get("TPU_RUNNER_SSH_PRIVATE_KEY", "")
        if bool(ssh_user) != bool(ssh_private_key):
            raise ValueError("TPU_RUNNER_SSH_USER and TPU_RUNNER_SSH_PRIVATE_KEY must be set together")
        target = resource.tpu_name
        if ssh_user:
            target = f"{ssh_user}@{target}"
            key_path = Path("/tmp/tpu-runner-ssh-key")
            key_path.write_text(ssh_private_key)
            key_path.chmod(0o600)
            public_key = subprocess.check_output(["ssh-keygen", "-y", "-f", str(key_path)], text=True)
            key_path.with_suffix(".pub").write_text(f"{public_key.strip()} {ssh_user}\n")
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            target,
            f"--zone={resource.zone}",
            "--worker=all",
            "--quiet",
            f"--command={encoded_bash_command(script)}",
        ]
        if self.ssh_transport == "iap":
            command.insert(-2, "--tunnel-through-iap")
        elif self.ssh_transport != "direct":
            raise ValueError(f"unsupported SSH transport: {self.ssh_transport!r}")
        if ssh_user:
            command.insert(-1, f"--ssh-key-file={key_path}")
        if self.project:
            command.insert(7, f"--project={self.project}")
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=SSH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            # gcloud fans one all-worker request out to child SSH processes.
            # Killing only the parent can leave descendants holding stdout or
            # stderr open forever, so bound the local IAP command's exact
            # process group. This does not signal any remote process.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Never call communicate() without a bound here. A descendant
                # that escaped gcloud's process group may still hold an
                # inherited pipe open even after the parent was reaped.
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise TemporaryAccessError(
                f"tpu ssh timed out after {SSH_TIMEOUT_SECONDS}s"
            ) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

def encoded_bash_command(script: str) -> str:
    payload = base64.b64encode(script.encode()).decode()
    return f"printf %s {shlex.quote(payload)} | base64 -d | bash"


def looks_like_temporary_access_error(output: str) -> bool:
    lowered = output.lower()
    permanent_markers = (
        "permission denied",
        "not authorized",
    )
    if any(marker in lowered for marker in permanent_markers):
        return False
    markers = (
        "error while connecting",
        "failed to connect to backend",
        "connection timed out",
        "connection reset",
        "connection refused",
        "ssh: connect",
        "could not resolve",
        "temporary failure",
        "unavailable",
        "device or resource busy",
        'this tpu has terminal state "preempted"',
        'this tpu has terminal state "terminated"',
        'this tpu has terminal state "stopped"',
        # A Spot TPU can enter a terminal transition after inventory marks it
        # ready but before the controller's all-worker SSH launch reaches it.
        # Treat these exact lifecycle races as temporary access failures so
        # reconciliation can mark the resource unavailable and requeue the
        # same scientific job instead of terminally failing setup.
        'this tpu has state "deleting"',
        'this tpu has state "suspending"',
    )
    return any(marker in lowered for marker in markers)


def bundle_sha256(uri: str) -> str:
    match = re.search(r"(?:^|/)bundles/([0-9a-f]{64})\.tar\.gz$", uri)
    return match.group(1) if match else ""
