#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT=/var/lib/tpu-runner
STARTUP_READY_MARKER="$STATE_ROOT/startup-ready"
install -d -m 0755 "$STATE_ROOT"
rm -f "$STARTUP_READY_MARKER"

id -u tpurunner >/dev/null 2>&1 || useradd -m -s /bin/bash tpurunner
# Install a fixed read-only checker, never grant arbitrary Python/root execution.
install -d -m 0755 -o root -g root /usr/local/libexec
probe_tmp="$(mktemp /usr/local/libexec/.tpu-runner-probe.XXXXXX)"
sudoers_tmp="$(mktemp)"
trap 'rm -f "$probe_tmp" "$sudoers_tmp"' EXIT
printf '%s' '__TPU_RUNNER_WORKER_PROBE_B64__' | base64 --decode > "$probe_tmp"
install -m 0755 -o root -g root "$probe_tmp" /usr/local/libexec/tpu-runner-attempt-stopped
printf '%s\n' 'tpurunner ALL=(root) NOPASSWD: /usr/local/libexec/tpu-runner-attempt-stopped *' > "$sudoers_tmp"
visudo -cf "$sudoers_tmp"
install -m 0440 -o root -g root "$sudoers_tmp" /etc/sudoers.d/tpu-runner-attempt-stopped
runuser -u tpurunner -- sudo -n -- /usr/local/libexec/tpu-runner-attempt-stopped startup-check >/dev/null
if ! command -v loginctl >/dev/null 2>&1; then
  echo "tpu-runner requires loginctl to keep the runner user manager alive" >&2
  exit 1
fi
loginctl enable-linger tpurunner
linger_enabled="$(loginctl show-user tpurunner --property=Linger --value 2>/dev/null || true)"
if [[ "$linger_enabled" != "yes" ]]; then
  echo "tpu-runner could not enable systemd linger for tpurunner" >&2
  exit 1
fi
install -d -m 700 -o tpurunner -g tpurunner /home/tpurunner/.ssh
printf '%s\n' '__TPU_RUNNER_SSH_PUBLIC_KEY__' \
  > /home/tpurunner/.ssh/authorized_keys
chown tpurunner:tpurunner /home/tpurunner/.ssh/authorized_keys
chmod 600 /home/tpurunner/.ssh/authorized_keys
install -d -m 0755 -o tpurunner -g tpurunner \
  /tmp/tpu-runner \
  /tmp/tpu-runner/work \
  /tmp/tpu-runner/cache \
  /tmp/tpu-runner/process \
  /dev/shm/tpu-runner
install -d -m 1777 /tmp/tpu_logs
chown -R tpurunner:tpurunner /tmp/tpu-runner /dev/shm/tpu-runner

thp_enabled=/sys/kernel/mm/transparent_hugepage/enabled
if [[ -w "$thp_enabled" ]]; then
  if printf 'always\n' >"$thp_enabled"; then
    echo "tpu-runner enabled transparent hugepages"
  else
    echo "tpu-runner could not enable transparent hugepages" >&2
  fi
fi

boot_id="$(cat /proc/sys/kernel/random/boot_id)"
ready_tmp="$(mktemp "$STATE_ROOT/.startup-ready.XXXXXX")"
printf 'schema_version=3\nboot_id=%s\nlinger_enabled=yes\ncleanup_probe=ready\nready_at=%s\n' \
  "$boot_id" "$(date -Is)" >"$ready_tmp"
chmod 0644 "$ready_tmp"
mv -f "$ready_tmp" "$STARTUP_READY_MARKER"
echo "tpu-runner scratch roots ready on $(hostname)"
