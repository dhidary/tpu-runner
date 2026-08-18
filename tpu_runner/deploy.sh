#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOYMENT="${TPU_RUNNER_DEPLOYMENT_PATH:?run deployment through tpu-runner deploy}"
NAME="${TPU_RUNNER_NAME:?}"
PROJECT="${TPU_RUNNER_PROJECT:?}"
BUCKET="${TPU_RUNNER_BUCKET:?}"
BUCKET_LOCATION="${TPU_RUNNER_BUCKET_LOCATION:?}"
REGION="${TPU_RUNNER_CONTROLLER_REGION:?}"
CONTROLLER_TIMEOUT="${TPU_RUNNER_CONTROLLER_TIMEOUT:?}"
CONTROLLER_MEMORY="${TPU_RUNNER_CONTROLLER_MEMORY:?}"
CONTROLLER_MAX_RETRIES="${TPU_RUNNER_CONTROLLER_MAX_RETRIES:?}"
SSH_TRANSPORT="${TPU_RUNNER_SSH_TRANSPORT:?}"
FIRESTORE_LOCATION="${TPU_RUNNER_FIRESTORE_LOCATION:?}"
NETWORK="${TPU_RUNNER_NETWORK:?}"
WORKER_SECRETS="${TPU_RUNNER_WORKER_SECRETS:-}"
PYTHON="${TPU_RUNNER_PYTHON:?}"

JOB_NAME="${NAME}-controller"
CONTROLLER_SA_NAME="${NAME}-controller"
CONTROLLER_SA_EMAIL="${CONTROLLER_SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
WORKER_SA_NAME="${NAME}-worker"
WORKER_SA_EMAIL="${WORKER_SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
SSH_SECRET="${NAME}-ssh-key"
FIREWALL_RULE="${NAME}-iap-ssh"
DEPLOYMENT_URI="${BUCKET%/}/specs/deployment.yaml"
STARTUP_SCRIPT_URI="${BUCKET%/}/artifacts/startup.sh"
CONTROLLER_EPOCH="${TPU_RUNNER_CONTROLLER_EPOCH:?}"
# A unique tag makes the deployed source generation auditable and prevents a
# later build from changing what this Cloud Run job revision means.
IMAGE="gcr.io/${PROJECT}/${NAME}-controller:${CONTROLLER_EPOCH}"

SSH_KEY_FILE="$(mktemp)"
RENDERED_STARTUP="$(mktemp)"
BUILD_CONTEXT="$(mktemp -d)"
cleanup() {
  rm -f "$SSH_KEY_FILE" "$SSH_KEY_FILE.pub" "$RENDERED_STARTUP"
  rm -rf "$BUILD_CONTEXT"
}
trap cleanup EXIT
chmod 600 "$SSH_KEY_FILE"
cp -R "$SCRIPT_DIR" "$BUILD_CONTEXT/tpu_runner"

REQUIRED_SERVICES=(
  artifactregistry.googleapis.com
  cloudbuild.googleapis.com
  compute.googleapis.com
  firestore.googleapis.com
  logging.googleapis.com
  run.googleapis.com
  secretmanager.googleapis.com
  tpu.googleapis.com
)
if [[ "$SSH_TRANSPORT" == "iap" ]]; then
  REQUIRED_SERVICES+=(iap.googleapis.com)
fi
gcloud services enable "${REQUIRED_SERVICES[@]}" --project "$PROJECT"

if ! gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" \
    --project "$PROJECT" \
    --location "$BUCKET_LOCATION" \
    --uniform-bucket-level-access
fi

if [[ "$SSH_TRANSPORT" == "iap" ]] && \
    ! gcloud compute firewall-rules describe "$FIREWALL_RULE" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --project "$PROJECT" \
    --network "$NETWORK" \
    --direction ingress \
    --action allow \
    --rules tcp:22 \
    --source-ranges 35.235.240.0/20
fi

if ! gcloud firestore databases describe --database="(default)" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud firestore databases create \
    --database="(default)" \
    --location="$FIRESTORE_LOCATION" \
    --project "$PROJECT"
fi

for account in "$CONTROLLER_SA_NAME" "$WORKER_SA_NAME"; do
  email="${account}@${PROJECT}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$email" --project "$PROJECT" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$account" --project "$PROJECT"
  fi
done

CONTROLLER_ROLES=(
  roles/datastore.user
  roles/compute.viewer
  roles/iam.serviceAccountUser
  roles/storage.objectViewer
  roles/tpu.admin
)
if [[ "$SSH_TRANSPORT" == "iap" ]]; then
  CONTROLLER_ROLES+=(roles/iap.tunnelResourceAccessor)
fi
for role in "${CONTROLLER_ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${CONTROLLER_SA_EMAIL}" --role "$role" --quiet >/dev/null
done

for role in roles/logging.logWriter roles/storage.objectAdmin; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${WORKER_SA_EMAIL}" --role "$role" --quiet >/dev/null
done

if gcloud secrets describe "$SSH_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud secrets versions access latest --secret "$SSH_SECRET" --project "$PROJECT" > "$SSH_KEY_FILE"
else
  rm -f "$SSH_KEY_FILE"
  ssh-keygen -q -t ed25519 -N '' -C tpurunner -f "$SSH_KEY_FILE"
  gcloud secrets create "$SSH_SECRET" \
    --project "$PROJECT" --replication-policy=automatic --data-file="$SSH_KEY_FILE"
fi
gcloud secrets add-iam-policy-binding "$SSH_SECRET" \
  --project "$PROJECT" \
  --member "serviceAccount:${CONTROLLER_SA_EMAIL}" \
  --role roles/secretmanager.secretAccessor \
  --quiet >/dev/null

while IFS= read -r secret; do
  [[ -z "$secret" ]] && continue
  gcloud secrets describe "$secret" --project "$PROJECT" >/dev/null
  gcloud secrets add-iam-policy-binding "$secret" \
    --project "$PROJECT" \
    --member "serviceAccount:${WORKER_SA_EMAIL}" \
    --role roles/secretmanager.secretAccessor \
    --quiet >/dev/null
done <<< "$WORKER_SECRETS"

SSH_PUBLIC_KEY="$(ssh-keygen -y -f "$SSH_KEY_FILE") tpurunner"
sed \
  -e "s|__TPU_RUNNER_SSH_PUBLIC_KEY__|${SSH_PUBLIC_KEY}|" \
  -e "s|/tmp/tpu-runner|/tmp/${NAME}|g" \
  -e "s|/dev/shm/tpu-runner|/dev/shm/${NAME}|g" \
  -e "s|/var/lib/tpu-runner|/var/lib/${NAME}|g" \
  "$SCRIPT_DIR/startup.sh" > "$RENDERED_STARTUP"

gcloud storage cp "$RENDERED_STARTUP" "$STARTUP_SCRIPT_URI" --project "$PROJECT"
gcloud storage cp "$DEPLOYMENT" "$DEPLOYMENT_URI" --project "$PROJECT"

"$PYTHON" -m tpu_runner.cli bootstrap-ready \
  --deployment "$DEPLOYMENT" \
  --startup "$RENDERED_STARTUP"

gcloud builds submit \
  --project "$PROJECT" \
  --config "$SCRIPT_DIR/cloudbuild.yaml" \
  --substitutions "_IMAGE=${IMAGE}" \
  "$BUILD_CONTEXT"

"$PYTHON" -m tpu_runner.cli set-controller-epoch \
  --deployment "$DEPLOYMENT" \
  --epoch "$CONTROLLER_EPOCH"

# Cancel visible old executions before updating the job. Updating first can
# make an older execution disappear from the API while its container continues
# renewing the Firestore lease with an obsolete fleet specification.
while IFS= read -r execution; do
  [[ -z "$execution" ]] && continue
  cancel_status=0
  gcloud run jobs executions cancel "$execution" \
    --project "$PROJECT" \
    --region "$REGION" \
    --quiet || cancel_status=$?
  # Release only after Cloud Run proves this exact owner is terminal.  If the
  # execution is missing or cancellation is still propagating, retain the
  # lease and let the bounded wait below fail closed. A cancellation command
  # can itself return non-zero when the container becomes terminal while that
  # request is in flight; the exact terminal describe is authoritative.
  if gcloud run jobs executions describe "$execution" \
      --project "$PROJECT" \
      --region "$REGION" \
      --format json |
    "$PYTHON" -c '
import json
import sys

execution = json.load(sys.stdin)
status = execution.get("status", {})
completed = next(
    (
        condition
        for condition in status.get("conditions", [])
        if condition.get("type") == "Completed"
    ),
    None,
)
terminal = completed is not None and completed.get("status") in {"True", "False"}
raise SystemExit(0 if terminal else 1)
'; then
    "$PYTHON" -m tpu_runner.cli release-controller-lease \
      --deployment "$DEPLOYMENT" \
      --owner "$execution"
  elif (( cancel_status != 0 )); then
    printf 'Controller cancellation failed and execution is not terminal: %s\n' "$execution" >&2
    exit "$cancel_status"
  fi
done < <(
  gcloud run jobs executions list \
    --job "$JOB_NAME" \
    --project "$PROJECT" \
    --region "$REGION" \
    --format json |
  "$PYTHON" -c '
import json
import sys

for execution in json.load(sys.stdin):
    status = execution.get("status", {})
    completed = next(
        (
            condition
            for condition in status.get("conditions", [])
            if condition.get("type") == "Completed"
        ),
        None,
    )
    # Cloud Run reports cancelled and failed executions as Completed=False and
    # may omit completionTime for them. Both True and False are terminal;
    # Unknown (or a missing condition) still represents an active execution.
    if completed is not None and completed.get("status") in {"True", "False"}:
        continue
    name = execution.get("metadata", {}).get("name")
    if name:
        print(name)
'
)

# Setting the new epoch fences every old controller from renewing.  Do not
# release its lease optimistically: cancellation can return before the old
# container has stopped, and that container may already be inside a mutating
# reconcile pass.  Waiting for the lease to be released or expire prevents an
# old and a new fleet specification from acting concurrently.
"$PYTHON" -m tpu_runner.cli wait-controller-release \
  --deployment "$DEPLOYMENT" \
  --timeout-seconds 1200 \
  --poll-seconds 5

gcloud run jobs deploy "$JOB_NAME" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --service-account "$CONTROLLER_SA_EMAIL" \
  --task-timeout "$CONTROLLER_TIMEOUT" \
  --memory "$CONTROLLER_MEMORY" \
  --max-retries "$CONTROLLER_MAX_RETRIES" \
  --args controller \
  --set-env-vars "TPU_RUNNER_DEPLOYMENT=${DEPLOYMENT_URI},TPU_RUNNER_SSH_USER=tpurunner,TPU_RUNNER_CONTROLLER_EPOCH=${CONTROLLER_EPOCH}" \
  --set-secrets "TPU_RUNNER_SSH_PRIVATE_KEY=${SSH_SECRET}:latest"

gcloud run jobs execute "$JOB_NAME" \
  --project "$PROJECT" \
  --region "$REGION" \
  --async

printf 'Controller job: %s\nDeployment:   %s\n' "$JOB_NAME" "$DEPLOYMENT_URI"
