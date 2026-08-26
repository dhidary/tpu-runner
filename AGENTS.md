# TPU Runner agent guide

## Project intent

Keep this repository a small, understandable TPU job runner. The public surface
is the `tpu-runner` CLI, a deployment YAML file, and a job YAML file. Prefer
direct code and targeted validation over frameworks, background services, CI
workflows, or large test scaffolds unless the user explicitly asks for them.

The user-facing README should stay short and conceptual. Put detailed
architecture, invariants, race analysis, and maintainer procedures here.

## Module map

- `tpu_runner/specs.py`: immutable fleet/job/cache models and YAML validation.
- `tpu_runner/placement.py`: resolves regional job buckets and rejects literal
  `gs://` dependencies that would defeat local placement.
- `tpu_runner/capacity_policy.py`: pure scheduling, stable priority, compatibility,
  and desired-capacity policy.
- `tpu_runner/runtime.py`: Firestore records, serialization, leases, and all
  multi-record transactional state transitions.
- `tpu_runner/gcp.py`: exact-name TPU and queued-resource descriptions and
  mutations through `gcloud`.
- `tpu_runner/controller.py`: ordered reconciliation of inventory, jobs,
  attempts, cancellations, interruptions, preemption, and capacity.
- `tpu_runner/distributed.py`: all-worker SSH, remote launch/cancel protocol,
  GCS status polling, logs, diagnostics, caches, and process identity.
- `tpu_runner/cli.py`: CLI parsing, submission, observation, deployment entry,
  controller loop, and leader-lease renewal.
- `tpu_runner/deploy.sh`: GCP control-plane provisioning and controller rollout.
- `tpu_runner/startup.sh`: TPU VM user, SSH, scratch, linger, and readiness setup.
- `tpu_runner/Dockerfile` and `cloudbuild.yaml`: controller image build.

## Deployment lifecycle

`tpu-runner deploy deployment.yaml` validates the fleet, then `deploy.sh`:

1. Enables required APIs.
2. Creates the derived `us-central2` runner bucket, or an explicitly named
   runner bucket, if absent.
3. Creates the default Firestore database if absent.
4. Creates controller and worker service accounts.
5. Adds controller roles (`datastore.user`, `compute.viewer`,
   `iam.serviceAccountUser`, `storage.objectViewer`, `tpu.admin`, plus IAP when
   configured).
6. Adds worker roles (`logging.logWriter`, `storage.objectAdmin`) in the runner
   project and grants access to declared pre-existing worker secrets.
7. Creates or reads the runner SSH private key in Secret Manager.
8. Renders and uploads the worker startup script to
   `BUCKET/artifacts/startup.sh`.
9. Uploads the deployment spec to `specs/deployment.yaml`.
10. Stages a temporary package-local build context and builds the controller.
11. Sets a new controller epoch, cancels older executions, waits for the old
    lease to release or expire, deploys the new Cloud Run job, and executes it.

`worker_secrets` contains existing Secret Manager names in the runner project,
never secret values. Deployment grants the worker service account accessor
permission but does not inject values into jobs; workloads retrieve them at
runtime. Job environment values are materialized in Firestore and GCS and must
not contain secrets.

The package-local build context is required for PyPI installations. Do not make
deployment depend on repository-root files that are absent from an installed
wheel. Keep `controller-requirements.txt` aligned with imports needed inside the
controller image.

The packaged `tpu_runner/deployment.example.yaml` is the deployment template.
`tpu-runner init` copies it into the user's working directory.

## Submission lifecycle

`submit_jobs` performs all fallible preparation before publishing jobs:

1. Load the manifest and generate IDs for unnamed jobs.
2. Select requested manifest IDs before external work.
3. Validate TPU name/type/zone compatibility against declared fleet ordinals.
4. Resolve every declared job bucket to one distinct exact region.
5. Reject multi-region jobs with literal GCS references; they must select
   mirrored data through `JOB_BUCKET`.
6. Reproducibly archive local bundles, hash content, and upload to every
   candidate regional bucket only if absent.
7. Upload a uniquely named materialized job spec to every candidate bucket.
8. Atomically create the full set of Firestore job records.
9. Record events and trigger the Cloud Run controller in a `finally` block.

Artifacts may exist without a Firestore job if preparation or atomic creation
fails. That is harmless. A Firestore job must never be visible before its
bundle and materialized spec exist.

## Controller reconciliation order

`Controller.reconcile_once` deliberately runs in this order:

1. Renew the lease and describe exact current/previously known resource names.
2. Refresh queued-resource records and ready TPU records.
3. Recover runner-owned scratch pressure on eligible idle adopted TPUs.
4. Process job cancellations.
5. Process exact controlled-interruption requests.
6. Poll or launch controller-managed attempts.
7. Recycle managed Spot ordinals scheduled after repeated infrastructure or
   cancellation failures.
8. Reconcile missing, terminal, transport-incompatible, or unhealthy resources.
9. Assign pending jobs to already-idle compatible resources.
10. Reconcile desired managed capacity and issue exact creates/deletes.
11. Re-read controlled interruptions.
12. Poll or launch attempts again so assignments from step 9 can start without
    waiting for another cycle.

Do not reorder these phases casually. Examples:

- Interruptions precede attempt polling so a poisoned TPU is not terminalized
  and reassigned before its deletion request is consumed.
- Assignment precedes capacity creation so ready TPUs satisfy demand first.
- The second attempt pass starts newly assigned work in the same cycle.

The controller loop sleeps 30 seconds between completed active cycles. Remote
operations make actual cycle time longer. A background renewer refreshes the
15-minute Firestore lease every 60 seconds while the main thread blocks; normal
phase heartbeats remain synchronous. Stop and join the renewer before releasing
the lease so it cannot reacquire after an intentional handoff.

## State and transaction invariants

Primary Firestore records:

- Job: spec, submission time, status, assigned resource, current attempt.
- Attempt: exact job/resource pair, lifecycle timestamps, exit, failure kind,
  and terminal reason.
- Resource: exact TPU identity, fleet entry, adoption flag, status, reciprocal
  job/attempt ownership, and bounded failure counters.
- Interruption request: exact resource/job/attempt/fleet-entry target and claim
  state.
- Lock and lease epoch: controller ownership and deployment-generation fence.

Important invariants:

- Job creation is an all-or-nothing Firestore transaction.
- First assignment atomically selects the winning regional bucket and bundle
  while changing a pending job, idle resource, and new launching attempt. One
  job cannot run twice and one TPU cannot own two attempts.
- Job, resource, and attempt store reciprocal exact IDs while work is active.
- `finish_attempt` compares job status and exact current ownership before
  changing all three records atomically.
- `cancel --if-pending` and reprioritization conflict instead of mutating a job
  whose assignment won the race.
- Interruption request creation and claim both revalidate the exact live managed
  Spot attempt.
- Remote process actions require the exact attempt ID stored beside the PID.
- Deploy epochs prevent an older controller generation from renewing.

Inventory upserts and event records are intentionally simple non-transactional
writes. Do not move ownership transitions out of their existing transactions.

## Capacity and scheduling policy

Managed entry `count` is a hard ceiling, not a standing request. Desired count
is busy resources plus pending demand, capped by the entry count, with at least
`keep_warm_count` TPUs retained in each managed entry. Chip limits are enforced
again before creation using all reserved TPU/queued-resource chips for the same
model, family, and zone. This is a runner-side ceiling, not GCP quota discovery
or a quota request. TPU v4 accelerator names count TensorCores, so validation
and capacity accounting divide the v4 type suffix by two to obtain physical
chips.

Managed TPUs record `idle_since` when an attempt releases them or when newly
ready capacity first becomes idle. Inventory refreshes must preserve that
timestamp. Undesired capacity is deleted after `idle_timeout_seconds`;
`keep_warm_count` is an indefinite floor that overrides idle scale-down.

Adopted idle capacity is matched before managed demand. Compatible unpinned
jobs may use it automatically; `tpu_name` remains an exact resource pin.

Pending jobs are ordered by:

1. Explicit priority (`high`, `normal`, `low`).
2. Fewest compatible entries/resources.
3. Submission time.
4. Job ID.

Idle-resource assignment uses deterministic augmenting paths. An earlier job
may move to another compatible idle TPU so a later constrained job can also
run, but it is never dropped from the batch to make room for that later job.

Priority never changes automatically. It can only be changed while a job is
pending and unassigned. Reprioritization is transactional with assignment.

Each pending job creates demand in every compatible managed entry with a free
slot and a matching declared bucket region. This intentionally races Spot pools
across zones and regions. Firestore assignment is the single execution
boundary; it selects the local bucket, and losing capacity disappears from
desired demand on the next cycle and drains when idle. Retries remain pinned to
the first winning region and bucket.

Generated queued-resource and TPU names are stable deployment ordinals. Never
substitute undeclared ordinals just to satisfy a count; exact TPU pins depend on
these names remaining stable.

## GCS and placement invariants

The deployment bucket contains only runner infrastructure:

```text
artifacts/startup.sh
specs/deployment.yaml
```

Every candidate regional job bucket is prepared before Firestore publication:

```text
bundles/<sha256>.tar.gz
jobs/<job>/spec-<submission-token>.json
jobs/<job>/attempts/<attempt>/logs/<host>.log
jobs/<job>/attempts/<attempt>/diagnostics/<host>.log
jobs/<job>/attempts/<attempt>/status/<host>.json
jobs/<job>/checkpoints/
```

Pending jobs store one resolved bucket and bundle per candidate exact region,
with no selected bucket or region. Multi-region jobs cannot contain literal
GCS references in the source bundle URI, command, or env; applications use
`JOB_BUCKET` to address the mirrored local data. First assignment selects the
resource's exact region, bucket, and bundle in the same Firestore transaction.
Assignment fails closed if the mapping is missing or inconsistent. Never relax
the selected pin after preemption.

The selected job bucket becomes the home for statuses, logs, diagnostics, and
checkpoints. Deployment does not create or grant IAM on regional job buckets.

## Worker execution protocol

The startup script runs as root and:

- Creates the `tpurunner` user.
- Enables systemd linger.
- Installs the rendered SSH public key.
- Creates runner-owned work, cache, process, and shared-memory roots.
- Writes a readiness marker tied to the current boot ID.

The controller SSHes to all workers. Each launch:

1. Verifies the current-boot readiness marker and an absolute executable
   `gcloud` path.
2. Takes a per-worker `flock` for launch serialization.
3. Returns idempotently for an already-running or already-complete exact attempt.
4. Stops only an exact stale runner process group recorded in the runner PID file.
5. Clears non-cache work and shared memory.
6. Downloads/extracts the bundle, discards undeclared or unready caches, and
   attaches declared local cache keys.
7. Waits for stale TPU device owners to release.
8. Starts the command in its own process group and streams chunks to Cloud
   Logging.
9. Uploads logs, diagnostics, and terminal status JSON to GCS.

The same command runs independently on every worker. `JOB_BUCKET` is the bucket
selected for the winning TPU region. Cache directories are local
to one worker and become reusable only after the job creates `.ready` beneath
the cache key in `CACHE_ROOT`. They are disposable and may be evicted for disk
pressure. `CHECKPOINT_GCS_DIR` is stable and durable, but the application owns
checkpoint format, creation, and restore.

Exit code 75 is reserved for retryable infrastructure failure. It is accepted
as distributed infrastructure loss only when sibling exit codes are within the
small allowed fallout set. Application failures must not be broadly reclassified
as retryable.

## Resource safety boundaries

- Inventory describes only exact declared/generated names plus exact known
  managed records. Do not introduce broad zone listing or cleanup.
- Managed deletes require exact fleet labels, type, zone, model, ordinal, and
  queued-resource/node relationships.
- Adopted TPUs are never created, stopped, recycled, or deleted.
- Capacity reduction and zero demand never delete a busy resource.
- Idempotent delete accepts only a precise GCP not-found result; permission and
  identity failures must propagate.
- Cancellation targets an exact attempt-qualified process group.
- Do not use broad remote process kills.

## TPU operations

When polling, launching, or managing a fleet configured for IAP, use IAP SSH:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$TPU_ZONE" \
  --tunnel-through-iap
```

Do not use broad remote kill commands such as `pkill` when managing TPU jobs;
they can kill the IAP SSH tunnel. If an IAP SSH session hangs, interrupt the
local `gcloud` command with Ctrl-C and reconnect. Only stop remote training
processes with narrow, verified commands when there is an actionable failure.

## Verification

Keep verification proportional and local. Useful release checks:

```bash
python -m tpu_runner --version
python -m tpu_runner --help
python -m tpu_runner validate-fleet tpu_runner/deployment.example.yaml
bash -n tpu_runner/deploy.sh tpu_runner/startup.sh
python -m build
python -m twine check dist/*
```

Also install the built wheel into a new temporary environment, run
`tpu-runner init`, validate the generated deployment, and confirm these package
resources exist: Dockerfile, cloudbuild YAML, controller requirements, deploy
script, deployment template, and startup script.

The Docker daemon may be unavailable locally. Do not claim the image was built
when only Python artifacts were verified.

Keep generated `build/`, `dist/`, egg-info, manifests, logs, and live campaign
configuration out of commits unless explicitly requested.

## Git workflow

Commit and push relevant reusable core changes as work progresses. Keep tests,
temporary artifacts, generated manifests, live campaign configuration, logs,
and campaign-specific state out of commits unless explicitly requested.

Preserve unrelated user changes. In particular, do not stage or commit the
automatically added `# Hillclimb hook files` block in `.gitignore` unless the
user explicitly asks for it.
