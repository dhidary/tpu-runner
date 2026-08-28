# TPU Runner

`tpu-runner` is a job orchestrator designed to **maximize utilization of your Google Cloud TPU allocation.** 

It **races capacity** requests across compatible TPU types and regions, **assigns jobs** efficiently, **retries jobs** interrupted by Spot preemptions, and **automatically scales** capacity with demand. It also minimizes inter-region transfer costs, prepares a clean workspace for new jobs, and lets related jobs reuse local caches (e.g. for previously compiled XLA artifacts or software environments).

We use Firestore for queue and orchestration state, GCS for source bundles and job artifacts, and a Cloud Run controller to manage TPU capacity and execution.

## Install

Install `tpu-runner` as a standalone CLI:

```bash
uv tool install tpu-runner
```

For local development, run this from the repository root:

```bash
uv tool install --editable .
```

Configure `gcloud`:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## Set up the runner

Create an example deployment:

```bash
tpu-runner init
```

Edit `deployment.yaml` with your project ID, existing Secret Manager secret names, and the TPU types, zones, maximum counts, runtime versions, and chip limits you are willing to use. Counts are provisioning ceilings and TPU Runner scales capacity with demand. Managed TPUs remain available for `idle_timeout_seconds` after their final job (600 seconds by default); `keep_warm_count` retains that many TPUs in a Spot pool indefinitely.

The default `ssh_transport: direct` creates public-IP TPUs; set it to `iap` to create private-IP TPUs and reach them through IAP tunnels instead.

Validate and deploy:

```bash
tpu-runner validate-fleet deployment.yaml
tpu-runner deploy deployment.yaml
```

`deploy` enables the required APIs and creates or updates the runner bucket, Firestore database, service accounts and IAM, SSH identity, worker startup script, controller image, and Cloud Run controller job.

## Submit work

In the root of the code you want to run, create `job.yaml`:

```yaml
jobs:
  - tpu: [v4-32, v6e-64]
    buckets:
      - gs://my-training-us-central2
      - gs://my-training-us-east1
    bundle: .
    priority: high
    caches:
      - key: pip
        path: .cache/pip
    env:
      PIP_CACHE_DIR: .cache/pip
      WANDB_PROJECT: my-project
    command: python3 -m pip install -r requirements.txt && touch "$PIP_CACHE_DIR/.ready" && python3 train.py --data "$JOB_BUCKET/data" --checkpoints "$CHECKPOINT_GCS_DIR"
```

- `tpu` may contain one or several compatible TPU types to race.
- Omit `zone` and `tpu_name` to race all compatible fleet capacity. Set `zone`
  to use one zone, or `tpu_name` to use one exact declared TPU.
- Create one listed bucket in each candidate region and mirror required data at
  the same object paths. For a single-region job, list one bucket. The winning
  region's bucket becomes `JOB_BUCKET`, and retries remain pinned to that
  region.
- `bundle` is a local directory relative to `job.yaml`. TPU Runner archives and
  uploads it; use `.tpu-runnerignore` to exclude files.
- `priority` may be `low`, `normal`, or `high`.
- Before a new job starts, TPU Runner clears the previous runner workspace, 
  then extracts the new bundle into a fresh directory. A directory
  declared under `caches` is preserved when it is marked `.ready` and the next job
  on that worker declares the same `key`. Configure the relevant tool, such as
  pip, to write to the cache path. Incomplete caches are discarded, and all
  caches disappear when the TPU is deleted.
- `command` runs independently on every TPU worker. Other runner-provided
  variables include `JOB_ID`, `ATTEMPT_ID`, `TPU_WORKER_HOST`,
  `TPU_WORKER_COUNT`, `JOB_DIR`, and `ATTEMPT_GCS_DIR`.

Submit and watch the job:

```bash
tpu-runner validate-jobs job.yaml
tpu-runner submit job.yaml
tpu-runner watch JOB_ID
tpu-runner logs JOB_ID
tpu-runner cancel JOB_ID
```

Spot preemption and infrastructure failures return a job to
`pending` with the same region and checkpoint directory. Application and setup
failures are terminal. TPU Runner schedules higher-priority jobs first. Within
each priority it considers the most constrained jobs first and moves flexible
jobs to alternative idle TPUs when that allows more jobs to run.
TPU Runner stores job bundles, logs, diagnostics, status, and checkpoints in the job’s GCS bucket, 
while your application data remains in the GCS buckets you provide.

Use `tpu-runner --help` to see all commands and `tpu-runner COMMAND --help` for command-specific guidance.

PRs and feature requests are welcome. Thank you to the [Google TPU Research Cloud (TRC) program](https://sites.research.google/trc/) for inspiring this work.
