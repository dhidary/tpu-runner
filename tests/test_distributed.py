from __future__ import annotations

import subprocess
import unittest
from datetime import datetime, timedelta, timezone

from tpu_runner.distributed import (
    SLICE_FAILURE_CHIP_DRIVER_ERROR,
    DistributedTPURunner,
    aggregate_worker_statuses,
)
from tpu_runner.runtime import AttemptRecord, ResourceRecord
from tpu_runner.specs import JobSpec


class DistributedStatusAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 3, 5, 2, 54, tzinfo=timezone.utc)

    def status(
        self,
        code: int,
        *,
        seconds_ago: int = 0,
        infrastructure_failure: str = "",
    ) -> dict:
        return {
            "state": "failed" if code else "succeeded",
            "exit_code": code,
            "command_exit_code": code,
            "finished_at": (self.now - timedelta(seconds=seconds_ago)).isoformat(),
            "artifact_upload_error": "",
            "infrastructure_failure": infrastructure_failure,
        }

    def test_partial_application_failure_waits_for_sibling_statuses(self) -> None:
        result = aggregate_worker_statuses(
            {"worker-1": self.status(1)},
            expected=4,
            now=self.now,
        )

        self.assertFalse(result.complete)

    def test_partial_application_failure_still_fails_closed_after_grace(self) -> None:
        result = aggregate_worker_statuses(
            {"worker-1": self.status(1, seconds_ago=61)},
            expected=4,
            now=self.now,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.failure_kind, "command_failed")
        self.assertFalse(result.recycle_resource)

    def test_partial_failure_without_terminal_time_fails_closed(self) -> None:
        status = self.status(1)
        status["finished_at"] = ""

        result = aggregate_worker_statuses(
            {"worker-1": status},
            expected=4,
            now=self.now,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.failure_kind, "command_failed")

    def test_late_slice_failure_overrides_incidental_exit_with_missing_peers(self) -> None:
        result = aggregate_worker_statuses(
            {
                "worker-1": self.status(1, seconds_ago=28),
                "worker-0": self.status(
                    134,
                    infrastructure_failure=SLICE_FAILURE_CHIP_DRIVER_ERROR,
                ),
            },
            expected=4,
            now=self.now,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.exit_code, 75)
        self.assertEqual(result.failure_kind, "retryable_infrastructure")
        self.assertTrue(result.recycle_resource)
        self.assertIn("2 worker status object(s) missing", result.error_summary)

    def test_unmarked_application_abort_is_not_retryable(self) -> None:
        result = aggregate_worker_statuses(
            {
                "worker-0": self.status(134),
                "worker-1": self.status(1),
                "worker-2": self.status(143),
                "worker-3": self.status(143),
            },
            expected=4,
            now=self.now,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.failure_kind, "command_failed")
        self.assertFalse(result.recycle_resource)

    def test_worker_status_records_exact_slice_failure_marker(self) -> None:
        job = JobSpec(
            id="job",
            tpu="v6e-16",
            bundle="gs://bucket/bundles/bundle.tar.gz",
            command="python train.py",
            buckets=("gs://bucket",),
            bucket_regions=(("us-east1", "gs://bucket"),),
            regional_bundles=(("us-east1", "gs://bucket/bundles/bundle.tar.gz"),),
            region="us-east",
            storage_region="us-east1",
            bucket="gs://bucket",
        )
        attempt = AttemptRecord(id="attempt", job_id=job.id, resource_id="resource")
        resource = ResourceRecord(
            id="resource",
            tpu_name="resource",
            zone="us-east1-d",
            tpu_type="v6e-16",
            worker_count=4,
        )

        script = DistributedTPURunner(name="runner").launch_script(
            job=job,
            attempt=attempt,
            resource=resource,
        )

        self.assertIn("SLICE_FAILURE_CHIP_DRIVER_ERROR", script)
        self.assertIn('infrastructure_failure="slice_failure_chip_driver_error"', script)
        self.assertIn(
            '"infrastructure_failure": os.environ["STATUS_INFRASTRUCTURE_FAILURE"]',
            script,
        )
        syntax = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
