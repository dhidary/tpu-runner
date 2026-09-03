from __future__ import annotations

import unittest

from tpu_runner.controller import Controller
from tpu_runner.gcp import generated_resource_names
from tpu_runner.runtime import AttemptRecord, ResourceRecord
from tpu_runner.specs import FleetSpec, TPUEntry


class RecordingStore:
    def __init__(self, attempt: AttemptRecord, resource: ResourceRecord) -> None:
        self.attempt = attempt
        self.resource = resource
        self.finished: dict | None = None
        self.events: list[tuple[str, dict]] = []

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        return self.attempt if attempt_id == self.attempt.id else None

    def get_resource(self, resource_id: str) -> ResourceRecord | None:
        return self.resource if resource_id == self.resource.id else None

    def finish_attempt(self, attempt_id: str, **kwargs) -> AttemptRecord:
        self.finished = {"attempt_id": attempt_id, **kwargs}
        if kwargs["retryable_infrastructure_recycle_threshold"] == 1:
            self.resource.status = "recycling"
        return self.attempt

    def record_event(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))


class RetryableInfrastructureControllerTests(unittest.TestCase):
    def test_exact_slice_failure_requeues_and_schedules_immediate_recycle(self) -> None:
        entry = TPUEntry(
            id="us-east1d-v6e-spot",
            type="v6e-16",
            zone="us-east1-d",
            count=1,
            runtime="v6e-ubuntu-2404",
            provisioning_model="spot",
            chip_limit=16,
            runner_name="runner",
        )
        _, tpu_name = generated_resource_names(entry, 1)
        resource = ResourceRecord(
            id=tpu_name,
            tpu_name=tpu_name,
            zone=entry.zone,
            tpu_type=entry.type,
            fleet_entry_id=entry.id,
            status="busy",
            current_job_id="job",
            current_attempt_id="attempt",
            worker_count=4,
        )
        attempt = AttemptRecord(id="attempt", job_id="job", resource_id=resource.id)
        store = RecordingStore(attempt, resource)
        fleet = FleetSpec(
            name="runner",
            project="project",
            bucket="gs://runner",
            controller_region="us-central1",
            controller_timeout="1h",
            firestore_location="nam5",
            network="default",
            subnetwork="default",
            worker_secrets=(),
            tpus=(entry,),
        )
        controller = Controller(fleet=fleet, store=store, gcp=object())

        controller.mark_attempt_retryable_infrastructure(
            attempt.id,
            exit_code=75,
            error_summary="slice_failure_chip_driver_error",
            recycle_immediately=True,
        )

        self.assertIsNotNone(store.finished)
        self.assertEqual(store.finished["job_status"], "pending")
        self.assertEqual(store.finished["retryable_infrastructure_recycle_threshold"], 1)
        self.assertEqual(resource.status, "recycling")
        self.assertIn(
            "retryable_infrastructure_recycle_scheduled",
            [kind for kind, _ in store.events],
        )


if __name__ == "__main__":
    unittest.main()
