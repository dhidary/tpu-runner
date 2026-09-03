from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tpu_runner.controller import (
    ORPHANED_PROVISIONING_NODE_GRACE_SECONDS,
    Controller,
)
from tpu_runner.gcp import QueuedResource, TPUVM, generated_resource_names
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

    def list_resources(self) -> list[ResourceRecord]:
        return [self.resource]

    def upsert_resource(self, resource: ResourceRecord) -> None:
        self.resource = resource


class RecordingGCP:
    def __init__(self) -> None:
        self.deleted_queued_resources: list[dict] = []

    def delete_queued_resource(self, **kwargs) -> None:
        self.deleted_queued_resources.append(kwargs)


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


class OrphanedProvisioningControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = TPUEntry(
            id="us-east1d-v6e-spot",
            type="v6e-16",
            zone="us-east1-d",
            count=1,
            runtime="v6e-ubuntu-2404",
            provisioning_model="spot",
            chip_limit=16,
            runner_name="runner",
        )
        _, self.tpu_name = generated_resource_names(self.entry, 1)
        self.resource = ResourceRecord(
            id=self.tpu_name,
            tpu_name=self.tpu_name,
            zone=self.entry.zone,
            tpu_type=self.entry.type,
            fleet_entry_id=self.entry.id,
            status="provisioning",
        )
        self.attempt = AttemptRecord(
            id="attempt",
            job_id="job",
            resource_id=self.resource.id,
        )
        self.store = RecordingStore(self.attempt, self.resource)
        self.gcp = RecordingGCP()
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
            tpus=(self.entry,),
        )
        self.controller = Controller(fleet=fleet, store=self.store, gcp=self.gcp)
        self.queued = QueuedResource(
            name="qr-" + self.tpu_name,
            zone=self.entry.zone,
            accelerator_type=self.entry.type,
            state="PROVISIONING",
            node_id=self.tpu_name,
        )

    def test_never_materialized_node_is_not_recycled(self) -> None:
        self.controller.reconcile_unusable_resources([], [self.queued])

        self.assertEqual(self.gcp.deleted_queued_resources, [])
        self.assertEqual(self.resource.provisioning_node_missing_since, "")

    def test_missing_seen_node_starts_grace_period(self) -> None:
        self.resource.provisioning_node_seen_at = datetime.now(
            timezone.utc
        ).isoformat()

        self.controller.reconcile_unusable_resources([], [self.queued])

        self.assertEqual(self.gcp.deleted_queued_resources, [])
        self.assertTrue(self.resource.provisioning_node_missing_since)

    def test_orphaned_seen_node_is_recycled_after_grace_period(self) -> None:
        now = datetime.now(timezone.utc)
        self.resource.provisioning_node_seen_at = now.isoformat()
        self.resource.provisioning_node_missing_since = (
            now
            - timedelta(seconds=ORPHANED_PROVISIONING_NODE_GRACE_SECONDS + 1)
        ).isoformat()

        self.controller.reconcile_unusable_resources([], [self.queued])

        self.assertEqual(
            self.gcp.deleted_queued_resources,
            [
                {
                    "name": self.queued.name,
                    "zone": self.queued.zone,
                    "project": "project",
                }
            ],
        )
        self.assertEqual(self.resource.status, "draining")
        self.assertIn(
            "orphaned_provisioning_tpu_recycling",
            [kind for kind, _ in self.store.events],
        )

    def test_reappearing_node_clears_missing_timer(self) -> None:
        now = datetime.now(timezone.utc)
        self.resource.provisioning_node_seen_at = now.isoformat()
        self.resource.provisioning_node_missing_since = now.isoformat()
        tpu = TPUVM(
            name=self.tpu_name,
            zone=self.entry.zone,
            accelerator_type=self.entry.type,
            state="CREATING",
            labels={"managed-by": "runner", "fleet-entry": self.entry.id},
            worker_count=4,
        )

        self.controller.reconcile_unusable_resources([tpu], [self.queued])

        self.assertEqual(self.gcp.deleted_queued_resources, [])
        self.assertEqual(self.resource.provisioning_node_missing_since, "")


if __name__ == "__main__":
    unittest.main()
