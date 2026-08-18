from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .region_pools import logical_region_pool, region_is_in_pool
from .specs import (
    STORED_JOB_SPEC_FIELDS,
    CacheSpec,
    JobSpec,
    region_from_zone,
    stable_id,
)


ATTEMPT_END_REASONS = frozenset(
    {
        "succeeded",
        "preempted",
        "cancelled",
        "application_failed",
        "setup_failed",
        "lost",
    }
)
ATTEMPT_END_REASON_BY_STATUS = {
    "succeeded": frozenset({"succeeded"}),
    "interrupted": frozenset({"preempted", "lost"}),
    "cancelled": frozenset({"cancelled"}),
    "failed": frozenset({"application_failed"}),
    "failed_setup": frozenset({"setup_failed"}),
}


def lease_is_acquirable(
    current: dict,
    *,
    owner: str,
    epoch: str,
    required_epoch: str,
    now: datetime,
) -> bool:
    """Return whether this controller generation may acquire the lease.

    A non-empty epoch fences controllers from earlier deployments. This lets a
    newly deployed controller take over immediately even when an old Cloud Run
    execution is still alive or has disappeared from the executions API.
    """

    if required_epoch and epoch != required_epoch:
        return False
    held_until = parse_datetime(current.get("expires_at", ""))
    if not held_until or held_until <= now:
        return True
    held_by = current.get("owner", "")
    if held_by == owner:
        return True
    # A new deployment epoch fences renewal by the old controller, but it must
    # not steal a still-live lease.  The old controller may already be inside a
    # reconcile pass with mutating GCP calls in flight.  Letting the new epoch
    # acquire immediately would create a split-brain window where both fleet
    # specifications can create or delete resources concurrently.
    return False


def lease_is_live(current: dict | None, *, now: datetime) -> bool:
    """Return whether another controller can still rely on this lease."""

    if not current:
        return False
    held_until = parse_datetime(current.get("expires_at", ""))
    return bool(held_until and held_until > now)


@dataclass
class ResourceRecord:
    id: str
    tpu_name: str
    zone: str
    tpu_type: str
    fleet_entry_id: str | None = None
    adopted: bool = False
    status: str = "idle"
    current_job_id: str | None = None
    current_attempt_id: str | None = None
    worker_count: int = 1
    idle_since: str = ""
    retryable_infrastructure_failures: int = 0
    retryable_infrastructure_job_id: str = ""
    cancellation_failures: int = 0
    cancellation_job_id: str = ""
    cancellation_attempt_id: str = ""


@dataclass
class JobRecord:
    spec: JobSpec
    submitted_at: str = ""
    status: str = "pending"
    assigned_resource_id: str | None = None
    current_attempt_id: str | None = None


@dataclass
class AttemptRecord:
    id: str
    job_id: str
    resource_id: str
    status: str = "running"
    exit_code: int | None = None
    error_summary: str = ""
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    end_reason: str = ""


@dataclass
class InterruptionRequestRecord:
    id: str
    resource_id: str
    job_id: str
    attempt_id: str
    fleet_entry_id: str
    status: str = "requested"
    requested_at: str = ""
    processed_at: str = ""
    queued_resource_name: str = ""
    error_summary: str = ""


def validate_interruption_target(
    *,
    resource: ResourceRecord,
    job: JobRecord,
    attempt: AttemptRecord,
    eligible_fleet_entry_ids: set[str] | frozenset[str],
) -> None:
    """Reject any target that is not one exact, live managed-Spot attempt."""
    if resource.adopted:
        raise ValueError(f"resource {resource.id!r} is adopted and cannot be interrupted")
    if not resource.fleet_entry_id or resource.fleet_entry_id not in eligible_fleet_entry_ids:
        raise ValueError(f"resource {resource.id!r} is not declared managed Spot capacity")
    if resource.status != "busy":
        raise ValueError(f"resource {resource.id!r} is not busy: {resource.status!r}")
    if resource.current_job_id != job.spec.id or resource.current_attempt_id != attempt.id:
        raise ValueError("resource does not own the exact requested job and attempt")
    if job.status != "running":
        raise ValueError(f"job {job.spec.id!r} is not running: {job.status!r}")
    if job.assigned_resource_id != resource.id or job.current_attempt_id != attempt.id:
        raise ValueError("job is not assigned to the exact requested resource and attempt")
    if attempt.status not in {"launching", "running"}:
        raise ValueError(f"attempt {attempt.id!r} is not active: {attempt.status!r}")
    if attempt.job_id != job.spec.id or attempt.resource_id != resource.id:
        raise ValueError("attempt does not match the exact requested job and resource")


def validate_cancellation_recycle_target(
    *,
    resource: ResourceRecord,
    job: JobRecord,
    attempt: AttemptRecord,
    eligible_fleet_entry_ids: set[str] | frozenset[str],
) -> None:
    """Reject cancellation fallback unless one exact managed-Spot attempt owns the TPU."""
    if resource.adopted:
        raise ValueError(f"resource {resource.id!r} is adopted and cannot be recycled")
    if not resource.fleet_entry_id or resource.fleet_entry_id not in eligible_fleet_entry_ids:
        raise ValueError(f"resource {resource.id!r} is not declared managed Spot capacity")
    if resource.status != "busy":
        raise ValueError(f"resource {resource.id!r} is not busy: {resource.status!r}")
    if job.status != "cancelling":
        raise ValueError(f"job {job.spec.id!r} is not cancelling: {job.status!r}")
    if attempt.status not in {"launching", "running"}:
        raise ValueError(f"attempt {attempt.id!r} is not active: {attempt.status!r}")
    if (
        resource.current_job_id != job.spec.id
        or resource.current_attempt_id != attempt.id
        or job.assigned_resource_id != resource.id
        or job.current_attempt_id != attempt.id
        or attempt.job_id != job.spec.id
        or attempt.resource_id != resource.id
    ):
        raise ValueError("resource, job, and attempt do not have exact reciprocal ownership")


class FirestoreStateStore:
    def __init__(self, *, collection_prefix: str = "tpu_runner", project: str | None = None):
        from google.cloud import firestore

        self.client = firestore.Client(project=project)
        self.prefix = collection_prefix.strip("/")

    def _collection(self, name: str):
        return self.client.collection(f"{self.prefix}_{name}")

    def upsert_resource(self, resource: ResourceRecord) -> None:
        self._collection("resources").document(resource.id).set(asdict(resource))

    def list_resources(self) -> list[ResourceRecord]:
        return [ResourceRecord(**doc.to_dict()) for doc in self._collection("resources").stream()]

    def get_resource(self, resource_id: str) -> ResourceRecord | None:
        snapshot = self._collection("resources").document(resource_id).get()
        return ResourceRecord(**snapshot.to_dict()) if snapshot.exists else None

    def create_jobs(self, jobs: list[JobRecord]) -> None:
        """Atomically create a fully materialized submission."""
        from google.cloud import firestore

        refs = [self._collection("jobs").document(job.spec.id) for job in jobs]

        @firestore.transactional
        def create(transaction):
            snapshots = list(transaction.get_all(refs))
            existing = [snapshot.id for snapshot in snapshots if snapshot.exists]
            if existing:
                raise ValueError(f"job already exists: {', '.join(sorted(existing))}")
            for ref, job in zip(refs, jobs, strict=True):
                transaction.set(ref, job_record_to_dict(job))

        create(self.client.transaction())

    def list_jobs(self) -> list[JobRecord]:
        return [job_record_from_dict(doc.to_dict()) for doc in self._collection("jobs").stream()]

    def list_jobs_with_statuses(self, statuses: set[str] | frozenset[str]) -> list[JobRecord]:
        from google.cloud.firestore_v1.base_query import FieldFilter

        if not statuses:
            return []
        query = self._collection("jobs").where(
            filter=FieldFilter("status", "in", sorted(statuses))
        )
        return [job_record_from_dict(doc.to_dict()) for doc in query.stream()]

    def has_jobs_with_statuses(self, statuses: set[str] | frozenset[str]) -> bool:
        from google.cloud.firestore_v1.base_query import FieldFilter

        if not statuses:
            return False
        query = (
            self._collection("jobs")
            .where(filter=FieldFilter("status", "in", sorted(statuses)))
            .limit(1)
        )
        return next(iter(query.stream()), None) is not None

    def get_job(self, job_id: str) -> JobRecord | None:
        snapshot = self._collection("jobs").document(job_id).get()
        return job_record_from_dict(snapshot.to_dict()) if snapshot.exists else None

    def cancel_job(self, job_id: str, *, if_pending: bool = False) -> str | None:
        """Cancel a job, optionally only while it is pending and unassigned.

        Firestore retries this transaction if assignment commits after its
        read.  The retry then observes the assignment and returns ``conflict``
        without writing, so guarded cancellation cannot deactivate a job that
        has begun assignment.
        """
        from google.cloud import firestore

        ref = self._collection("jobs").document(job_id)

        @firestore.transactional
        def cancel(transaction):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            job = job_record_from_dict(snapshot.to_dict())
            if if_pending and (
                job.status != "pending"
                or job.assigned_resource_id is not None
                or job.current_attempt_id is not None
            ):
                return "conflict"
            if job.status == "pending":
                job.status = "deactivated"
            elif job.status == "running":
                job.status = "cancelling"
            transaction.set(ref, job_record_to_dict(job))
            return job.status

        return cancel(self.client.transaction())

    def reprioritize_pending_job(
        self,
        job_id: str,
        *,
        priority: str,
    ) -> str | None:
        """Atomically reprioritize one pending, unassigned job.

        Firestore retries the transaction if assignment commits after its read.
        The retry then observes a running or assigned job and returns
        ``conflict`` without mutating live work.
        """
        from google.cloud import firestore

        job_ref = self._collection("jobs").document(job_id)
        event_ref = self._collection("events").document()

        @firestore.transactional
        def reprioritize(transaction):
            snapshot = job_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            job = job_record_from_dict(snapshot.to_dict())
            if job.status != "pending" or job.assigned_resource_id is not None:
                return "conflict"
            replacement_spec = replace(
                job.spec,
                priority=priority,
            )
            if replacement_spec == job.spec:
                return "unchanged"
            old_priority = job.spec.priority
            job.spec = replacement_spec
            changed_at = datetime.now(timezone.utc).isoformat()
            transaction.set(job_ref, job_record_to_dict(job))
            transaction.set(
                event_ref,
                {
                    "kind": "pending_job_reprioritized",
                    "payload": {
                        "job_id": job_id,
                        "old_priority": old_priority,
                        "new_priority": replacement_spec.priority,
                    },
                    "created_at": changed_at,
                },
            )
            return "reprioritized"

        return reprioritize(self.client.transaction())

    def upsert_attempt(self, attempt: AttemptRecord) -> None:
        self._collection("attempts").document(attempt.id).set(asdict(attempt))

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        snapshot = self._collection("attempts").document(attempt_id).get()
        return attempt_record_from_dict(snapshot.to_dict()) if snapshot.exists else None

    def list_attempts(self) -> list[AttemptRecord]:
        return [attempt_record_from_dict(doc.to_dict()) for doc in self._collection("attempts").stream()]

    def create_interruption_request(
        self,
        *,
        resource_id: str,
        job_id: str,
        attempt_id: str,
        eligible_fleet_entry_ids: set[str] | frozenset[str],
    ) -> InterruptionRequestRecord:
        """Atomically record one fail-closed, controller-consumed interruption."""
        from google.cloud import firestore

        request_id = stable_id(
            "interrupt",
            {"resource_id": resource_id, "job_id": job_id, "attempt_id": attempt_id},
        )
        resource_ref = self._collection("resources").document(resource_id)
        job_ref = self._collection("jobs").document(job_id)
        attempt_ref = self._collection("attempts").document(attempt_id)
        request_ref = self._collection("interruption_requests").document(request_id)

        @firestore.transactional
        def create(transaction):
            snapshots = list(
                transaction.get_all([resource_ref, job_ref, attempt_ref, request_ref])
            )
            by_path = {snapshot.reference.path: snapshot for snapshot in snapshots}
            missing = [
                label
                for label, ref in (
                    ("resource", resource_ref),
                    ("job", job_ref),
                    ("attempt", attempt_ref),
                )
                if not by_path[ref.path].exists
            ]
            if missing:
                raise ValueError(f"unknown interruption target: {', '.join(missing)}")
            if by_path[request_ref.path].exists:
                raise ValueError(f"interruption request already exists: {request_id}")
            resource = ResourceRecord(**by_path[resource_ref.path].to_dict())
            job = job_record_from_dict(by_path[job_ref.path].to_dict())
            attempt = attempt_record_from_dict(by_path[attempt_ref.path].to_dict())
            validate_interruption_target(
                resource=resource,
                job=job,
                attempt=attempt,
                eligible_fleet_entry_ids=eligible_fleet_entry_ids,
            )
            request = InterruptionRequestRecord(
                id=request_id,
                resource_id=resource.id,
                job_id=job.spec.id,
                attempt_id=attempt.id,
                fleet_entry_id=resource.fleet_entry_id or "",
                requested_at=datetime.now(timezone.utc).isoformat(),
            )
            transaction.create(request_ref, asdict(request))
            return request

        return create(self.client.transaction())

    def list_interruption_requests(self) -> list[InterruptionRequestRecord]:
        return [
            interruption_request_from_dict(doc.to_dict())
            for doc in self._collection("interruption_requests").stream()
        ]

    def list_interruption_requests_with_statuses(
        self, statuses: set[str] | frozenset[str]
    ) -> list[InterruptionRequestRecord]:
        from google.cloud.firestore_v1.base_query import FieldFilter

        if not statuses:
            return []
        query = self._collection("interruption_requests").where(
            filter=FieldFilter("status", "in", sorted(statuses))
        )
        return [
            interruption_request_from_dict(doc.to_dict()) for doc in query.stream()
        ]

    def claim_interruption_request(
        self,
        request_id: str,
        *,
        eligible_fleet_entry_ids: set[str] | frozenset[str],
    ) -> bool:
        """Atomically revalidate and claim a request exactly once."""
        from google.cloud import firestore

        ref = self._collection("interruption_requests").document(request_id)

        @firestore.transactional
        def claim(transaction):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                raise KeyError(request_id)
            request = interruption_request_from_dict(snapshot.to_dict())
            if request.status != "requested":
                return False
            resource_ref = self._collection("resources").document(request.resource_id)
            job_ref = self._collection("jobs").document(request.job_id)
            attempt_ref = self._collection("attempts").document(request.attempt_id)
            target_snapshots = list(
                transaction.get_all([resource_ref, job_ref, attempt_ref])
            )
            by_path = {
                target_snapshot.reference.path: target_snapshot
                for target_snapshot in target_snapshots
            }
            if any(
                not by_path[target_ref.path].exists
                for target_ref in (resource_ref, job_ref, attempt_ref)
            ):
                raise ValueError("interruption target no longer exists")
            resource = ResourceRecord(**by_path[resource_ref.path].to_dict())
            job = job_record_from_dict(by_path[job_ref.path].to_dict())
            attempt = attempt_record_from_dict(by_path[attempt_ref.path].to_dict())
            if resource.fleet_entry_id != request.fleet_entry_id:
                raise ValueError("resource fleet entry changed after interruption was requested")
            validate_interruption_target(
                resource=resource,
                job=job,
                attempt=attempt,
                eligible_fleet_entry_ids=eligible_fleet_entry_ids,
            )
            request.status = "processing"
            transaction.set(ref, asdict(request))
            return True

        return bool(claim(self.client.transaction()))

    def finish_interruption_request(
        self,
        request_id: str,
        *,
        status: str,
        queued_resource_name: str = "",
        error_summary: str = "",
    ) -> None:
        from google.cloud import firestore

        if status not in {"deletion_requested", "rejected", "failed"}:
            raise ValueError(f"invalid terminal interruption-request status: {status!r}")
        ref = self._collection("interruption_requests").document(request_id)

        @firestore.transactional
        def finish(transaction):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                raise KeyError(request_id)
            request = interruption_request_from_dict(snapshot.to_dict())
            if request.status not in {"requested", "processing"}:
                raise ValueError(
                    f"interruption request {request_id!r} is already terminal: "
                    f"{request.status!r}"
                )
            request.status = status
            request.processed_at = datetime.now(timezone.utc).isoformat()
            request.queued_resource_name = queued_resource_name
            request.error_summary = error_summary
            transaction.set(ref, asdict(request))

        finish(self.client.transaction())

    def assign_job(self, job_id: str, resource_id: str) -> AttemptRecord | None:
        """Atomically claim one pending job and idle resource."""
        from google.cloud import firestore

        job_ref = self._collection("jobs").document(job_id)
        resource_ref = self._collection("resources").document(resource_id)

        @firestore.transactional
        def assign(transaction):
            snapshots = list(transaction.get_all([job_ref, resource_ref]))
            by_path = {snapshot.reference.path: snapshot for snapshot in snapshots}
            job_snapshot = by_path.get(job_ref.path)
            resource_snapshot = by_path.get(resource_ref.path)
            if not job_snapshot or not job_snapshot.exists or not resource_snapshot or not resource_snapshot.exists:
                return None
            job = job_record_from_dict(job_snapshot.to_dict())
            resource = ResourceRecord(**resource_snapshot.to_dict())
            if job.status != "pending" or resource.status != "idle":
                return None
            if not job.spec.accepts_tpu_type(resource.tpu_type):
                return None
            if job.spec.tpu_name and job.spec.tpu_name != resource.tpu_name:
                return None
            if job.spec.zone and job.spec.zone != resource.zone:
                return None
            resource_region = region_from_zone(resource.zone)
            if not job.spec.accepts_region(resource_region):
                return None
            if job.spec.storage_region:
                if (
                    not job.spec.region
                    or not region_is_in_pool(resource_region, job.spec.region)
                    or job.spec.storage_region != resource_region
                    or not job.spec.bucket
                    or not job.spec.bundle
                ):
                    return None
            else:
                bucket = job.spec.bucket_for_region(resource_region)
                bundle = job.spec.bundle_for_region(resource_region)
                if not bucket or not bundle:
                    return None
                job.spec = replace(
                    job.spec,
                    region=logical_region_pool(resource_region),
                    storage_region=resource_region,
                    bucket=bucket,
                    bundle=bundle,
                )
            attempt_id = stable_id(
                "attempt",
                {"job": job.spec.id, "resource": resource.id, "previous": job.current_attempt_id or ""},
            )
            attempt = AttemptRecord(
                id=attempt_id,
                job_id=job.spec.id,
                resource_id=resource.id,
                status="launching",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            job.status = "running"
            job.assigned_resource_id = resource.id
            job.current_attempt_id = attempt_id
            resource.status = "busy"
            resource.current_job_id = job.spec.id
            resource.current_attempt_id = attempt_id
            resource.idle_since = ""
            resource.cancellation_failures = 0
            resource.cancellation_job_id = ""
            resource.cancellation_attempt_id = ""
            transaction.set(self._collection("attempts").document(attempt_id), asdict(attempt))
            transaction.set(job_ref, job_record_to_dict(job))
            transaction.set(resource_ref, asdict(resource))
            return attempt

        return assign(self.client.transaction())

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        attempt_status: str,
        job_status: str,
        resource_status: str,
        exit_code: int | None = None,
        error_summary: str = "",
        end_reason: str,
        expected_job_status: str = "running",
        retryable_infrastructure_failure: bool = False,
        retryable_infrastructure_recycle_threshold: int = 0,
    ) -> AttemptRecord | None:
        """Atomically finish the current attempt and release its job and resource."""
        from google.cloud import firestore

        if end_reason not in ATTEMPT_END_REASONS:
            raise ValueError(f"invalid attempt end_reason: {end_reason!r}")
        if end_reason not in ATTEMPT_END_REASON_BY_STATUS.get(attempt_status, frozenset()):
            raise ValueError(
                f"attempt status {attempt_status!r} cannot end as {end_reason!r}"
            )

        attempt_ref = self._collection("attempts").document(attempt_id)

        @firestore.transactional
        def finish(transaction):
            next_resource_status = resource_status
            attempt_snapshot = attempt_ref.get(transaction=transaction)
            if not attempt_snapshot.exists:
                raise KeyError(attempt_id)
            attempt = attempt_record_from_dict(attempt_snapshot.to_dict())
            job_ref = self._collection("jobs").document(attempt.job_id)
            resource_ref = self._collection("resources").document(attempt.resource_id)
            snapshots = list(transaction.get_all([job_ref, resource_ref]))
            by_path = {snapshot.reference.path: snapshot for snapshot in snapshots}
            job_snapshot = by_path[job_ref.path]
            resource_snapshot = by_path[resource_ref.path]
            if not job_snapshot.exists or not resource_snapshot.exists:
                raise KeyError(f"missing job or resource for attempt {attempt_id}")
            job = job_record_from_dict(job_snapshot.to_dict())
            resource = ResourceRecord(**resource_snapshot.to_dict())
            if job.status != expected_job_status:
                return None
            if job.current_attempt_id != attempt_id or resource.current_attempt_id != attempt_id:
                raise ValueError(f"attempt {attempt_id} is no longer current")
            if attempt.ended_at or attempt.end_reason:
                raise ValueError(f"attempt {attempt_id} already has terminal timing")

            attempt.status = attempt_status
            attempt.exit_code = exit_code
            attempt.error_summary = error_summary
            attempt.ended_at = datetime.now(timezone.utc).isoformat()
            attempt.end_reason = end_reason
            job.status = job_status
            job.assigned_resource_id = None
            if retryable_infrastructure_failure:
                if resource.retryable_infrastructure_job_id == attempt.job_id:
                    resource.retryable_infrastructure_failures += 1
                else:
                    resource.retryable_infrastructure_failures = 1
                    resource.retryable_infrastructure_job_id = attempt.job_id
                if (
                    retryable_infrastructure_recycle_threshold > 0
                    and resource.retryable_infrastructure_failures
                    >= retryable_infrastructure_recycle_threshold
                ):
                    next_resource_status = "recycling"
            else:
                resource.retryable_infrastructure_failures = 0
                resource.retryable_infrastructure_job_id = ""
            resource.cancellation_failures = 0
            resource.cancellation_job_id = ""
            resource.cancellation_attempt_id = ""
            resource.status = next_resource_status
            resource.current_job_id = None
            resource.current_attempt_id = None
            resource.idle_since = (
                datetime.now(timezone.utc).isoformat()
                if next_resource_status == "idle" and not resource.adopted
                else ""
            )
            transaction.set(attempt_ref, asdict(attempt))
            transaction.set(job_ref, job_record_to_dict(job))
            transaction.set(resource_ref, asdict(resource))
            return attempt

        return finish(self.client.transaction())

    def note_cancellation_failure(
        self,
        *,
        resource_id: str,
        job_id: str,
        attempt_id: str,
        eligible_fleet_entry_ids: set[str] | frozenset[str],
        threshold: int,
        error_summary: str,
    ) -> ResourceRecord:
        """Persist one exact cancellation failure and recycle only at the bound."""
        from google.cloud import firestore

        if threshold <= 0:
            raise ValueError("cancellation recycle threshold must be positive")
        resource_ref = self._collection("resources").document(resource_id)
        job_ref = self._collection("jobs").document(job_id)
        attempt_ref = self._collection("attempts").document(attempt_id)

        @firestore.transactional
        def record(transaction):
            snapshots = list(transaction.get_all([resource_ref, job_ref, attempt_ref]))
            by_path = {snapshot.reference.path: snapshot for snapshot in snapshots}
            if any(not by_path[ref.path].exists for ref in (resource_ref, job_ref, attempt_ref)):
                raise KeyError("missing resource, job, or attempt during cancellation fallback")
            resource = ResourceRecord(**by_path[resource_ref.path].to_dict())
            job = job_record_from_dict(by_path[job_ref.path].to_dict())
            attempt = attempt_record_from_dict(by_path[attempt_ref.path].to_dict())
            validate_cancellation_recycle_target(
                resource=resource,
                job=job,
                attempt=attempt,
                eligible_fleet_entry_ids=eligible_fleet_entry_ids,
            )
            if (
                resource.cancellation_job_id == job_id
                and resource.cancellation_attempt_id == attempt_id
            ):
                resource.cancellation_failures += 1
            else:
                resource.cancellation_failures = 1
                resource.cancellation_job_id = job_id
                resource.cancellation_attempt_id = attempt_id
            if resource.cancellation_failures >= threshold:
                attempt.status = "cancelled"
                attempt.error_summary = error_summary
                attempt.ended_at = datetime.now(timezone.utc).isoformat()
                attempt.end_reason = "cancelled"
                job.status = "deactivated"
                job.assigned_resource_id = None
                resource.status = "recycling"
                resource.current_job_id = None
                resource.current_attempt_id = None
                resource.idle_since = ""
                transaction.set(attempt_ref, asdict(attempt))
                transaction.set(job_ref, job_record_to_dict(job))
            transaction.set(resource_ref, asdict(resource))
            return resource

        return record(self.client.transaction())

    def reset_cancellation_failures(
        self,
        *,
        resource_id: str,
        job_id: str,
        attempt_id: str,
    ) -> bool:
        """Clear the counter only while the same exact attempt still owns the resource."""
        from google.cloud import firestore

        ref = self._collection("resources").document(resource_id)

        @firestore.transactional
        def reset(transaction):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            resource = ResourceRecord(**snapshot.to_dict())
            if (
                resource.current_job_id != job_id
                or resource.current_attempt_id != attempt_id
            ):
                return False
            resource.cancellation_failures = 0
            resource.cancellation_job_id = ""
            resource.cancellation_attempt_id = ""
            transaction.set(ref, asdict(resource))
            return True

        return reset(self.client.transaction())

    def record_event(self, kind: str, payload: dict) -> None:
        event = {
            "kind": kind,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._collection("events").add(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    def acquire_lease(
        self,
        name: str,
        owner: str,
        ttl_seconds: int,
        *,
        epoch: str = "",
    ) -> bool:
        from google.cloud import firestore

        ref = self._collection("locks").document(name)
        epoch_ref = self._collection("lease_epochs").document(name)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)

        @firestore.transactional
        def acquire(transaction):
            epoch_snapshot = epoch_ref.get(transaction=transaction)
            required_epoch = (
                (epoch_snapshot.to_dict() or {}).get("epoch", "")
                if epoch_snapshot.exists
                else ""
            )
            snapshot = ref.get(transaction=transaction)
            if snapshot.exists:
                data = snapshot.to_dict() or {}
                if not lease_is_acquirable(
                    data,
                    owner=owner,
                    epoch=epoch,
                    required_epoch=required_epoch,
                    now=now,
                ):
                    return False
            elif required_epoch and epoch != required_epoch:
                return False
            transaction.set(
                ref,
                {
                    "owner": owner,
                    "epoch": epoch,
                    "acquired_at": now.isoformat(),
                    "expires_at": expires_at.isoformat(),
                },
            )
            return True

        return bool(acquire(self.client.transaction()))

    def set_lease_epoch(self, name: str, epoch: str) -> None:
        if not epoch:
            raise ValueError("lease epoch cannot be empty")
        self._collection("lease_epochs").document(name).set(
            {
                "epoch": epoch,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_lease(self, name: str) -> dict | None:
        snapshot = self._collection("locks").document(name).get()
        return (snapshot.to_dict() or {}) if snapshot.exists else None

    def release_lease(self, name: str, owner: str) -> None:
        from google.cloud import firestore

        ref = self._collection("locks").document(name)

        @firestore.transactional
        def release(transaction):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                return
            data = snapshot.to_dict() or {}
            if data.get("owner") == owner:
                transaction.delete(ref)

        release(self.client.transaction())


class GCSArtifactClient:
    def upload(self, source: Path, uri: str) -> None:
        subprocess.run(["gcloud", "storage", "cp", str(source), uri], check=True)


def checkpoint_dir(bucket: str, job_id: str) -> str:
    return f"{bucket.rstrip('/')}/jobs/{job_id}/checkpoints"


def job_bucket(job: JobSpec) -> str:
    if not job.bucket:
        raise ValueError(f"job {job.id!r} has no selected regional bucket")
    return job.bucket.rstrip("/")


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def job_record_to_dict(job: JobRecord) -> dict:
    spec = asdict(job.spec)
    if isinstance(job.spec.tpu, tuple):
        spec["tpu"] = list(job.spec.tpu)
    spec["caches"] = [asdict(cache) for cache in job.spec.caches]
    spec["env"] = dict(job.spec.env)
    spec["buckets"] = list(job.spec.buckets)
    spec["bucket_regions"] = [
        {"region": region, "bucket": bucket}
        for region, bucket in job.spec.bucket_regions
    ]
    spec["regional_bundles"] = [
        {"region": region, "bundle": bundle}
        for region, bundle in job.spec.regional_bundles
    ]
    return {
        "spec": spec,
        "submitted_at": job.submitted_at,
        "status": job.status,
        "assigned_resource_id": job.assigned_resource_id,
        "current_attempt_id": job.current_attempt_id,
    }


def job_record_from_dict(data: dict) -> JobRecord:
    spec = data["spec"]
    unknown = sorted(set(spec) - STORED_JOB_SPEC_FIELDS)
    if unknown:
        raise ValueError(
            "stored job spec has unknown field(s): " + ", ".join(unknown)
        )
    tpu = tuple(spec["tpu"]) if isinstance(spec["tpu"], list) else spec["tpu"]
    caches = tuple(CacheSpec(**cache) for cache in spec.get("caches", []))
    bucket = str(spec.get("bucket", "")).rstrip("/")
    storage_region = str(spec.get("storage_region", ""))
    buckets = tuple(str(value).rstrip("/") for value in spec.get("buckets", []))
    bucket_regions = tuple(
        (str(item["region"]), str(item["bucket"]).rstrip("/"))
        for item in spec.get("bucket_regions", [])
    )
    regional_bundles = tuple(
        (str(item["region"]), str(item["bundle"]))
        for item in spec.get("regional_bundles", [])
    )
    # Existing stored jobs were already pinned to one job bucket. Preserve
    # their selected placement while new submissions require `buckets`.
    if not buckets and bucket:
        buckets = (bucket,)
    if not bucket_regions and bucket and storage_region:
        bucket_regions = ((storage_region, bucket),)
    if not regional_bundles and spec.get("bundle") and storage_region:
        regional_bundles = ((storage_region, str(spec["bundle"])),)
    return JobRecord(
        spec=JobSpec(
            id=spec["id"],
            tpu=tpu,
            bundle=spec["bundle"],
            command=spec["command"],
            tpu_name=spec.get("tpu_name", ""),
            zone=spec.get("zone", ""),
            buckets=buckets,
            bucket_regions=bucket_regions,
            regional_bundles=regional_bundles,
            region=spec.get("region", ""),
            storage_region=storage_region,
            bucket=bucket,
            priority=spec.get("priority", "normal"),
            caches=caches,
            env={str(key): str(value) for key, value in spec.get("env", {}).items()},
        ),
        submitted_at=data.get("submitted_at", ""),
        status=data.get("status", "pending"),
        assigned_resource_id=data.get("assigned_resource_id"),
        current_attempt_id=data.get("current_attempt_id"),
    )


def attempt_record_from_dict(data: dict) -> AttemptRecord:
    return AttemptRecord(
        id=data["id"],
        job_id=data["job_id"],
        resource_id=data["resource_id"],
        status=data.get("status", "running"),
        exit_code=data.get("exit_code"),
        error_summary=data.get("error_summary", ""),
        created_at=data.get("created_at", ""),
        started_at=data.get("started_at", ""),
        ended_at=data.get("ended_at", ""),
        end_reason=data.get("end_reason", ""),
    )


def interruption_request_from_dict(data: dict) -> InterruptionRequestRecord:
    return InterruptionRequestRecord(
        id=data["id"],
        resource_id=data["resource_id"],
        job_id=data["job_id"],
        attempt_id=data["attempt_id"],
        fleet_entry_id=data["fleet_entry_id"],
        status=data.get("status", "requested"),
        requested_at=data.get("requested_at", ""),
        processed_at=data.get("processed_at", ""),
        queued_resource_name=data.get("queued_resource_name", ""),
        error_summary=data.get("error_summary", ""),
    )
