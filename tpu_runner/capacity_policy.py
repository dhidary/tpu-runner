"""Pure placement and capacity policy."""

from __future__ import annotations

from .gcp import generated_resource_names
from .region_pools import region_is_in_pool
from .runtime import JobRecord, ResourceRecord
from .specs import FleetSpec, TPUEntry, region_from_zone


JOB_PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2}


def pending_job_accepts_entry(job: JobRecord, entry) -> bool:
    if job.status != "pending":
        return False
    if job.spec.zone and job.spec.zone != entry.zone:
        return False
    entry_region = region_from_zone(entry.zone)
    if not job.spec.accepts_region(entry_region):
        return False
    if job.spec.storage_region and (
        not job.spec.region or not region_is_in_pool(entry_region, job.spec.region)
    ):
        return False
    if job.spec.tpu_name:
        if entry.adopted:
            return (
                job.spec.accepts_tpu_type(entry.type)
                and job.spec.tpu_name == entry.existing
            )
        return job.spec.accepts_tpu_type(entry.type) and any(
            generated_resource_names(entry, ordinal)[1] == job.spec.tpu_name
            for ordinal in range(1, entry.count + 1)
        )
    return (entry.adopted or entry.count > 0) and job.spec.accepts_tpu_type(
        entry.type
    )


def pending_job_accepts_resource(job: JobRecord, resource: ResourceRecord) -> bool:
    """Match compatible idle capacity while preserving exact TPU pins."""

    if job.status != "pending":
        return False
    if not job.spec.accepts_tpu_type(resource.tpu_type):
        return False
    if job.spec.zone and job.spec.zone != resource.zone:
        return False
    resource_region = region_from_zone(resource.zone)
    if not job.spec.accepts_region(resource_region):
        return False
    if job.spec.storage_region and (
        not job.spec.region or not region_is_in_pool(resource_region, job.spec.region)
    ):
        return False
    return not job.spec.tpu_name or job.spec.tpu_name == resource.tpu_name


def resource_is_busy(resource: ResourceRecord) -> bool:
    return bool(
        resource.status == "busy"
        or resource.current_job_id
        or resource.current_attempt_id
    )


def pending_job_entry_constraint_key(job: JobRecord, entries: list) -> tuple:
    compatible = sum(pending_job_accepts_entry(job, entry) for entry in entries)
    return (
        compatible == 0,
        -JOB_PRIORITY_RANK[job.spec.priority],
        compatible,
        job.submitted_at,
        job.spec.id,
    )


def pending_job_resource_constraint_key(
    job: JobRecord,
    resources: list[ResourceRecord],
) -> tuple:
    compatible = sum(
        pending_job_accepts_resource(job, resource) for resource in resources
    )
    return (
        compatible == 0,
        -JOB_PRIORITY_RANK[job.spec.priority],
        compatible,
        job.submitted_at,
        job.spec.id,
    )


def plan_idle_assignments(
    jobs: list[JobRecord],
    resources: list[ResourceRecord],
    *,
    entries: list[TPUEntry] | None = None,
) -> list[tuple[JobRecord, ResourceRecord]]:
    """Match idle resources without displacing an earlier pending job.

    Jobs are considered by priority, constraint count, and FIFO order. An
    augmenting path may move an earlier flexible job to another idle resource,
    but it never drops that job merely to fit a later one.
    """

    idle_resources = sorted(
        (
            resource
            for resource in resources
            if resource.status == "idle" and not resource_is_busy(resource)
        ),
        key=lambda resource: (not resource.adopted, resource.id),
    )
    resources_by_id = {resource.id: resource for resource in idle_resources}
    ordered_jobs = sorted(
        (job for job in jobs if job.status == "pending"),
        key=lambda job: (
            pending_job_entry_constraint_key(job, entries)
            if entries is not None
            else pending_job_resource_constraint_key(job, idle_resources)
        ),
    )
    matched_by_resource: dict[str, JobRecord] = {}

    def match(job: JobRecord, visited: set[str]) -> bool:
        candidates = sorted(
            (
                resource
                for resource in idle_resources
                if resource.id not in visited
                and pending_job_accepts_resource(job, resource)
            ),
            key=lambda resource: (
                resource.id in matched_by_resource,
                not resource.adopted,
                resource.id,
            ),
        )
        for resource in candidates:
            visited.add(resource.id)
            previous = matched_by_resource.get(resource.id)
            if previous is None or match(previous, visited):
                matched_by_resource[resource.id] = job
                return True
        return False

    for job in ordered_jobs:
        match(job, set())

    resources_by_job = {
        job.spec.id: resources_by_id[resource_id]
        for resource_id, job in matched_by_resource.items()
    }
    return [
        (job, resources_by_job[job.spec.id])
        for job in ordered_jobs
        if job.spec.id in resources_by_job
    ]


def allocate_managed_pending_demand(
    jobs: list[JobRecord],
    *,
    fleet: FleetSpec,
    resources: tuple[ResourceRecord, ...],
) -> dict[str, int]:
    """Request every pending job in every compatible Spot entry.

    These are capacity races, not duplicate executions. Assignment remains a
    Firestore transaction, and the losing entry's demand disappears on the
    next reconciliation. Every physical entry count remains a hard ceiling.
    """

    entries = [entry for entry in fleet.tpus if not entry.adopted]
    busy = {
        entry.id: sum(
            1
            for resource in resources
            if resource.fleet_entry_id == entry.id
            and not resource.adopted
            and resource_is_busy(resource)
        )
        for entry in entries
    }
    demand = {entry.id: 0 for entry in entries}
    for job in sorted(
        jobs,
        key=lambda candidate: pending_job_entry_constraint_key(candidate, entries),
    ):
        for entry in entries:
            available = max(0, entry.count - busy[entry.id])
            if demand[entry.id] >= available:
                continue
            if pending_job_accepts_entry(job, entry):
                demand[entry.id] += 1
    return demand


def desired_managed_capacity_counts(
    jobs: list[JobRecord],
    *,
    fleet: FleetSpec,
    resources: tuple[ResourceRecord, ...],
) -> dict[str, int]:
    """Return busy plus raced pending demand, capped by physical ceilings."""

    declared_adopted = {
        (entry.id, entry.existing) for entry in fleet.tpus if entry.adopted
    }
    idle_adopted = [
        resource
        for resource in resources
        if resource.adopted
        and resource.status == "idle"
        and (resource.fleet_entry_id, resource.tpu_name) in declared_adopted
    ]
    adopted_job_ids = {
        job.spec.id
        for job, _ in plan_idle_assignments(
            jobs,
            idle_adopted,
            entries=list(fleet.tpus),
        )
    }
    jobs_requiring_managed = [
        job
        for job in jobs
        if job.status == "pending" and job.spec.id not in adopted_job_ids
    ]

    pending_demand = allocate_managed_pending_demand(
        jobs_requiring_managed,
        fleet=fleet,
        resources=resources,
    )
    desired: dict[str, int] = {}
    for entry in fleet.tpus:
        if entry.adopted:
            continue
        busy_count = sum(
            1
            for resource in resources
            if resource.fleet_entry_id == entry.id
            and not resource.adopted
            and resource_is_busy(resource)
        )
        desired[entry.id] = min(
            entry.count,
            max(entry.keep_warm_count, busy_count + pending_demand.get(entry.id, 0)),
        )
    return desired
