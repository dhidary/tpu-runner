from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from .gcp import (
    QueuedResource,
    SubprocessGCPClient,
    TPUVM,
    generated_resource_names,
    resource_matches_entry,
)
from .capacity_policy import (
    desired_managed_capacity_counts,
    pending_job_accepts_entry,
    plan_idle_assignments,
    resource_is_busy,
)
from .specs import FleetSpec, region_from_zone, tpu_family_and_chips
from .runtime import (
    AttemptRecord,
    FirestoreStateStore,
    InterruptionRequestRecord,
    JobRecord,
    ResourceRecord,
    validate_interruption_target,
)

CANCELLATION_RECYCLE_THRESHOLD = 2
MANAGED_SPOT_LAUNCH_ACCESS_TIMEOUT_SECONDS = 30 * 60


def reset_cancellation_tracking(resource: ResourceRecord) -> None:
    resource.cancellation_failures = 0
    resource.cancellation_job_id = ""
    resource.cancellation_attempt_id = ""


def attempt_age_seconds(attempt: AttemptRecord) -> float | None:
    if not attempt.created_at:
        return None
    try:
        created_at = datetime.fromisoformat(attempt.created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds())


@dataclass
class Controller:
    fleet: FleetSpec
    store: FirestoreStateStore
    gcp: SubprocessGCPClient
    renew_lease: Callable[[], bool] | None = None

    def heartbeat(self) -> None:
        """Renew the controller lease between bounded remote operations."""

        if self.renew_lease is not None and not self.renew_lease():
            raise RuntimeError("controller lost its leader lease")

    def reconcile_once(self) -> None:
        self.heartbeat()
        stored_resources = self.store.list_resources()
        inventory_resources = [
            resource
            for resource in stored_resources
            if resource_is_inventory_owned_for_fleet(resource, self.fleet)
            and resource.status not in {"deleted", "preempted"}
        ]
        tpu_targets = tuple(
            (resource.tpu_name, resource.zone) for resource in inventory_resources
        )
        queued_targets = tuple(
            (f"qr-{resource.tpu_name}", resource.zone)
            for resource in inventory_resources
            if not resource.adopted
        )
        tpus = self.gcp.list_tpus(
            project=self.fleet.project,
            additional_targets=tpu_targets,
        )
        queued = self.gcp.list_queued_resources(
            project=self.fleet.project,
            additional_targets=queued_targets,
        )
        self.heartbeat()
        self.refresh_queued_resources(queued)
        self.refresh_ready_resources(tpus)
        self.recover_idle_adopted_disk_pressure(tpus)
        self.heartbeat()
        self.reconcile_cancellations()
        # Consume an already-authorized exact interruption before polling the
        # remote process. A poisoned TPU slice can make that process exit in
        # seconds; polling it first terminalizes the attempt, invalidates the
        # interruption target, and lets the same bad resource be reassigned.
        self.reconcile_interruption_requests(tpus, queued)
        self.heartbeat()
        self.reconcile_controller_managed_attempts()
        self.heartbeat()
        self.reconcile_managed_spot_recycles(tpus, queued)
        self.reconcile_unusable_resources(tpus, queued)
        self.assign_pending_jobs()
        self.heartbeat()
        self.reconcile_capacity(tpus, queued)
        # A controlled interruption can be requested while the first attempt
        # pass is blocked in an all-worker SSH launch.  Re-read the exact
        # interruption queue before the second attempt pass so the request is
        # not starved behind another full round of SSH timeouts.
        self.reconcile_interruption_requests(tpus, queued)
        self.heartbeat()
        self.reconcile_controller_managed_attempts()

    def reconcile_managed_spot_recycles(
        self,
        tpus: list[TPUVM],
        queued: list[QueuedResource],
    ) -> None:
        """Replace only an exact managed Spot ordinal after repeated exit-75 failures."""

        entries = {entry.id: entry for entry in self.fleet.tpus}
        for resource in self.store.list_resources():
            if resource.status != "recycling":
                continue
            entry = entries.get(resource.fleet_entry_id or "")
            if (
                entry is None
                or entry.adopted
                or entry.provisioning_model != "spot"
                or resource.adopted
                or not resource_is_declared_for_fleet(resource, self.fleet)
            ):
                self.store.record_event(
                    "retryable_infrastructure_recycle_rejected",
                    {
                        "resource_id": resource.id,
                        "fleet_entry_id": resource.fleet_entry_id or "",
                        "reason": "resource is not one declared runner-managed Spot ordinal",
                    },
                )
                continue

            recycle_reason = (
                "stuck_cancellation"
                if resource.cancellation_failures >= CANCELLATION_RECYCLE_THRESHOLD
                else "repeated_retryable_infrastructure_failure"
            )
            expected_queued_name = queued_resource_name_for_node(entry, resource.tpu_name)
            queued_by_name = [candidate for candidate in queued if candidate.name == expected_queued_name]
            queued_by_node = [candidate for candidate in queued if candidate.node_id == resource.tpu_name]
            mismatched = [
                candidate
                for candidate in queued_by_name + queued_by_node
                if candidate.name != expected_queued_name
                or candidate.node_id != resource.tpu_name
                or not resource_matches_entry(candidate, entry)
            ]
            if mismatched:
                self.store.record_event(
                    "retryable_infrastructure_recycle_rejected",
                    {
                        "resource_id": resource.id,
                        "fleet_entry_id": entry.id,
                        "reason": "queued-resource identity mismatch",
                    },
                )
                continue
            matching_queued = [
                candidate
                for candidate in queued_by_name
                if candidate.node_id == resource.tpu_name
                and resource_matches_entry(candidate, entry)
            ]
            if len(matching_queued) > 1:
                self.store.record_event(
                    "retryable_infrastructure_recycle_rejected",
                    {
                        "resource_id": resource.id,
                        "fleet_entry_id": entry.id,
                        "reason": "multiple exact queued resources",
                    },
                )
                continue
            if matching_queued:
                queued_resource = matching_queued[0]
                if queued_resource.deleting:
                    continue
                self.gcp.delete_queued_resource(
                    name=queued_resource.name,
                    zone=queued_resource.zone,
                    project=self.fleet.project,
                )
                self.store.record_event(
                    "retryable_infrastructure_recycle_requested",
                    {
                        "resource_id": resource.id,
                        "fleet_entry_id": entry.id,
                        "queued_resource": queued_resource.name,
                        "reason": recycle_reason,
                    },
                )
                resource.status = "recycling_requested"
                resource.idle_since = ""
                self.store.upsert_resource(resource)
                continue

            matching_tpus = [
                tpu
                for tpu in tpus
                if tpu.name == resource.tpu_name
                and tpu.spot
                and resource_matches_entry(tpu, entry)
            ]
            if len(matching_tpus) > 1:
                self.store.record_event(
                    "retryable_infrastructure_recycle_rejected",
                    {
                        "resource_id": resource.id,
                        "fleet_entry_id": entry.id,
                        "reason": "multiple exact TPU VMs",
                    },
                )
                continue
            if matching_tpus:
                tpu = matching_tpus[0]
                if tpu.queued_resource and tpu.queued_resource != expected_queued_name:
                    self.store.record_event(
                        "retryable_infrastructure_recycle_rejected",
                        {
                            "resource_id": resource.id,
                            "fleet_entry_id": entry.id,
                            "reason": "TPU points at a different queued resource",
                        },
                    )
                    continue
                if not tpu.terminal:
                    self.gcp.delete_tpu_vm(
                        name=tpu.name,
                        zone=tpu.zone,
                        project=self.fleet.project,
                    )
                    self.store.record_event(
                        "retryable_infrastructure_tpu_recycle_requested",
                        {
                            "resource_id": resource.id,
                            "fleet_entry_id": entry.id,
                            "tpu": tpu.name,
                            "reason": f"{recycle_reason}:queued_resource_absent",
                        },
                    )
                    resource.status = "recycling_requested"
                    resource.idle_since = ""
                    self.store.upsert_resource(resource)
                continue

            resource.status = "preempted"
            resource.retryable_infrastructure_failures = 0
            resource.retryable_infrastructure_job_id = ""
            reset_cancellation_tracking(resource)
            resource.idle_since = ""
            self.store.upsert_resource(resource)
            self.store.record_event(
                "retryable_infrastructure_recycle_completed",
                {"resource_id": resource.id, "fleet_entry_id": entry.id},
            )

    def reconcile_interruption_requests(
        self,
        tpus: list[TPUVM],
        queued: list[QueuedResource],
    ) -> None:
        """Delete only the exact managed Spot queued resource named by a request."""
        entries = {entry.id: entry for entry in self.fleet.tpus}
        for request in sorted(
            self.store.list_interruption_requests_with_statuses({"requested"}),
            key=lambda item: item.id,
        ):
            self.heartbeat()
            try:
                entry = entries.get(request.fleet_entry_id)
                if entry is None or entry.adopted or entry.provisioning_model != "spot":
                    raise ValueError("request does not name a declared managed Spot fleet entry")
                resource = self.store.get_resource(request.resource_id)
                job = self.store.get_job(request.job_id)
                attempt = self.store.get_attempt(request.attempt_id)
                if resource is None or job is None or attempt is None:
                    raise ValueError("request target no longer exists")
                validate_interruption_target(
                    resource=resource,
                    job=job,
                    attempt=attempt,
                    eligible_fleet_entry_ids={entry.id},
                )
                if not resource_is_declared_for_fleet(resource, self.fleet):
                    raise ValueError("resource is not a currently declared managed ordinal")
                expected_queued_name = queued_resource_name_for_node(entry, resource.tpu_name)
                live_tpus = [
                    tpu
                    for tpu in tpus
                    if tpu.name == resource.tpu_name
                    and tpu.ready_for_ssh(self.fleet.ssh_transport)
                    and not tpu.terminal
                    and tpu.spot
                    and resource_matches_entry(tpu, entry)
                ]
                if len(live_tpus) != 1:
                    raise ValueError(
                        f"expected exactly one live managed Spot TPU, found {len(live_tpus)}"
                    )
                live_tpu = live_tpus[0]
                matching_queued = [
                    candidate
                    for candidate in queued
                    if candidate.name == expected_queued_name
                    and candidate.node_id == resource.tpu_name
                    and candidate.pending_or_active
                    and not candidate.deleting
                    and resource_matches_entry(candidate, entry)
                ]
                if len(matching_queued) != 1:
                    raise ValueError(
                        "expected exactly one active managed Spot queued resource, "
                        f"found {len(matching_queued)}"
                    )
                queued_resource = matching_queued[0]
                if live_tpu.queued_resource and live_tpu.queued_resource != queued_resource.name:
                    raise ValueError("live TPU points at a different queued resource")
                if not self.store.claim_interruption_request(
                    request.id, eligible_fleet_entry_ids={entry.id}
                ):
                    continue
                try:
                    self.gcp.delete_queued_resource(
                        name=queued_resource.name,
                        zone=queued_resource.zone,
                        project=self.fleet.project,
                    )
                except Exception as exc:
                    self.store.finish_interruption_request(
                        request.id,
                        status="failed",
                        queued_resource_name=queued_resource.name,
                        error_summary=str(exc),
                    )
                    self.store.record_event(
                        "controlled_interruption_failed",
                        interruption_event_payload(request, queued_resource.name, error=str(exc)),
                    )
                    continue
                self.store.finish_interruption_request(
                    request.id,
                    status="deletion_requested",
                    queued_resource_name=queued_resource.name,
                )
                try:
                    finished_attempt = self.store.finish_attempt(
                        request.attempt_id,
                        attempt_status="interrupted",
                        job_status="pending",
                        resource_status="draining",
                        exit_code=75,
                        error_summary="controlled managed-Spot interruption",
                        end_reason="lost",
                    )
                except Exception as exc:
                    finished_attempt = None
                    self.store.record_event(
                        "controlled_interruption_attempt_requeue_failed",
                        interruption_event_payload(
                            request, queued_resource.name, error=str(exc)
                        ),
                    )
                if finished_attempt is not None:
                    self.store.record_event(
                        "controlled_interruption_attempt_requeued",
                        interruption_event_payload(request, queued_resource.name),
                    )
                self.store.record_event(
                    "controlled_interruption_deletion_requested",
                    interruption_event_payload(request, queued_resource.name),
                )
            except Exception as exc:
                self.store.finish_interruption_request(
                    request.id,
                    status="rejected",
                    error_summary=str(exc),
                )
                self.store.record_event(
                    "controlled_interruption_rejected",
                    interruption_event_payload(request, error=str(exc)),
                )

    def reconcile_cancellations(self) -> None:
        from .distributed import DistributedTPURunner, TemporaryAccessError

        runner = DistributedTPURunner(
            name=self.fleet.name,
            project=self.fleet.project,
            ssh_transport=self.fleet.ssh_transport,
        )
        for job in self.store.list_jobs_with_statuses({"cancelling"}):
            self.heartbeat()
            if not job.current_attempt_id:
                continue
            attempt = self.store.get_attempt(job.current_attempt_id)
            resource = self.store.get_resource(job.assigned_resource_id or "")
            if attempt is None or resource is None:
                continue
            try:
                result = runner.poll(job=job.spec, attempt=attempt, resource=resource)
                if result.complete:
                    self.mark_attempt_cancelled(attempt, resource.id, exit_code=result.exit_code)
                    continue
                cancel_outcome = runner.cancel(attempt=attempt, resource=resource)
                if cancel_outcome == "not_running":
                    self.mark_attempt_cancelled(attempt, resource.id)
                elif cancel_outcome == "sent":
                    self.store.reset_cancellation_failures(
                        resource_id=resource.id,
                        job_id=job.spec.id,
                        attempt_id=attempt.id,
                    )
                elif cancel_outcome == "attempt_mismatch":
                    self.store.record_event(
                        "job_cancel_attempt_mismatch",
                        {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource.id},
                    )
                    self.record_cancellation_failure(
                        job=job,
                        attempt=attempt,
                        resource=resource,
                        error_summary="exact remote attempt identity mismatch during cancellation",
                    )
            except TemporaryAccessError as exc:
                self.store.record_event(
                    "controller_temporary_access_error",
                    {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource.id, "error": str(exc)},
                )
                self.record_cancellation_failure(
                    job=job,
                    attempt=attempt,
                    resource=resource,
                    error_summary=str(exc),
                )
            except Exception as exc:
                self.store.record_event(
                    "job_cancel_error",
                    {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource.id, "error": str(exc)},
                )
                self.record_cancellation_failure(
                    job=job,
                    attempt=attempt,
                    resource=resource,
                    error_summary=str(exc),
                )

    def record_cancellation_failure(
        self,
        *,
        job: JobRecord,
        attempt: AttemptRecord,
        resource: ResourceRecord,
        error_summary: str,
    ) -> None:
        eligible = frozenset(
            entry.id
            for entry in self.fleet.tpus
            if not entry.adopted and entry.provisioning_model == "spot"
        )
        try:
            updated = self.store.note_cancellation_failure(
                resource_id=resource.id,
                job_id=job.spec.id,
                attempt_id=attempt.id,
                eligible_fleet_entry_ids=eligible,
                threshold=CANCELLATION_RECYCLE_THRESHOLD,
                error_summary=error_summary,
            )
        except (KeyError, ValueError) as exc:
            self.store.record_event(
                "job_cancel_recycle_rejected",
                {
                    "job_id": job.spec.id,
                    "attempt_id": attempt.id,
                    "resource_id": resource.id,
                    "error": str(exc),
                },
            )
            return
        event = "job_cancel_failure_recorded"
        if updated.status == "recycling":
            event = "job_cancel_recycle_scheduled"
        self.store.record_event(
            event,
            {
                "job_id": job.spec.id,
                "attempt_id": attempt.id,
                "resource_id": resource.id,
                "failure_count": updated.cancellation_failures,
                "error": error_summary,
            },
        )

    def mark_attempt_cancelled(
        self,
        attempt: AttemptRecord,
        resource_id: str,
        *,
        exit_code: int | None = None,
    ) -> None:
        finished = self.store.finish_attempt(
            attempt.id,
            attempt_status="cancelled",
            job_status="deactivated",
            resource_status="idle",
            exit_code=exit_code,
            end_reason="cancelled",
            expected_job_status="cancelling",
        )
        if finished:
            self.store.record_event(
                "job_cancelled",
                {"job_id": attempt.job_id, "attempt_id": attempt.id, "resource_id": resource_id},
            )

    def refresh_ready_resources(self, tpus: list[TPUVM]) -> None:
        resources = {resource.tpu_name: resource for resource in self.store.list_resources()}
        for entry in self.fleet.tpus:
            for tpu in tpus:
                matches = (
                    tpu.name == entry.existing
                    and tpu.zone == entry.zone
                    and tpu.accelerator_type == entry.type
                    and ("spot" if tpu.spot else "on-demand") == entry.provisioning_model
                    if entry.adopted
                    else resource_matches_entry(tpu, entry)
                )
                if not matches:
                    continue
                if tpu.ready_for_ssh(self.fleet.ssh_transport):
                    self._upsert_inventory_resource(resource_record_from_tpu(tpu, entry.id, adopted=entry.adopted))
                elif tpu.name not in resources:
                    resource = resource_record_from_tpu(tpu, entry.id, adopted=entry.adopted)
                    resource.status = "unavailable"
                    self.store.upsert_resource(resource)

    def refresh_queued_resources(self, queued: list[QueuedResource]) -> None:
        """Persist exact managed ordinals before a TPU VM exists."""

        entries = tuple(entry for entry in self.fleet.tpus if not entry.adopted)
        resources = {resource.tpu_name: resource for resource in self.store.list_resources()}
        for queued_resource in queued:
            if not queued_resource.pending_or_active or not queued_resource.node_id:
                continue
            entry = next(
                (
                    candidate
                    for candidate in entries
                    if resource_matches_entry(queued_resource, candidate)
                ),
                None,
            )
            if entry is None:
                continue
            resource = resources.get(queued_resource.node_id)
            if resource is None:
                resource = ResourceRecord(
                    id=queued_resource.node_id,
                    tpu_name=queued_resource.node_id,
                    zone=entry.zone,
                    tpu_type=entry.type,
                    fleet_entry_id=entry.id,
                    adopted=False,
                    status="provisioning",
                )
            elif resource.current_attempt_id:
                continue
            else:
                resource.zone = entry.zone
                resource.tpu_type = entry.type
                resource.fleet_entry_id = entry.id
                resource.adopted = False
                resource.status = "provisioning"
                resource.current_job_id = None
                resource.current_attempt_id = None
                resource.idle_since = ""
            self.store.upsert_resource(resource)
            resources[resource.tpu_name] = resource

    def recover_idle_adopted_disk_pressure(self, tpus: list[TPUVM]) -> None:
        """Recover runner scratch only on idle adopted TPUs with unhealthy runtime health."""
        from .distributed import DistributedTPURunner, MINIMUM_ROOT_FREE_KB

        entries = {entry.id: entry for entry in self.fleet.tpus}
        resources = {resource.tpu_name: resource for resource in self.store.list_resources()}
        runner = DistributedTPURunner(
            name=self.fleet.name,
            project=self.fleet.project,
            ssh_transport=self.fleet.ssh_transport,
        )
        for tpu in tpus:
            resource = resources.get(tpu.name)
            if resource is None:
                continue
            entry = entries.get(resource.fleet_entry_id or "")
            if (
                entry is None
                or not entry.adopted
                or not resource.adopted
                or resource.status not in {"idle", "unavailable"}
                or resource.current_job_id
                or resource.current_attempt_id
                or tpu.state.upper() not in {"READY", "HEALTHY"}
                or tpu.health.upper() == "HEALTHY"
                or tpu.terminal
                or not tpu.supports_ssh_transport(self.fleet.ssh_transport)
                or tpu.name != entry.existing
                or tpu.zone != entry.zone
                or tpu.accelerator_type != entry.type
                or ("spot" if tpu.spot else "on-demand") != entry.provisioning_model
            ):
                continue
            try:
                workers = runner.recover_adopted_disk_pressure(resource=resource)
            except Exception as exc:
                self.store.record_event(
                    "adopted_tpu_disk_recovery_failed",
                    {
                        "resource_id": resource.id,
                        "tpu": tpu.name,
                        "state": tpu.state,
                        "health": tpu.health,
                        "error": str(exc),
                    },
                )
                continue
            recovered = {
                host: details
                for host, details in workers.items()
                if details["outcome"] == "recovered"
            }
            if recovered:
                self.store.record_event(
                    "adopted_tpu_disk_recovery_completed",
                    {
                        "resource_id": resource.id,
                        "tpu": tpu.name,
                        "state": tpu.state,
                        "health": tpu.health,
                        "minimum_root_free_kb": MINIMUM_ROOT_FREE_KB,
                        "workers": workers,
                    },
                )

    def reconcile_unusable_resources(
        self,
        tpus: list[TPUVM],
        queued: list[QueuedResource] | None = None,
    ) -> None:
        by_name = {tpu.name: tpu for tpu in tpus}
        terminal_queued_by_node = {
            item.node_id: item
            for item in (queued or [])
            if item.node_id and item.terminal
        }
        queued_by_node = {
            item.node_id: item
            for item in (queued or [])
            if item.node_id and item.pending_or_active
        }
        entries = {entry.id: entry for entry in self.fleet.tpus}
        for resource in self.store.list_resources():
            entry = entries.get(resource.fleet_entry_id or "")
            if entry is None:
                continue
            tpu = by_name.get(resource.tpu_name)
            terminal_queued_resource = terminal_queued_by_node.get(resource.tpu_name)
            if terminal_queued_resource is not None:
                # A suspending/terminal queued resource is already lost
                # capacity even when its TPU VM briefly continues to report
                # READY.  Requeue work immediately and let GCP finish teardown
                # asynchronously instead of leaving the attempt falsely live.
                self.mark_tpu_preempted(resource.id)
                continue
            declared = bool(tpu) and (
                (
                    tpu.zone == entry.zone
                    and tpu.accelerator_type == entry.type
                    and ("spot" if tpu.spot else "on-demand") == entry.provisioning_model
                )
                if entry.adopted
                else resource_matches_entry(tpu, entry)
            )
            if (
                tpu
                and tpu.ready_for_ssh(self.fleet.ssh_transport)
                and declared
                and resource.status != "draining"
            ):
                continue
            if not declared:
                tpu = None
            if resource.adopted:
                if tpu and not tpu.terminal:
                    if not resource.current_attempt_id:
                        resource.status = "unavailable"
                        self.store.upsert_resource(resource)
                    continue
                if resource.current_attempt_id:
                    self.mark_tpu_preempted(resource.id)
                else:
                    resource.status = "unavailable"
                    resource.current_job_id = None
                    resource.current_attempt_id = None
                    self.store.upsert_resource(resource)
                continue
            queued_resource = queued_by_node.get(resource.tpu_name)
            if tpu is None and queued_resource is not None:
                if resource.current_attempt_id:
                    self.mark_tpu_preempted(resource.id)
                    resource = self.store.get_resource(resource.id) or resource
                if not resource.current_attempt_id:
                    resource.status = "provisioning"
                    resource.current_job_id = None
                    resource.idle_since = ""
                    self.store.upsert_resource(resource)
                continue
            if (
                tpu is not None
                and queued_resource is not None
                and not tpu.operationally_ready
                and not tpu.terminal
                and not resource.current_attempt_id
            ):
                # GCP can expose the TPU VM while its queued resource is still
                # provisioning.  This is expected inventory convergence, not
                # unusable capacity; keep one truthful state across refresh
                # and reconciliation instead of flipping to ``unavailable``.
                resource.status = "provisioning"
                resource.current_job_id = None
                resource.idle_since = ""
                self.store.upsert_resource(resource)
                continue
            if tpu:
                state = tpu.state.upper()
                if tpu.terminal and resource.current_attempt_id:
                    # Requeue the scientific job as soon as GCP reports the TPU
                    # terminal.  Waiting for the node to disappear can leave the
                    # attempt falsely running while replacement capacity comes up.
                    self.mark_tpu_preempted(resource.id)
                    resource = self.store.get_resource(resource.id) or resource
                    if resource.current_attempt_id:
                        # The compare-and-swap in finish_attempt did not commit;
                        # leave the resource untouched and retry next reconcile.
                        continue
                if (
                    state in {"READY", "HEALTHY"}
                    and tpu.health.upper() == "UNHEALTHY_MAINTENANCE"
                    and resource.status != "draining"
                ):
                    if resource.status != "unavailable":
                        resource.status = "unavailable"
                        self.store.upsert_resource(resource)
                        self.store.record_event(
                            "spot_tpu_waiting_for_maintenance",
                            {
                                "resource_id": resource.id,
                                "tpu": tpu.name,
                                "state": tpu.state,
                                "health": tpu.health,
                            },
                        )
                    continue
                if (
                    resource.status != "draining"
                    and not tpu.terminal
                    and state not in {"READY", "HEALTHY"}
                ):
                    resource.status = "unavailable"
                    self.store.upsert_resource(resource)
                    continue
                if state != "DELETING":
                    self.gcp.delete_tpu_vm(
                        name=tpu.name,
                        zone=tpu.zone,
                        project=self.fleet.project,
                    )
                if resource.status != "draining":
                    resource.status = "draining"
                    self.store.upsert_resource(resource)
                    self.store.record_event(
                        "unusable_spot_tpu_deleting",
                        {
                            "resource_id": resource.id,
                            "tpu": tpu.name,
                            "state": tpu.state,
                            "health": tpu.health,
                        },
                    )
                continue
            if resource.current_attempt_id:
                self.mark_tpu_preempted(resource.id)
            elif resource.status not in {"deleted", "preempted"}:
                resource.status = "preempted"
                resource.current_job_id = None
                resource.current_attempt_id = None
                resource.idle_since = ""
                self.store.upsert_resource(resource)
                self.store.record_event(
                    "idle_spot_tpu_missing",
                    {"resource_id": resource.id, "tpu": resource.tpu_name},
                )

    def reconcile_capacity(self, tpus: list[TPUVM], queued: list[QueuedResource]) -> None:
        jobs = self.store.list_jobs_with_statuses({"pending"})
        stored_resources = {
            resource.tpu_name: resource for resource in self.store.list_resources()
        }
        quota_remaining: dict[tuple[str, str, str], int] = {}
        for entry in self.fleet.tpus:
            key = quota_key(entry)
            quota_remaining.setdefault(
                key,
                max(0, entry.chip_limit - reserved_chips(entry, tpus=tpus, queued=queued)),
            )
        desired_counts = desired_managed_capacity_counts(
            jobs,
            fleet=self.fleet,
            resources=tuple(stored_resources.values()),
        )

        for entry in self.fleet.tpus:
            self.heartbeat()
            if entry.adopted:
                continue
            declared_capacity = [
                generated_resource_names(entry, ordinal)
                for ordinal in range(1, entry.count + 1)
            ]
            desired_count = desired_counts[entry.id]
            declared_node_ids = {node_id for _, node_id in declared_capacity}
            busy_node_ids = {
                resource.tpu_name
                for resource in stored_resources.values()
                if resource.fleet_entry_id == entry.id
                and not resource.adopted
                and resource_is_busy(resource)
            }
            exact_pending_node_ids = {
                job.spec.tpu_name
                for job in jobs
                if job.spec.tpu_name in declared_node_ids
                and pending_job_accepts_entry(job, entry)
            }
            terminal_queued_names = {
                qr.name
                for qr in queued
                if qr.terminal and resource_matches_entry(qr, entry)
            }
            terminal_queued_node_ids = {
                qr.node_id
                for qr in queued
                if qr.node_id and qr.terminal and resource_matches_entry(qr, entry)
            }
            live_node_ids = {
                tpu.name
                for tpu in tpus
                if resource_matches_entry(tpu, entry)
                and tpu.ready_for_ssh(self.fleet.ssh_transport)
                and tpu.name not in terminal_queued_node_ids
                and (not tpu.queued_resource or tpu.queued_resource not in terminal_queued_names)
            } | {
                qr.node_id
                for qr in queued
                if qr.node_id
                and resource_matches_entry(qr, entry)
                and qr.pending_or_active
            }
            preferred_node_ids: list[str] = []
            for candidates in (
                busy_node_ids,
                exact_pending_node_ids,
                live_node_ids,
                declared_node_ids - terminal_queued_node_ids,
            ):
                for _, node_id in declared_capacity:
                    if node_id in candidates and node_id not in preferred_node_ids:
                        preferred_node_ids.append(node_id)
            desired_node_ids = set(preferred_node_ids[:desired_count])
            desired_capacity = [
                pair for pair in declared_capacity if pair[1] in desired_node_ids
            ]
            matching_tpus = [
                tpu
                for tpu in tpus
                if resource_matches_entry(tpu, entry)
                and tpu.ready_for_ssh(self.fleet.ssh_transport)
                and tpu.name not in terminal_queued_node_ids
                and (not tpu.queued_resource or tpu.queued_resource not in terminal_queued_names)
            ]
            terminal_tpus = [tpu for tpu in tpus if resource_matches_entry(tpu, entry) and tpu.terminal]
            deleted_queued_resources: set[str] = set()
            # A lower configured count or a warm-idle target removes higher
            # ordinals from the current target. Drain those ordinals once idle
            # so they neither consume quota nor receive new work. A busy
            # ordinal remains untouched until its attempt completes.
            stage_cleanup_tpus = matching_tpus if desired_count else []
            for tpu in stage_cleanup_tpus:
                if tpu.name in desired_node_ids:
                    continue
                resource = stored_resources.get(tpu.name) or resource_record_from_tpu(
                    tpu, entry.id, adopted=False
                )
                if resource_is_busy(resource):
                    continue
                self.gcp.delete_tpu_vm(
                    name=tpu.name,
                    zone=tpu.zone,
                    project=self.fleet.project,
                )
                resource.status = "deleted"
                resource.idle_since = ""
                self.store.upsert_resource(resource)
                self.store.record_event(
                    "undeclared_spot_tpu_deleted",
                    {
                        "fleet_entry_id": entry.id,
                        "tpu": tpu.name,
                        "reason": "capacity_target_reduced",
                    },
                )
            desired_pairs = set(desired_capacity)
            stage_cleanup_queued = queued if desired_count else []
            for qr in stage_cleanup_queued:
                if (
                    not resource_matches_entry(qr, entry)
                    or (qr.name, qr.node_id) in desired_pairs
                    or qr.deleting
                ):
                    continue
                resource = stored_resources.get(qr.node_id or "")
                if resource is not None and resource_is_busy(resource):
                    continue
                self.gcp.delete_queued_resource(
                    name=qr.name,
                    zone=qr.zone,
                    project=self.fleet.project,
                )
                deleted_queued_resources.add(qr.name)
                self.store.record_event(
                    "undeclared_spot_queued_resource_deleted",
                    {
                        "fleet_entry_id": entry.id,
                        "queued_resource": qr.name,
                        "reason": "capacity_target_reduced",
                    },
                )
            if desired_count == 0:
                ready_tpu_names = {tpu.name for tpu in matching_tpus}
                deleted_tpu_names: set[str] = set()
                for tpu in matching_tpus:
                    resource = stored_resources.get(tpu.name) or resource_record_from_tpu(
                        tpu, entry.id, adopted=False
                    )
                    if resource_is_busy(resource):
                        continue
                    self.gcp.delete_tpu_vm(
                        name=tpu.name,
                        zone=tpu.zone,
                        project=self.fleet.project,
                    )
                    deleted_tpu_names.add(tpu.name)
                    resource.status = "deleted"
                    resource.current_job_id = None
                    resource.current_attempt_id = None
                    resource.idle_since = ""
                    self.store.upsert_resource(resource)
                    self.store.record_event(
                        "idle_spot_tpu_deleted",
                        {
                            "fleet_entry_id": entry.id,
                            "tpu": tpu.name,
                            "queued_resource": tpu.queued_resource or "",
                            "reason": "no_pending_or_running_jobs",
                        },
                    )
                for qr in queued:
                    if (
                        resource_matches_entry(qr, entry)
                        and not qr.deleting
                        and (
                            not qr.node_id
                            or qr.node_id not in ready_tpu_names
                            or qr.node_id in deleted_tpu_names
                        )
                    ):
                        self.gcp.delete_queued_resource(
                            name=qr.name,
                            zone=qr.zone,
                            project=self.fleet.project,
                        )
                        deleted_queued_resources.add(qr.name)
                        self.store.record_event(
                            "idle_spot_queued_resource_deleted",
                            {
                                "fleet_entry_id": entry.id,
                                "queued_resource": qr.name,
                                "state": qr.state,
                                "reason": "no_pending_or_running_jobs",
                            },
                        )
                for resource in self.store.list_resources():
                    if resource.tpu_name in deleted_tpu_names and not resource.adopted:
                        resource.status = "deleted"
                        resource.idle_since = ""
                        self.store.upsert_resource(resource)
                continue
            # Capacity names are a fixed part of the deployment contract.  In
            # particular, do not replace a deleting ordinal-1 TPU with an
            # undeclared ordinal-2 TPU: a job pinned to ordinal 1 would then be
            # pending forever while the substitute incorrectly satisfied the
            # fleet count.
            matching_tpus = [tpu for tpu in matching_tpus if tpu.name in desired_node_ids]
            terminal_tpus = [tpu for tpu in terminal_tpus if tpu.name in desired_node_ids]
            terminal_node_ids = {tpu.name for tpu in terminal_tpus}
            terminal_queued_resources = {tpu.queued_resource for tpu in terminal_tpus if tpu.queued_resource}
            for qr in queued:
                if (
                    resource_matches_entry(qr, entry)
                    and not qr.deleting
                    and (qr.node_id in terminal_node_ids or qr.name in terminal_queued_resources)
                ):
                    self.gcp.delete_queued_resource(
                        name=qr.name,
                        zone=qr.zone,
                        project=self.fleet.project,
                    )
                    deleted_queued_resources.add(qr.name)
                    self.store.record_event(
                        "queued_resource_deleted",
                        {
                            "fleet_entry_id": entry.id,
                            "queued_resource": qr.name,
                            "state": qr.state,
                            "reason": "tpu_not_ready",
                        },
                    )
            matching_pending = [
                qr
                for qr in queued
                if resource_matches_entry(qr, entry) and qr.pending_or_active
                and qr.name not in deleted_queued_resources
                and qr.node_id not in terminal_node_ids
                and (qr.name, qr.node_id) in desired_pairs
            ]
            for qr in queued:
                if (
                    resource_matches_entry(qr, entry)
                    and qr.terminal
                    and not qr.deleting
                    and qr.name not in deleted_queued_resources
                    and (qr.name, qr.node_id) in desired_pairs
                ):
                    self.gcp.delete_queued_resource(
                        name=qr.name,
                        zone=qr.zone,
                        project=self.fleet.project,
                    )
                    self.store.record_event(
                        "queued_resource_deleted",
                        {"fleet_entry_id": entry.id, "queued_resource": qr.name, "state": qr.state},
                    )

            matching_tpu_names = {tpu.name for tpu in matching_tpus}
            queued_without_tpu = [qr for qr in matching_pending if qr.node_id not in matching_tpu_names]
            current_count = len(matching_tpus) + len(queued_without_tpu)
            missing = max(0, desired_count - current_count)
            _, chips_per_tpu = tpu_family_and_chips(entry.type)
            key = quota_key(entry)
            available_chips = quota_remaining[key]
            quota_slots = available_chips // chips_per_tpu
            if missing > quota_slots:
                self.store.record_event(
                    "tpu_chip_limit_reached",
                    {
                        "fleet_entry_id": entry.id,
                        "zone": entry.zone,
                        "type": entry.type,
                        "chip_limit": entry.chip_limit,
                        "available_chips": available_chips,
                    },
                )
                missing = quota_slots
            used_queued_resource_names = {qr.name for qr in queued}
            used_node_ids = {qr.node_id for qr in queued if qr.node_id}
            created = 0
            for queued_name, node_id in desired_capacity:
                if created >= missing:
                    break
                if queued_name in used_queued_resource_names or node_id in used_node_ids:
                    continue
                try:
                    self.gcp.create_queued_resource(
                        fleet=self.fleet,
                        entry=entry,
                        queued_resource_name=queued_name,
                        node_id=node_id,
                    )
                except Exception as exc:
                    self.store.record_event(
                        "queued_resource_create_failed",
                        {
                            "fleet_entry_id": entry.id,
                            "queued_resource": queued_name,
                            "node_id": node_id,
                            "type": entry.type,
                            "zone": entry.zone,
                            "error": str(exc),
                        },
                    )
                    break
                self.store.record_event(
                    "queued_resource_created",
                    {
                        "fleet_entry_id": entry.id,
                        "queued_resource": queued_name,
                        "node_id": node_id,
                        "type": entry.type,
                        "zone": entry.zone,
                    },
                )
                resource = stored_resources.get(node_id) or ResourceRecord(
                    id=node_id,
                    tpu_name=node_id,
                    zone=entry.zone,
                    tpu_type=entry.type,
                    fleet_entry_id=entry.id,
                    adopted=False,
                )
                resource.status = "provisioning"
                resource.current_job_id = None
                resource.current_attempt_id = None
                resource.idle_since = ""
                self.store.upsert_resource(resource)
                stored_resources[node_id] = resource
                used_queued_resource_names.add(queued_name)
                used_node_ids.add(node_id)
                quota_remaining[key] -= chips_per_tpu
                created += 1

    def managed_capacity_exists(self) -> bool:
        managed_entries = [entry for entry in self.fleet.tpus if not entry.adopted]
        stored_resources = self.store.list_resources()
        inventory_resources = [
            resource
            for resource in stored_resources
            if resource_is_inventory_owned_for_fleet(resource, self.fleet)
            and resource.status not in {"deleted", "preempted"}
        ]
        tpus = self.gcp.list_tpus(
            project=self.fleet.project,
            additional_targets=tuple(
                (resource.tpu_name, resource.zone)
                for resource in inventory_resources
            ),
        )
        queued = self.gcp.list_queued_resources(
            project=self.fleet.project,
            additional_targets=tuple(
                (f"qr-{resource.tpu_name}", resource.zone)
                for resource in inventory_resources
                if not resource.adopted
            ),
        )
        return any(
            resource_matches_entry(resource, entry)
            for entry in managed_entries
            for resource in [*tpus, *queued]
        )

    def assign_pending_jobs(self) -> None:
        resources = [
            resource
            for resource in self.store.list_resources()
            if resource.status == "idle"
            and resource_is_declared_for_fleet(resource, self.fleet)
        ]
        jobs = self.store.list_jobs_with_statuses({"pending"})
        for job, resource in plan_idle_assignments(
            jobs,
            resources,
            entries=list(self.fleet.tpus),
        ):
            attempt = self.store.assign_job(job.spec.id, resource.id)
            if attempt is None:
                continue
            selected_region = region_from_zone(resource.zone)
            self.store.record_event(
                "job_assigned",
                {
                    "job_id": job.spec.id,
                    "attempt_id": attempt.id,
                    "resource_id": resource.id,
                    "tpu_name": resource.tpu_name,
                    "region": selected_region,
                    "bucket": job.spec.bucket_for_region(selected_region),
                    "priority": job.spec.priority,
                },
            )

    def reconcile_controller_managed_attempts(self) -> None:
        from .distributed import DistributedTPURunner, TemporaryAccessError

        runner = DistributedTPURunner(
            name=self.fleet.name,
            project=self.fleet.project,
            ssh_transport=self.fleet.ssh_transport,
        )
        for job in self.store.list_jobs_with_statuses({"running"}):
            self.heartbeat()
            if not job.current_attempt_id or not job.assigned_resource_id:
                continue
            attempt = self.store.get_attempt(job.current_attempt_id)
            resource = self.store.get_resource(job.assigned_resource_id)
            if attempt is None or resource is None:
                continue
            if (
                attempt.id != job.current_attempt_id
                or attempt.job_id != job.spec.id
                or attempt.resource_id != resource.id
            ):
                continue
            try:
                if attempt.status == "launching":
                    existing_result = runner.poll(job=job.spec, attempt=attempt, resource=resource)
                    if not existing_result.complete:
                        try:
                            runner.launch(job=job.spec, attempt=attempt, resource=resource)
                        except TemporaryAccessError:
                            raise
                        except Exception as exc:
                            self.mark_attempt_command_failed(
                                attempt.id,
                                exit_code=2,
                                error_summary=f"failed_setup: {str(exc)[-4000:]}",
                            )
                            continue
                    attempt.status = "running"
                    if not attempt.started_at:
                        attempt.started_at = utc_now_iso()
                    self.store.upsert_attempt(attempt)
                    self.store.record_event(
                        "distributed_job_launched",
                        {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource.id},
                    )
                if attempt.status == "running":
                    result = runner.poll(job=job.spec, attempt=attempt, resource=resource)
                    if result.complete:
                        if result.exit_code == 0:
                            self.mark_attempt_succeeded(attempt.id)
                        else:
                            all_workers_terminal = len(result.statuses) >= max(
                                1, resource.worker_count
                            )
                            if all_workers_terminal:
                                cancel_outcome = "terminal_statuses"
                            else:
                                cancel_outcome = runner.cancel(
                                    attempt=attempt, resource=resource
                                )
                                if cancel_outcome == "attempt_mismatch":
                                    self.store.record_event(
                                        "distributed_attempt_cancel_mismatch",
                                        {
                                            "job_id": job.spec.id,
                                            "attempt_id": attempt.id,
                                            "resource_id": resource.id,
                                        },
                                    )
                                    continue
                            self.store.record_event(
                                "distributed_attempt_stopped",
                                {
                                    "job_id": job.spec.id,
                                    "attempt_id": attempt.id,
                                    "resource_id": resource.id,
                                    "outcome": cancel_outcome,
                                },
                            )
                            if result.failure_kind == "retryable_infrastructure":
                                self.mark_attempt_retryable_infrastructure(
                                    attempt.id,
                                    exit_code=result.exit_code or 75,
                                    error_summary=result.error_summary or result.failure_kind,
                                )
                            else:
                                self.mark_attempt_command_failed(
                                    attempt.id,
                                    exit_code=result.exit_code or 1,
                                    error_summary=result.error_summary or result.failure_kind,
                                )
            except TemporaryAccessError as exc:
                self.store.record_event(
                    "controller_temporary_access_error",
                    {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource.id, "error": str(exc)},
                )
                entry = next(
                    (
                        candidate
                        for candidate in self.fleet.tpus
                        if candidate.id == (resource.fleet_entry_id or "")
                    ),
                    None,
                )
                age_seconds = attempt_age_seconds(attempt)
                if (
                    attempt.status == "launching"
                    and age_seconds is not None
                    and age_seconds >= MANAGED_SPOT_LAUNCH_ACCESS_TIMEOUT_SECONDS
                    and entry is not None
                    and not entry.adopted
                    and entry.provisioning_model == "spot"
                    and not resource.adopted
                ):
                    try:
                        request = self.store.create_interruption_request(
                            resource_id=resource.id,
                            job_id=job.spec.id,
                            attempt_id=attempt.id,
                            eligible_fleet_entry_ids={entry.id},
                        )
                    except ValueError:
                        # The exact request may already exist from a prior
                        # controller pass or an operator.  The interruption
                        # reconciliation phase remains authoritative.
                        pass
                    else:
                        self.store.record_event(
                            "stale_launch_interruption_requested",
                            {
                                "job_id": job.spec.id,
                                "attempt_id": attempt.id,
                                "resource_id": resource.id,
                                "request_id": request.id,
                                "age_seconds": int(age_seconds),
                            },
                        )
                    return
                # If an exact controlled interruption arrived while this SSH
                # call was blocked, return to reconcile_once immediately.
                # That lets the interruption phase consume it before another
                # launching job incurs the same bounded timeout.
                if self.store.list_interruption_requests_with_statuses({"requested"}):
                    return
            except Exception as exc:
                self.mark_attempt_command_failed(attempt.id, exit_code=1, error_summary=str(exc))

    def mark_attempt_command_failed(self, attempt_id: str, *, exit_code: int, error_summary: str = "") -> None:
        attempt = self.store.finish_attempt(
            attempt_id,
            attempt_status="failed_setup" if error_summary.startswith("failed_setup") else "failed",
            job_status="failed",
            resource_status="idle",
            exit_code=exit_code,
            error_summary=error_summary,
            end_reason=(
                "setup_failed"
                if error_summary.startswith("failed_setup")
                else "application_failed"
            ),
        )
        if attempt is None:
            return
        self.store.record_event(
            "job_failed",
            {"job_id": attempt.job_id, "attempt_id": attempt.id, "exit_code": exit_code},
        )

    def mark_attempt_retryable_infrastructure(
        self, attempt_id: str, *, exit_code: int, error_summary: str
    ) -> None:
        attempt_before = self.store.get_attempt(attempt_id)
        resource = None
        entry = None
        if attempt_before is not None:
            resource = self.store.get_resource(attempt_before.resource_id)
        if resource is not None:
            entry = next(
                (
                    candidate
                    for candidate in self.fleet.tpus
                    if candidate.id == (resource.fleet_entry_id or "")
                ),
                None,
            )
        recycle_threshold = (
            2
            if resource is not None
            and entry is not None
            and not resource.adopted
            and not entry.adopted
            and entry.provisioning_model == "spot"
            and resource_is_declared_for_fleet(resource, self.fleet)
            else 0
        )
        attempt = self.store.finish_attempt(
            attempt_id,
            attempt_status="interrupted",
            job_status="pending",
            resource_status="idle",
            exit_code=exit_code,
            error_summary=error_summary,
            end_reason="lost",
            retryable_infrastructure_failure=True,
            retryable_infrastructure_recycle_threshold=recycle_threshold,
        )
        if attempt is None:
            return
        self.store.record_event(
            "job_interrupted",
            {
                "job_id": attempt.job_id,
                "attempt_id": attempt.id,
                "resource_id": attempt.resource_id,
                "reason": "retryable_infrastructure",
                "exit_code": exit_code,
            },
        )
        if recycle_threshold:
            refreshed = self.store.get_resource(attempt.resource_id)
            if refreshed is not None and refreshed.status == "recycling":
                self.store.record_event(
                    "retryable_infrastructure_recycle_scheduled",
                    {
                        "job_id": attempt.job_id,
                        "attempt_id": attempt.id,
                        "resource_id": attempt.resource_id,
                        "failure_count": refreshed.retryable_infrastructure_failures,
                    },
                )

    def mark_attempt_succeeded(self, attempt_id: str) -> None:
        attempt = self.store.finish_attempt(
            attempt_id,
            attempt_status="succeeded",
            job_status="succeeded",
            resource_status="idle",
            end_reason="succeeded",
        )
        if attempt is None:
            return
        self.store.record_event("job_succeeded", {"job_id": attempt.job_id, "attempt_id": attempt.id})

    def mark_tpu_preempted(self, resource_id: str) -> None:
        resource = self.store.get_resource(resource_id)
        if resource is None:
            raise KeyError(resource_id)
        if resource.current_attempt_id:
            attempt = self.store.get_attempt(resource.current_attempt_id)
            if attempt:
                job = self.store.get_job(attempt.job_id)
                if job is None:
                    raise KeyError(attempt.job_id)
                if job.status == "cancelling":
                    finished = self.store.finish_attempt(
                        attempt.id,
                        attempt_status="cancelled",
                        job_status="deactivated",
                        resource_status="preempted",
                        end_reason="cancelled",
                        expected_job_status="cancelling",
                    )
                    if finished:
                        self.store.record_event(
                            "job_cancelled",
                            {"job_id": job.spec.id, "attempt_id": attempt.id, "resource_id": resource_id},
                        )
                    return
                finished = self.store.finish_attempt(
                    attempt.id,
                    attempt_status="interrupted",
                    job_status="pending",
                    resource_status="preempted",
                    end_reason="preempted",
                )
                if finished:
                    self.store.record_event(
                        "job_interrupted",
                        {
                            "job_id": job.spec.id,
                            "attempt_id": attempt.id,
                            "resource_id": resource_id,
                        },
                    )
                return
        resource.status = "preempted"
        resource.current_job_id = None
        resource.current_attempt_id = None
        resource.retryable_infrastructure_failures = 0
        resource.retryable_infrastructure_job_id = ""
        reset_cancellation_tracking(resource)
        resource.idle_since = ""
        self.store.upsert_resource(resource)

    def _upsert_inventory_resource(self, incoming: ResourceRecord) -> None:
        existing = self.store.get_resource(incoming.id)
        if existing:
            existing.tpu_name = incoming.tpu_name
            existing.zone = incoming.zone
            existing.tpu_type = incoming.tpu_type
            existing.worker_count = incoming.worker_count
            existing.fleet_entry_id = incoming.fleet_entry_id
            existing.adopted = incoming.adopted
            if existing.current_attempt_id:
                existing.status = "busy"
            elif existing.status in {
                "deleted",
                "preempted",
                "provisioning",
                "unavailable",
            }:
                existing.status = "idle"
                existing.idle_since = "" if incoming.adopted else utc_now_iso()
                existing.retryable_infrastructure_failures = 0
                existing.retryable_infrastructure_job_id = ""
                reset_cancellation_tracking(existing)
            self.store.upsert_resource(existing)
        else:
            self.store.upsert_resource(incoming)


def resource_record_from_tpu(tpu: TPUVM, fleet_entry_id: str, *, adopted: bool) -> ResourceRecord:
    resource_id = f"adopted-{tpu.name}" if adopted else tpu.name
    return ResourceRecord(
        id=resource_id,
        tpu_name=tpu.name,
        zone=tpu.zone,
        tpu_type=tpu.accelerator_type,
        fleet_entry_id=fleet_entry_id,
        adopted=adopted,
        worker_count=tpu.worker_count,
        idle_since="" if adopted else utc_now_iso(),
    )


def resource_is_declared_for_fleet(resource: ResourceRecord, fleet: FleetSpec) -> bool:
    """Exclude stale managed ordinals before they can receive a new job."""
    entries = {entry.id: entry for entry in fleet.tpus}
    entry = entries.get(resource.fleet_entry_id or "")
    if entry is None or resource.tpu_type != entry.type or resource.zone != entry.zone:
        return False
    if entry.adopted:
        return resource.adopted and resource.tpu_name == entry.existing
    if resource.adopted:
        return False
    return resource.tpu_name in {
        generated_resource_names(entry, ordinal)[1]
        for ordinal in range(1, entry.count + 1)
    }


def resource_is_inventory_owned_for_fleet(
    resource: ResourceRecord,
    fleet: FleetSpec,
) -> bool:
    """Allow exact discovery only for current entry identities.

    Managed records above a newly lowered ordinal ceiling remain discoverable
    so reconciliation can drain them. Adopted or removed-entry records are not
    included.
    """

    entry = next(
        (
            candidate
            for candidate in fleet.tpus
            if candidate.id == (resource.fleet_entry_id or "")
        ),
        None,
    )
    if (
        entry is None
        or resource.zone != entry.zone
        or resource.tpu_type != entry.type
    ):
        return False
    if entry.adopted:
        return resource.adopted and resource.tpu_name == entry.existing
    return not resource.adopted


def queued_resource_name_for_node(entry, node_id: str) -> str:
    matches = [
        queued_name
        for ordinal in range(1, entry.count + 1)
        for queued_name, declared_node_id in [generated_resource_names(entry, ordinal)]
        if declared_node_id == node_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"node {node_id!r} is not exactly one declared ordinal for {entry.id!r}"
        )
    return matches[0]


def interruption_event_payload(
    request: InterruptionRequestRecord,
    queued_resource_name: str = "",
    *,
    error: str = "",
) -> dict[str, str]:
    payload = {
        "request_id": request.id,
        "resource_id": request.resource_id,
        "job_id": request.job_id,
        "attempt_id": request.attempt_id,
        "fleet_entry_id": request.fleet_entry_id,
    }
    if queued_resource_name:
        payload["queued_resource"] = queued_resource_name
    if error:
        payload["error"] = error
    return payload


def reserved_chips(entry, *, tpus: list[TPUVM], queued: list[QueuedResource]) -> int:
    family, _ = tpu_family_and_chips(entry.type)
    terminal_queued_names = {
        qr.name for qr in queued if qr.terminal and resource_matches_entry(qr, entry)
    }
    terminal_queued_node_ids = {
        qr.node_id
        for qr in queued
        if qr.node_id and qr.terminal and resource_matches_entry(qr, entry)
    }
    node_ids: set[str] = set()
    total = 0
    for tpu in tpus:
        if (
            tpu.terminal
            or tpu.name in terminal_queued_node_ids
            or (tpu.queued_resource and tpu.queued_resource in terminal_queued_names)
        ):
            continue
        try:
            tpu_family, chips = tpu_family_and_chips(tpu.accelerator_type)
        except ValueError:
            continue
        model = "spot" if tpu.spot else "on-demand"
        if tpu.zone == entry.zone and tpu_family == family and model == entry.provisioning_model:
            total += chips
            node_ids.add(tpu.name)
    for qr in queued:
        if not qr.pending_or_active:
            continue
        if qr.node_id in node_ids:
            continue
        try:
            qr_family, chips = tpu_family_and_chips(qr.accelerator_type)
        except ValueError:
            continue
        if (
            qr.zone == entry.zone
            and qr_family == family
            and qr.provisioning_model.lower() == entry.provisioning_model
        ):
            total += chips
    return total


def quota_key(entry) -> tuple[str, str, str]:
    family, _ = tpu_family_and_chips(entry.type)
    return entry.provisioning_model, family, entry.zone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
