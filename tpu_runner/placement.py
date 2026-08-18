from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from urllib.parse import quote

from .specs import JobSpec


_GCS_BUCKET_RE = re.compile(r"gs://(?P<bucket>[a-z0-9][a-z0-9._-]*[a-z0-9])")
_REGION_RE = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+[0-9]$")
BUCKET_LOCATION_TIMEOUT_SECONDS = 90


def gcs_buckets_in_values(values: Iterable[str]) -> tuple[str, ...]:
    """Return every distinct GCS bucket referenced by arbitrary job strings."""

    return tuple(
        sorted(
            {
                match.group("bucket")
                for value in values
                for match in _GCS_BUCKET_RE.finditer(value)
            }
        )
    )


def normalize_exact_region(location: str) -> str:
    value = location.strip().lower().replace("_", "-")
    if not _REGION_RE.fullmatch(value):
        raise ValueError(
            f"bucket location {location.strip()!r} is not one exact GCP region"
        )
    return value


def resolve_regional_buckets(
    buckets: Iterable[str],
    *,
    resolve_bucket_location: Callable[[str], str],
) -> tuple[tuple[str, str], ...]:
    """Resolve one existing job bucket per exact candidate region."""

    resolved: dict[str, str] = {}
    for bucket_uri in buckets:
        bucket_name = bucket_uri.removeprefix("gs://")
        region = normalize_exact_region(resolve_bucket_location(bucket_name))
        if region in resolved:
            raise ValueError(
                f"job buckets {resolved[region]!r} and {bucket_uri!r} are both "
                f"in {region}; declare only one bucket per region"
            )
        resolved[region] = bucket_uri.rstrip("/")
    return tuple(sorted(resolved.items()))


def validate_job_gcs_dependencies(
    job: JobSpec,
    *,
    bucket_regions: tuple[tuple[str, str], ...],
    resolve_bucket_location: Callable[[str], str],
) -> None:
    """Ensure literal GCS references cannot defeat regional placement."""

    referenced = gcs_buckets_in_values([job.bundle, job.command, *job.env.values()])
    if len(bucket_regions) > 1:
        if referenced:
            detail = ", ".join(f"gs://{bucket}" for bucket in referenced)
            raise ValueError(
                f"multi-region job {job.id!r} contains literal GCS reference(s): "
                f"{detail}; use JOB_BUCKET so the command selects the bucket local "
                "to the winning TPU"
            )
        return
    expected_region = bucket_regions[0][0]
    mismatched: dict[str, str] = {}
    for storage_bucket in referenced:
        region = normalize_exact_region(resolve_bucket_location(storage_bucket))
        if region != expected_region:
            mismatched[storage_bucket] = region
    if mismatched:
        detail = ", ".join(
            f"gs://{storage_bucket}={region}"
            for storage_bucket, region in sorted(mismatched.items())
        )
        raise ValueError(
            f"job {job.id!r} references GCS outside {expected_region}: {detail}"
        )


def gcloud_bucket_location(
    bucket: str,
    *,
    project: str,
    credentials: object | None = None,
) -> str:
    """Read one bucket location without forking a live Firestore gRPC process."""

    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    session = None
    try:
        if credentials is None:
            credentials, _ = google.auth.default(
                scopes=("https://www.googleapis.com/auth/devstorage.read_only",)
            )
        if project and hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(project)
        session = AuthorizedSession(credentials)
        response = session.get(
            "https://storage.googleapis.com/storage/v1/b/"
            + quote(bucket, safe=""),
            params={"fields": "location"},
            timeout=BUCKET_LOCATION_TIMEOUT_SECONDS,
        )
        if not response.ok:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise RuntimeError(
                f"bucket location lookup failed for gs://{bucket}: {detail}"
            )
        location = response.json().get("location", "")
        if not isinstance(location, str) or not location.strip():
            raise RuntimeError(
                f"bucket location lookup failed for gs://{bucket}: missing location"
            )
        return location
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"bucket location lookup failed for gs://{bucket}: {exc}"
        ) from exc
    finally:
        if session is not None:
            session.close()
