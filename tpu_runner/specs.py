from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .region_pools import logical_region_pool, region_is_in_pool


_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TPU_TYPE_RE = re.compile(
    r"^(?P<family>[a-z][a-z0-9]*)-(?P<chips>[1-9][0-9]*)$"
)
_REGION_RE = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+[0-9]$")
_ZONE_RE = re.compile(r"^(?P<region>[a-z]+(?:-[a-z0-9]+)+[0-9])-[a-z]$")
_GCS_BUCKET_URI_RE = re.compile(r"^gs://[^/\s]+$")
JOB_PRIORITY_CLASSES = ("low", "normal", "high")
DEFAULT_RUNNER_BUCKET_LOCATION = "us-central2"
JOB_SPEC_FIELDS = frozenset(
    {
        "id",
        "tpu",
        "bundle",
        "command",
        "tpu_name",
        "zone",
        "buckets",
        "caches",
        "env",
        "priority",
    }
)
STORED_JOB_SPEC_FIELDS = JOB_SPEC_FIELDS | {
    "region",
    "storage_region",
    "bucket",
    "bucket_regions",
    "regional_bundles",
}
_RUNNER_ENV_NAMES = {
    "HOST",
    "WORK_ROOT",
    "SHM_ROOT",
    "CACHE_ROOT",
    "JOB_DIR",
    "ATTEMPT_DIR",
    "SHM_DIR",
    "PROCESS_DIR",
    "PID_FILE",
    "CURRENT_ATTEMPT_ID",
    "JOB_ID",
    "ATTEMPT_ID",
    "JOB_SHM_DIR",
    "JOB_GCS_DIR",
    "JOB_BUCKET",
    "ATTEMPT_GCS_DIR",
    "CHECKPOINT_GCS_DIR",
    "RUNNER_PROJECT",
    "BUNDLE_SHA256",
    "TPU_WORKER_COUNT",
}


def slugify(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    value = _SLUG_RE.sub("-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "item"


def stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha1(encoded).hexdigest()[:10]}"


def default_runner_bucket(project: str, name: str) -> str:
    """Derive one stable, globally unique runner bucket URI."""

    value = slugify(f"{project}-{name}-{DEFAULT_RUNNER_BUCKET_LOCATION}")
    if len(value) > 63:
        digest = hashlib.sha1(value.encode()).hexdigest()[:10]
        value = f"{value[:52].rstrip('-')}-{digest}"
    return f"gs://{value}"


@dataclass(frozen=True)
class TPUEntry:
    id: str
    type: str
    zone: str
    count: int = 0
    existing: str | None = None
    runtime: str | None = None
    provisioning_model: str = "spot"
    chip_limit: int = 0
    runner_name: str = ""
    keep_warm_count: int = 0

    @property
    def adopted(self) -> bool:
        return self.existing is not None


@dataclass(frozen=True)
class FleetSpec:
    name: str
    project: str
    bucket: str
    controller_region: str
    controller_timeout: str
    firestore_location: str
    network: str
    subnetwork: str
    worker_secrets: tuple[str, ...]
    tpus: tuple[TPUEntry, ...]
    controller_memory: str = "1Gi"
    controller_max_retries: int = 3
    ssh_transport: str = "direct"
    idle_timeout_seconds: int = 600

@dataclass(frozen=True)
class CacheSpec:
    key: str
    path: str

    def __post_init__(self) -> None:
        if not self.key or "/" in self.key or self.key in {".", ".."}:
            raise ValueError(f"cache key must be one path component: {self.key!r}")
        path = PurePosixPath(self.path)
        if not self.path or not path.parts or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"cache path must stay within the job directory: {self.path!r}")


@dataclass(frozen=True)
class JobSpec:
    id: str
    tpu: str | tuple[str, ...]
    bundle: str
    command: str
    tpu_name: str = ""
    zone: str = ""
    buckets: tuple[str, ...] = field(default_factory=tuple)
    bucket_regions: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    regional_bundles: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    region: str = ""
    storage_region: str = ""
    bucket: str = ""
    caches: tuple[CacheSpec, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)
    priority: str = "normal"

    def __post_init__(self) -> None:
        if not isinstance(self.priority, str):
            raise ValueError("job priority must be a string")
        priority = self.priority.strip().lower()
        if priority not in JOB_PRIORITY_CLASSES:
            raise ValueError(
                "job priority must be one of: " + ", ".join(JOB_PRIORITY_CLASSES)
            )
        object.__setattr__(self, "priority", priority)
        if not isinstance(self.zone, str):
            raise ValueError("job zone must be a string")
        if not isinstance(self.region, str):
            raise ValueError("job region must be a string")
        if not isinstance(self.storage_region, str):
            raise ValueError("job storage_region must be a string")
        if self.storage_region and not _REGION_RE.fullmatch(self.storage_region):
            raise ValueError(f"invalid job storage_region: {self.storage_region!r}")
        if self.region:
            pool = logical_region_pool(self.region)
            object.__setattr__(self, "region", pool)
        if self.zone:
            zone_match = _ZONE_RE.fullmatch(self.zone)
            if zone_match is None:
                raise ValueError(f"invalid job zone: {self.zone!r}")
            if self.storage_region and zone_match.group("region") != self.storage_region:
                raise ValueError(
                    f"job zone {self.zone!r} is outside selected region "
                    f"{self.storage_region!r}"
                )
        if self.storage_region and self.region and not region_is_in_pool(
            self.storage_region, self.region
        ):
            raise ValueError("job storage_region is outside its logical region pool")
        if not self.buckets:
            raise ValueError("job requires at least one regional bucket")
        if len(set(self.buckets)) != len(self.buckets):
            raise ValueError("job buckets must be unique")
        invalid_buckets = [
            value for value in self.buckets if not _GCS_BUCKET_URI_RE.fullmatch(value)
        ]
        if invalid_buckets:
            raise ValueError(
                "job buckets must be gs:// bucket URIs without object paths: "
                + ", ".join(invalid_buckets)
            )
        region_keys = [region for region, _ in self.bucket_regions]
        if len(set(region_keys)) != len(region_keys):
            raise ValueError("job may declare only one bucket per exact region")
        for exact_region, regional_bucket in self.bucket_regions:
            if not _REGION_RE.fullmatch(exact_region):
                raise ValueError(f"invalid regional bucket region: {exact_region!r}")
            if regional_bucket not in self.buckets:
                raise ValueError(
                    f"regional bucket {regional_bucket!r} is not declared by the job"
                )
        bundle_regions = [region for region, _ in self.regional_bundles]
        if len(set(bundle_regions)) != len(bundle_regions):
            raise ValueError("job may materialize only one bundle per exact region")
        if self.regional_bundles and set(bundle_regions) != set(region_keys):
            raise ValueError("job regional bundles must exactly cover its bucket regions")
        if self.storage_region:
            regional_buckets = dict(self.bucket_regions)
            regional_bundle_uris = dict(self.regional_bundles)
            if regional_buckets.get(self.storage_region) != self.bucket:
                raise ValueError("selected job bucket does not match selected region")
            if regional_bundle_uris.get(self.storage_region) != self.bundle:
                raise ValueError("selected job bundle does not match selected region")
        elif self.bucket:
            raise ValueError("job bucket cannot be selected without a storage region")
        invalid = sorted(key for key in self.env if not _ENV_NAME_RE.fullmatch(key))
        if invalid:
            raise ValueError(f"invalid job environment variable name(s): {', '.join(invalid)}")
        reserved = sorted(self.env.keys() & _RUNNER_ENV_NAMES)
        if reserved:
            raise ValueError(f"job environment cannot override runner variable(s): {', '.join(reserved)}")

    def accepts_tpu_type(self, tpu_type: str) -> bool:
        if self.tpu == "any":
            return True
        if isinstance(self.tpu, tuple):
            return tpu_type in self.tpu
        return self.tpu == tpu_type

    def accepts_region(self, exact_region: str) -> bool:
        if self.storage_region:
            return exact_region == self.storage_region
        return exact_region in dict(self.bucket_regions)

    def bucket_for_region(self, exact_region: str) -> str:
        return dict(self.bucket_regions).get(exact_region, "")

    def bundle_for_region(self, exact_region: str) -> str:
        return dict(self.regional_bundles).get(exact_region, "")


def load_fleet_spec(path: str | Path) -> FleetSpec:
    return fleet_spec_from_dict(load_yaml_file(path))


def load_job_specs(path: str | Path) -> tuple[JobSpec, ...]:
    data = load_yaml_file(path)
    return job_specs_from_dict(data)


def fleet_spec_from_dict(data: dict[str, Any]) -> FleetSpec:
    required = [
        "name",
        "project",
        "controller_region",
        "controller_timeout",
        "firestore_location",
        "network",
        "subnetwork",
        "tpus",
    ]
    allowed = set(required) | {
        "bucket",
        "controller_memory",
        "controller_max_retries",
        "idle_timeout_seconds",
        "ssh_transport",
        "worker_secrets",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            "fleet spec has unknown field(s): " + ", ".join(unknown)
        )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"fleet spec missing required field(s): {', '.join(missing)}")
    tpus_raw = data["tpus"]
    if not isinstance(tpus_raw, list) or not tpus_raw:
        raise ValueError("fleet spec field 'tpus' must be a non-empty list")
    name = str(data["name"]).strip()
    if not name or slugify(name) != name:
        raise ValueError("deployment name must be a lowercase hyphenated slug")
    project = str(data["project"]).strip()
    bucket_raw = str(data.get("bucket", "")).rstrip("/")
    bucket = bucket_raw or default_runner_bucket(project, name)
    if not _GCS_BUCKET_URI_RE.fullmatch(bucket):
        raise ValueError("fleet bucket must be a gs:// bucket URI without an object path")
    worker_secrets_raw = data.get("worker_secrets", [])
    if not isinstance(worker_secrets_raw, list) or any(not str(item).strip() for item in worker_secrets_raw):
        raise ValueError("worker_secrets must be a list of non-empty secret names")
    controller_memory = str(data.get("controller_memory", "1Gi")).strip()
    if not re.fullmatch(r"[1-9][0-9]*(?:Mi|Gi)", controller_memory):
        raise ValueError("controller_memory must use a positive Mi or Gi value")
    controller_max_retries_raw = data.get("controller_max_retries", 3)
    if (
        isinstance(controller_max_retries_raw, bool)
        or not isinstance(controller_max_retries_raw, int)
        or not 0 <= controller_max_retries_raw <= 10
    ):
        raise ValueError("controller_max_retries must be an integer from 0 through 10")
    ssh_transport = str(data.get("ssh_transport", "direct")).strip().lower()
    if ssh_transport not in {"iap", "direct"}:
        raise ValueError("ssh_transport must be 'iap' or 'direct'")
    idle_timeout_seconds_raw = data.get("idle_timeout_seconds", 600)
    if (
        isinstance(idle_timeout_seconds_raw, bool)
        or not isinstance(idle_timeout_seconds_raw, int)
        or idle_timeout_seconds_raw < 0
    ):
        raise ValueError("idle_timeout_seconds must be a non-negative integer")

    entries: list[TPUEntry] = []
    used_ids: set[str] = set()
    requested_chips: dict[tuple[str, str, str], int] = {}
    limits: dict[tuple[str, str, str], int] = {}
    for index, raw in enumerate(tpus_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each TPU entry must be a mapping")
        allowed_entry_fields = {
            "id",
            "type",
            "zone",
            "count",
            "existing",
            "runtime",
            "provisioning_model",
            "chip_limit",
            "keep_warm_count",
        }
        unknown = sorted(set(raw) - allowed_entry_fields)
        if unknown:
            raise ValueError(
                "TPU entry has unknown field(s): " + ", ".join(unknown)
            )
        if "zone" not in raw or "type" not in raw:
            raise ValueError("each TPU entry requires 'zone' and 'type'")
        if isinstance(raw.get("zone"), list):
            raise ValueError("TPU entry 'zone' must be a single zone, not a list")

        tpu_type = str(raw["type"]).strip()
        if not tpu_type:
            raise ValueError("TPU entry 'type' must be non-empty")
        family, chips = tpu_family_and_chips(tpu_type)
        provisioning_model = str(raw.get("provisioning_model", "spot")).strip().lower()
        if provisioning_model not in {"spot", "on-demand"}:
            raise ValueError("TPU entry provisioning_model must be spot or on-demand")
        chip_limit = int(raw.get("chip_limit", 0))
        if chip_limit <= 0:
            raise ValueError("each TPU entry requires chip_limit > 0")
        existing = raw.get("existing")
        keep_warm_count = raw.get("keep_warm_count", 0)
        if isinstance(keep_warm_count, bool) or not isinstance(
            keep_warm_count, int
        ):
            raise ValueError("TPU entry 'keep_warm_count' must be an integer")
        if keep_warm_count < 0:
            raise ValueError("TPU entry 'keep_warm_count' must be >= 0")
        if existing:
            if keep_warm_count:
                raise ValueError("adopted TPU entries cannot set keep_warm_count")
            count = 0
        else:
            if "count" not in raw:
                raise ValueError("created TPU entries require 'count'")
            count = int(raw["count"])
            if count < 0:
                raise ValueError("TPU entry 'count' must be >= 0")
            if count > 0 and not raw.get("runtime"):
                raise ValueError("created TPU entries with count > 0 require 'runtime'")
            if provisioning_model != "spot":
                raise ValueError("created TPU entries currently support only spot capacity")
            if keep_warm_count > count:
                raise ValueError("TPU entry 'keep_warm_count' cannot exceed count")

        quota_key = (provisioning_model, family, str(raw["zone"]))
        if quota_key in limits and limits[quota_key] != chip_limit:
            raise ValueError(f"inconsistent chip_limit for {quota_key}")
        limits[quota_key] = chip_limit
        requested_chips[quota_key] = requested_chips.get(quota_key, 0) + chips * (1 if existing else count)

        raw_id = raw.get("id")
        entry_id = str(raw_id) if raw_id else f"{slugify(raw['type'])}-{slugify(raw['zone'])}-{index}"
        if entry_id in used_ids:
            raise ValueError(f"duplicate TPU entry id: {entry_id}")
        used_ids.add(entry_id)
        entries.append(
            TPUEntry(
                id=entry_id,
                type=tpu_type,
                zone=str(raw["zone"]),
                count=count,
                existing=str(existing) if existing else None,
                runtime=str(raw["runtime"]) if raw.get("runtime") else None,
                provisioning_model=provisioning_model,
                chip_limit=chip_limit,
                runner_name=name,
                keep_warm_count=keep_warm_count,
            )
        )

    exceeded = [
        f"{model} {family} {zone}: requested {requested_chips[key]} > limit {limit}"
        for key, limit in limits.items()
        for model, family, zone in [key]
        if requested_chips[key] > limit
    ]
    if exceeded:
        raise ValueError("fleet exceeds TPU chip limit(s): " + "; ".join(exceeded))

    return FleetSpec(
        name=name,
        project=project,
        bucket=bucket,
        controller_region=str(data["controller_region"]),
        controller_timeout=str(data["controller_timeout"]),
        firestore_location=str(data["firestore_location"]),
        network=str(data["network"]),
        subnetwork=str(data["subnetwork"]),
        worker_secrets=tuple(str(item).strip() for item in worker_secrets_raw),
        tpus=tuple(entries),
        controller_memory=controller_memory,
        controller_max_retries=controller_max_retries_raw,
        ssh_transport=ssh_transport,
        idle_timeout_seconds=idle_timeout_seconds_raw,
    )


def job_specs_from_dict(data: dict[str, Any]) -> tuple[JobSpec, ...]:
    unknown = sorted(set(data) - {"jobs"})
    if unknown:
        raise ValueError(
            "job spec has unknown top-level field(s): " + ", ".join(unknown)
        )
    jobs_raw = data.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("job spec field 'jobs' must be a non-empty list")

    jobs: list[JobSpec] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(jobs_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each job entry must be a mapping")
        unknown = sorted(set(raw) - JOB_SPEC_FIELDS)
        if unknown:
            raise ValueError(
                "job entry has unknown field(s): " + ", ".join(unknown)
            )
        for key in ("tpu", "command", "buckets"):
            if key not in raw:
                raise ValueError(f"job entry missing required field: {key}")

        raw_tpu = raw["tpu"]
        if isinstance(raw_tpu, list):
            if not raw_tpu:
                raise ValueError("job entry field 'tpu' must be a non-empty list")
            tpu_values = tuple(str(item).strip() for item in raw_tpu)
            if any(not item for item in tpu_values):
                raise ValueError("job entry field 'tpu' cannot contain an empty type")
            tpu: str | tuple[str, ...] = tpu_values
        else:
            tpu = str(raw_tpu).strip()
            if not tpu:
                raise ValueError("job entry field 'tpu' must be non-empty")

        raw_id = raw.get("id")
        job_id = str(raw_id) if raw_id else stable_id("job", {"index": index, **raw})
        if job_id in used_ids:
            raise ValueError(f"duplicate job id: {job_id}")
        used_ids.add(job_id)

        raw_buckets = raw["buckets"]
        if not isinstance(raw_buckets, list) or not raw_buckets:
            raise ValueError("job entry field 'buckets' must be a non-empty list")
        buckets = tuple(str(value).rstrip("/") for value in raw_buckets)

        caches = tuple(
            CacheSpec(key=str(cache["key"]), path=str(cache["path"]))
            for cache in raw.get("caches", [])
        )
        raw_env = raw.get("env", {})
        if raw_env is None:
            raw_env = {}
        if not isinstance(raw_env, dict):
            raise ValueError("job entry field 'env' must be a mapping")
        raw_zone = raw.get("zone", "")
        if not isinstance(raw_zone, str):
            raise ValueError("job zone must be a string")
        jobs.append(
            JobSpec(
                id=job_id,
                tpu=tpu,
                bundle=str(raw.get("bundle", ".")),
                command=str(raw["command"]),
                tpu_name=str(raw.get("tpu_name", "")).strip(),
                zone=raw_zone.strip(),
                buckets=buckets,
                priority=raw.get("priority", "normal"),
                caches=caches,
                env={str(key): str(value) for key, value in raw_env.items()},
            )
        )
    return tuple(jobs)


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    import yaml

    try:
        loaded = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from None
    if not isinstance(loaded, dict):
        raise ValueError("YAML document must be a mapping")
    return loaded


def tpu_family_and_chips(tpu_type: str) -> tuple[str, int]:
    match = _TPU_TYPE_RE.fullmatch(tpu_type)
    if not match:
        raise ValueError(f"invalid TPU accelerator type: {tpu_type}")
    family = match.group("family")
    chips = int(match.group("chips"))
    if family == "v4":
        # TPU v4 accelerator names count TensorCores; each v4 chip contains two.
        chips //= 2
    if family == "v5litepod":
        family = "v5e"
    return family, chips


def region_from_zone(zone: str) -> str:
    match = _ZONE_RE.fullmatch(zone)
    if match is None:
        raise ValueError(f"invalid TPU zone: {zone!r}")
    return match.group("region")
