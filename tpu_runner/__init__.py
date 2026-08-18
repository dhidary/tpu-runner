"""GCP TPU fleet and job orchestration helpers."""

from importlib.metadata import PackageNotFoundError, version

from .specs import FleetSpec, JobSpec, load_fleet_spec, load_job_specs

try:
    __version__ = version("tpu-runner")
except PackageNotFoundError:  # Source tree imported without installation.
    __version__ = "0+unknown"

__all__ = [
    "FleetSpec",
    "JobSpec",
    "__version__",
    "load_fleet_spec",
    "load_job_specs",
]
