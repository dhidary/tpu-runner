"""Logical placement pools layered over exact GCP storage regions."""

from __future__ import annotations

import re


_EXACT_REGION_RE = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+[0-9]$")
_LOGICAL_REGION_POOL_RE = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+$")


def logical_region_pool(value: str) -> str:
    """Normalize a logical pool or exact GCP region to one pool name."""

    normalized = value.strip().lower().replace("_", "-")
    if _EXACT_REGION_RE.fullmatch(normalized):
        return normalized.rstrip("0123456789")
    if _LOGICAL_REGION_POOL_RE.fullmatch(normalized):
        return normalized
    raise ValueError(f"invalid logical region pool: {value!r}")


def region_is_in_pool(exact_region: str, pool: str) -> bool:
    return logical_region_pool(exact_region) == logical_region_pool(pool)
