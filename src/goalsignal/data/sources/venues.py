"""Venue/travel geometry (contract + deterministic helpers).

Pure, deterministic geographic helpers usable now. Venue *reference data*
(coordinates, altitude per venue) is user-provided and ingested in Milestone B;
historical altitude exposure is estimated from prior matches in Milestone D —
never hard-coded as a team narrative.
"""

from __future__ import annotations

import math

from goalsignal.data.sources.base import MilestoneNotImplementedError

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in km between two (lat, lon) points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def altitude_change_m(from_altitude_m: float, to_altitude_m: float) -> float:
    """Signed altitude change (positive = ascending to the venue)."""
    return to_altitude_m - from_altitude_m


class VenueReference:
    name = "venues"
    role = "venue_reference"

    def load(self):
        raise MilestoneNotImplementedError(
            "Venue reference ingestion (coordinates, altitude) lands in Milestone B."
        )
