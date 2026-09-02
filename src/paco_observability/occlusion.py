"""Ray-based 3D visibility estimators for PACO.

Two related estimators are provided:

``projected_visible_fraction``
    Samples the target's angular projection and uses exact ray--OBB
    intersections for every sampled ray.  This is the higher-resolution,
    within-OBB-model reference used to validate cheaper approximations.

``center_top_visibility``
    Casts only a center ray and a top-center ray.  It is inexpensive enough for
    full-record sensitivity sweeps while retaining the main vertical effect.

The word ``exact`` refers to each analytic ray--OBB intersection, not to a
physical reconstruction of unobserved scene surfaces.  The projected visible
fraction remains resolution-dependent and only models the supplied dynamic
boxes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Hashable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import OrientedBox, angle_difference, ray_obb_intersection, wrap_angle


FloatArray: TypeAlias = NDArray[np.float64]
VisibilityCriterion: TypeAlias = Literal["any", "all"]

_ANGLE_EPS = 1.0e-10


def _point3(value: ArrayLike, *, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values")
    return point


def _validate_count(value: int, *, name: str) -> int:
    count = int(value)
    if count != value or count < 1:
        raise ValueError(f"{name} must be a positive integer")
    return count


@dataclass(frozen=True, slots=True)
class CoverageFlags:
    """Independent center-point coverage flags for a target OBB.

    Keeping ``in_fov`` and ``in_range`` separate avoids embedding an arbitrary
    precedence when a target fails both tests.  ``range_m`` is horizontal BEV
    center range, consistent with PACO's onboard sector model.
    """

    range_m: float
    bearing_rad: float
    relative_bearing_rad: float
    in_fov: bool
    in_range: bool

    @property
    def covered(self) -> bool:
        """Whether the target center passes both angular and range gates."""

        return self.in_fov and self.in_range

    @property
    def bearing_deg(self) -> float:
        """Absolute horizontal bearing in degrees."""

        return math.degrees(self.bearing_rad)

    @property
    def relative_bearing_deg(self) -> float:
        """Heading-relative horizontal bearing in degrees."""

        return math.degrees(self.relative_bearing_rad)


def center_coverage_flags(
    sensor_origin: ArrayLike,
    target: OrientedBox,
    *,
    heading_rad: float = 0.0,
    fov_rad: float = 2.0 * math.pi,
    near_range_m: float = 0.0,
    far_range_m: float = math.inf,
    epsilon: float = 1.0e-9,
) -> CoverageFlags:
    """Evaluate center bearing, horizontal FOV, and BEV range independently.

    Range follows the existing PACO convention ``near < range <= far``.  FOV
    boundaries are inclusive.  A FOV of at least ``2*pi`` covers the full
    circle, including targets whose bearing lies at the ``+/-pi`` wrap seam.
    If the target center has zero horizontal displacement, its relative
    bearing is defined as zero.
    """

    origin = _point3(sensor_origin, name="sensor_origin")
    heading = float(heading_rad)
    fov = float(fov_rad)
    near = float(near_range_m)
    far = float(far_range_m)
    tolerance = float(epsilon)
    if not math.isfinite(heading):
        raise ValueError("heading_rad must be finite")
    if not math.isfinite(fov) or fov <= 0.0:
        raise ValueError("fov_rad must be positive and finite")
    if not math.isfinite(near) or near < 0.0:
        raise ValueError("near_range_m must be nonnegative and finite")
    if math.isnan(far) or far < near:
        raise ValueError("far_range_m must be at least near_range_m")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    displacement = target.center_array - origin
    horizontal_range = float(math.hypot(displacement[0], displacement[1]))
    if horizontal_range <= tolerance:
        bearing = float(wrap_angle(heading))
        relative = 0.0
    else:
        bearing = math.atan2(float(displacement[1]), float(displacement[0]))
        relative = angle_difference(bearing, heading)

    if fov >= 2.0 * math.pi - tolerance:
        in_fov = True
    else:
        in_fov = abs(relative) <= fov / 2.0 + tolerance
    in_range = horizontal_range > near + tolerance and horizontal_range <= far + tolerance
    return CoverageFlags(
        range_m=horizontal_range,
        bearing_rad=float(wrap_angle(bearing)),
        relative_bearing_rad=relative,
        in_fov=bool(in_fov),
        in_range=bool(in_range),
    )


# More explicit alias for callers that mirror the manuscript terminology.
center_fov_range_flags = center_coverage_flags


@dataclass(frozen=True, slots=True)
class AngularVisibilityResult:
    """Result of angular-grid ray casting against one target.

    ``hit_rays`` is the number of sampled directions that analytically hit the
    target OBB; these rays form the denominator.  ``visible_rays`` is the subset
    whose first contact is not a supplied blocker.  ``score`` is therefore
    ``visible_rays / hit_rays`` (zero when no sampled ray hits the target).

    ``nearest_blocker`` is its ``object_id`` when present, otherwise its index
    in the blocker iterable.  It is the blocker with the smallest entry
    distance over blocked target rays, not necessarily the most frequent one.
    """

    score: float
    hit_rays: int
    visible_rays: int
    nearest_blocker: Hashable | None
    nearest_blocker_distance_m: float | None
    sampled_rays: int

    @property
    def blocked_rays(self) -> int:
        """Number of target-hitting rays intercepted by blockers."""

        return self.hit_rays - self.visible_rays

    @property
    def visible_fraction(self) -> float:
        """Alias for :attr:`score`."""

        return self.score


@dataclass(frozen=True, slots=True)
class MultiRayVisibilityResult:
    """Low-cost visibility result for a named collection of target rays."""

    visible: bool
    score: float
    hit_rays: int
    visible_rays: int
    nearest_blocker: Hashable | None
    nearest_blocker_distance_m: float | None
    ray_labels: tuple[str, ...]
    visible_labels: tuple[str, ...]
    blocked_labels: tuple[str, ...]

    @property
    def blocked_rays(self) -> int:
        """Number of target-hitting rays intercepted by blockers."""

        return self.hit_rays - self.visible_rays


def _blocker_identity(blocker: OrientedBox, index: int) -> Hashable:
    return index if blocker.object_id is None else blocker.object_id


def _same_object(first: OrientedBox, second: OrientedBox) -> bool:
    if first is second:
        return True
    return (
        first.object_id is not None
        and second.object_id is not None
        and first.object_id == second.object_id
    )


def _first_blocker(
    origin: FloatArray,
    direction: FloatArray,
    target_distance: float,
    target: OrientedBox,
    blockers: Sequence[OrientedBox],
    *,
    clearance_m: float,
    epsilon: float,
) -> tuple[Hashable | None, float | None]:
    """Return the nearest blocker strictly before a target contact."""

    nearest_identity: Hashable | None = None
    nearest_distance = math.inf
    # The max distance trims obvious behind-target intersections while the
    # strict clearance comparison handles touching/overlapping surfaces.
    max_blocker_distance = max(0.0, target_distance - clearance_m + epsilon)
    for index, blocker in enumerate(blockers):
        if _same_object(target, blocker):
            continue
        hit = ray_obb_intersection(
            origin,
            direction,
            blocker,
            max_distance=max_blocker_distance,
            epsilon=epsilon,
        )
        if hit is None:
            continue
        if hit.distance + clearance_m < target_distance - epsilon:
            if hit.distance < nearest_distance:
                nearest_distance = hit.distance
                nearest_identity = _blocker_identity(blocker, index)
    if nearest_identity is None:
        return None, None
    return nearest_identity, float(nearest_distance)


def _sample_interval(
    low: float,
    high: float,
    count: int,
    *,
    preferred: float,
) -> FloatArray:
    """Sample a closed interval while guaranteeing a preferred center ray."""

    if count == 1 or high - low <= _ANGLE_EPS:
        return np.asarray([min(max(preferred, low), high)], dtype=np.float64)
    values = np.linspace(low, high, count, dtype=np.float64)
    center = min(max(preferred, low), high)
    nearest = int(np.argmin(np.abs(values - center)))
    values[nearest] = center
    values.sort()
    return values


def _angular_grid(
    origin: FloatArray,
    target: OrientedBox,
    *,
    azimuth_samples: int,
    elevation_samples: int,
) -> list[FloatArray]:
    """Return directions covering the rectangular angular bound of a target."""

    corners = target.corners()
    relative_corners = corners - origin
    relative_center = target.center_array - origin
    center_horizontal = float(math.hypot(relative_center[0], relative_center[1]))
    if center_horizontal <= _ANGLE_EPS:
        center_azimuth = 0.0
    else:
        center_azimuth = math.atan2(relative_center[1], relative_center[0])
    center_elevation = math.atan2(relative_center[2], center_horizontal)

    corner_horizontal = np.hypot(relative_corners[:, 0], relative_corners[:, 1])
    corner_azimuth = np.arctan2(relative_corners[:, 1], relative_corners[:, 0])
    corner_elevation = np.arctan2(relative_corners[:, 2], corner_horizontal)

    local_origin = target.world_to_local(origin)
    half_size = target.half_size
    inside_horizontal_projection = bool(
        abs(local_origin[0]) <= half_size[0] + _ANGLE_EPS
        and abs(local_origin[1]) <= half_size[1] + _ANGLE_EPS
    )
    if inside_horizontal_projection:
        offsets = -math.pi + (
            np.arange(azimuth_samples, dtype=np.float64) + 0.5
        ) * (2.0 * math.pi / azimuth_samples)
        azimuths = np.asarray(wrap_angle(center_azimuth + offsets), dtype=np.float64)
    else:
        offsets = np.asarray(
            [angle_difference(float(value), center_azimuth) for value in corner_azimuth],
            dtype=np.float64,
        )
        azimuth_offsets = _sample_interval(
            float(np.min(offsets)),
            float(np.max(offsets)),
            azimuth_samples,
            preferred=0.0,
        )
        azimuths = np.asarray(wrap_angle(center_azimuth + azimuth_offsets), dtype=np.float64)

    elevations = _sample_interval(
        float(np.min(corner_elevation)),
        float(np.max(corner_elevation)),
        elevation_samples,
        preferred=center_elevation,
    )
    directions: list[FloatArray] = []
    for elevation in elevations:
        cos_elevation = math.cos(float(elevation))
        for azimuth in azimuths:
            directions.append(
                np.asarray(
                    (
                        cos_elevation * math.cos(float(azimuth)),
                        cos_elevation * math.sin(float(azimuth)),
                        math.sin(float(elevation)),
                    ),
                    dtype=np.float64,
                )
            )
    return directions


def projected_visible_fraction(
    sensor_origin: ArrayLike,
    target: OrientedBox,
    blockers: Iterable[OrientedBox] = (),
    *,
    azimuth_samples: int = 9,
    elevation_samples: int = 5,
    clearance_m: float = 0.0,
    epsilon: float = 1.0e-9,
) -> AngularVisibilityResult:
    """Estimate target visible fraction on a projected angular grid.

    A rectangular grid spans the target OBB's azimuth/elevation bounds.  Rays
    missing the exact target OBB are discarded, preventing empty corners of the
    angular bounding rectangle from entering the denominator.  Every remaining
    ray is visible unless another supplied OBB has a strictly earlier contact.

    Increase ``azimuth_samples`` and ``elevation_samples`` for convergence
    analysis.  Odd counts are recommended because they retain an explicit
    center ray.  ``clearance_m`` requires that a blocker be at least that much
    nearer than the target; zero gives the analytic OBB ordering apart from the
    numerical ``epsilon``.
    """

    origin = _point3(sensor_origin, name="sensor_origin")
    n_azimuth = _validate_count(azimuth_samples, name="azimuth_samples")
    n_elevation = _validate_count(elevation_samples, name="elevation_samples")
    clearance = float(clearance_m)
    tolerance = float(epsilon)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance_m must be nonnegative and finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    blocker_list = tuple(blockers)
    if not all(isinstance(blocker, OrientedBox) for blocker in blocker_list):
        raise TypeError("all blockers must be OrientedBox instances")

    directions = _angular_grid(
        origin,
        target,
        azimuth_samples=n_azimuth,
        elevation_samples=n_elevation,
    )
    hit_rays = 0
    visible_rays = 0
    nearest_blocker: Hashable | None = None
    nearest_distance = math.inf
    for direction in directions:
        target_hit = ray_obb_intersection(origin, direction, target, epsilon=tolerance)
        if target_hit is None:
            continue
        hit_rays += 1
        blocker_id, blocker_distance = _first_blocker(
            origin,
            direction,
            target_hit.distance,
            target,
            blocker_list,
            clearance_m=clearance,
            epsilon=tolerance,
        )
        if blocker_id is None:
            visible_rays += 1
        elif blocker_distance is not None and blocker_distance < nearest_distance:
            nearest_blocker = blocker_id
            nearest_distance = blocker_distance

    score = visible_rays / hit_rays if hit_rays else 0.0
    return AngularVisibilityResult(
        score=float(score),
        hit_rays=hit_rays,
        visible_rays=visible_rays,
        nearest_blocker=nearest_blocker,
        nearest_blocker_distance_m=(
            None if nearest_blocker is None else float(nearest_distance)
        ),
        sampled_rays=len(directions),
    )


# Alias emphasizing that this is an angular-grid estimator with exact ray tests.
angular_grid_visible_fraction = projected_visible_fraction


def multi_ray_visibility(
    sensor_origin: ArrayLike,
    target: OrientedBox,
    blockers: Iterable[OrientedBox] = (),
    *,
    target_points: Mapping[str, ArrayLike] | None = None,
    criterion: VisibilityCriterion = "any",
    clearance_m: float = 0.0,
    epsilon: float = 1.0e-9,
) -> MultiRayVisibilityResult:
    """Classify a target from a small named set of rays.

    By default the rays point to the target center and top-face center.  Under
    ``criterion='any'`` the target is visible when either ray is unblocked,
    approximating partial detectability.  ``criterion='all'`` requires every
    target-hitting ray to be clear.  Custom world-coordinate sample points may
    be supplied as an insertion-ordered mapping.
    """

    origin = _point3(sensor_origin, name="sensor_origin")
    if criterion not in {"any", "all"}:
        raise ValueError("criterion must be 'any' or 'all'")
    clearance = float(clearance_m)
    tolerance = float(epsilon)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance_m must be nonnegative and finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    blocker_list = tuple(blockers)
    if not all(isinstance(blocker, OrientedBox) for blocker in blocker_list):
        raise TypeError("all blockers must be OrientedBox instances")

    if target_points is None:
        points: Mapping[str, ArrayLike] = {
            "center": target.center_array,
            "top": target.top_center,
        }
    else:
        if not target_points:
            raise ValueError("target_points must contain at least one point")
        points = target_points

    ray_labels: list[str] = []
    visible_labels: list[str] = []
    blocked_labels: list[str] = []
    nearest_blocker: Hashable | None = None
    nearest_distance = math.inf
    for raw_label, raw_point in points.items():
        label = str(raw_label)
        point = _point3(raw_point, name=f"target_points[{label!r}]")
        direction = point - origin
        target_hit = ray_obb_intersection(origin, direction, target, epsilon=tolerance)
        if target_hit is None:
            # A custom sample point may not lie on/in the target.  Such a ray
            # is not part of the score denominator.
            continue
        ray_labels.append(label)
        blocker_id, blocker_distance = _first_blocker(
            origin,
            direction,
            target_hit.distance,
            target,
            blocker_list,
            clearance_m=clearance,
            epsilon=tolerance,
        )
        if blocker_id is None:
            visible_labels.append(label)
        else:
            blocked_labels.append(label)
            if blocker_distance is not None and blocker_distance < nearest_distance:
                nearest_blocker = blocker_id
                nearest_distance = blocker_distance

    hit_rays = len(ray_labels)
    visible_rays = len(visible_labels)
    if criterion == "any":
        visible = visible_rays > 0
    else:
        visible = hit_rays > 0 and visible_rays == hit_rays
    score = visible_rays / hit_rays if hit_rays else 0.0
    return MultiRayVisibilityResult(
        visible=visible,
        score=float(score),
        hit_rays=hit_rays,
        visible_rays=visible_rays,
        nearest_blocker=nearest_blocker,
        nearest_blocker_distance_m=(
            None if nearest_blocker is None else float(nearest_distance)
        ),
        ray_labels=tuple(ray_labels),
        visible_labels=tuple(visible_labels),
        blocked_labels=tuple(blocked_labels),
    )


def center_top_visibility(
    sensor_origin: ArrayLike,
    target: OrientedBox,
    blockers: Iterable[OrientedBox] = (),
    *,
    criterion: VisibilityCriterion = "any",
    clearance_m: float = 0.0,
    epsilon: float = 1.0e-9,
) -> MultiRayVisibilityResult:
    """Run the default two-ray center/top 3D visibility approximation."""

    return multi_ray_visibility(
        sensor_origin,
        target,
        blockers,
        criterion=criterion,
        clearance_m=clearance_m,
        epsilon=epsilon,
    )


# Short alias used in validation scripts.
center_top_visible = center_top_visibility

