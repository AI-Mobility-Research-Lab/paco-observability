"""Three-dimensional geometry primitives used by PACO occlusion models.

The module deliberately models *upright* oriented bounding boxes (OBBs): yaw
may be arbitrary, while roll and pitch are assumed to be zero.  That matches
the trajectory schema used by PACO and keeps the reference intersection test
analytic, deterministic, and dependency-free apart from NumPy.

Distances returned by the ray routines are in world-coordinate units.  Input
ray directions need not be normalized; normalization is performed internally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Hashable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ZMode: TypeAlias = Literal["raw", "ground_anchored"]

_EPS = 1.0e-12
_MISSING = object()


def _vector3(value: ArrayLike, *, name: str) -> FloatArray:
    """Return *value* as a finite ``float64`` vector of length three."""

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def wrap_angle(angle: float | ArrayLike) -> float | FloatArray:
    """Wrap radians to the half-open interval ``[-pi, pi)``.

    The function accepts either a scalar or an array.  Using a half-open
    interval makes the representation at the ``+/-pi`` seam deterministic and
    avoids special-case comparisons in field-of-view calculations.
    """

    values = np.asarray(angle, dtype=np.float64)
    wrapped = (values + math.pi) % (2.0 * math.pi) - math.pi
    if values.ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float64)


def angle_difference(angle: float, reference: float) -> float:
    """Return the signed shortest rotation from *reference* to *angle*."""

    return float(wrap_angle(float(angle) - float(reference)))


@dataclass(frozen=True, slots=True)
class GroundPlane:
    """A ground plane represented by ``normal dot point + d = 0``.

    Parameters are normalized at construction, so :meth:`signed_distance`
    returns metric distance when the input coordinates are metric.  A ground
    plane must have a nonzero vertical component because PACO anchors boxes by
    querying ``z`` at a supplied ``(x, y)`` location.
    """

    normal: Sequence[float] = (0.0, 0.0, 1.0)
    d: float = 0.0

    def __post_init__(self) -> None:
        normal = _vector3(self.normal, name="normal")
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= _EPS:
            raise ValueError("ground-plane normal must be nonzero")
        normal /= magnitude
        d = float(self.d) / magnitude
        if not math.isfinite(d):
            raise ValueError("ground-plane offset d must be finite")
        if abs(float(normal[2])) <= _EPS:
            raise ValueError("ground-plane normal must have a nonzero z component")
        object.__setattr__(self, "normal", tuple(float(v) for v in normal))
        object.__setattr__(self, "d", d)

    @classmethod
    def from_coefficients(cls, coefficients: Sequence[float]) -> GroundPlane:
        """Construct a plane from ``(a, b, c, d)`` coefficients."""

        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (4,):
            raise ValueError("ground-plane coefficients must have length four")
        return cls(normal=values[:3], d=float(values[3]))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GroundPlane:
        """Construct from a calibration mapping.

        Both ``{"normal": [a, b, c], "d": d}`` and
        ``{"a": a, "b": b, "c": c, "d": d}`` are accepted.
        """

        if "normal" in values:
            return cls(normal=values["normal"], d=float(values.get("d", 0.0)))
        try:
            coefficients = [values[name] for name in ("a", "b", "c", "d")]
        except KeyError as exc:
            raise KeyError(
                "ground-plane mapping needs either 'normal' and 'd', or a/b/c/d"
            ) from exc
        return cls.from_coefficients(coefficients)

    def height_at(self, x: float | ArrayLike, y: float | ArrayLike) -> float | FloatArray:
        """Return the world ``z`` coordinate of the plane at ``(x, y)``."""

        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        a, b, c = self.normal
        z_values = -(a * x_values + b * y_values + self.d) / c
        if z_values.ndim == 0:
            return float(z_values)
        return np.asarray(z_values, dtype=np.float64)

    # ``z_at`` is concise and useful in geometry-heavy call sites.
    z_at = height_at

    def signed_distance(self, points: ArrayLike) -> float | FloatArray:
        """Return signed perpendicular distance for one point or ``(..., 3)`` points."""

        array = np.asarray(points, dtype=np.float64)
        if array.shape == (3,):
            return float(np.dot(array, np.asarray(self.normal)) + self.d)
        if array.ndim < 2 or array.shape[-1] != 3:
            raise ValueError("points must have shape (3,) or (..., 3)")
        return np.asarray(array @ np.asarray(self.normal) + self.d, dtype=np.float64)


def _row_value(
    row: Any,
    candidates: Sequence[str],
    *,
    default: Any = _MISSING,
) -> Any:
    """Read the first available column/attribute from a dataframe-like row."""

    for name in candidates:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        try:
            return row[name]
        except (KeyError, IndexError, TypeError):
            pass
        if hasattr(row, name):
            return getattr(row, name)
    if default is not _MISSING:
        return default
    joined = ", ".join(repr(name) for name in candidates)
    raise KeyError(f"row does not provide any of the required fields: {joined}")


@dataclass(frozen=True, slots=True)
class OrientedBox:
    """An upright 3D oriented bounding box.

    ``center`` and ``size`` follow the PACO world-axis convention ``(x, y, z)``.
    ``yaw`` rotates the local x-axis counter-clockwise about world z.  ``size``
    components must be strictly positive.  ``object_id`` and ``class_name`` are
    optional metadata and do not affect geometry.  ``label`` is accepted as an
    alias for callers using a generic box schema.
    """

    center: Sequence[float]
    size: Sequence[float]
    yaw: float = 0.0
    object_id: Hashable | None = None
    class_name: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        center = _vector3(self.center, name="center")
        size = _vector3(self.size, name="size")
        if np.any(size <= 0.0):
            raise ValueError("all oriented-box dimensions must be positive")
        yaw = float(self.yaw)
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite")
        class_name = None if self.class_name is None else str(self.class_name)
        label = None if self.label is None else str(self.label)
        if class_name is not None and label is not None and class_name != label:
            raise ValueError("class_name and label must agree when both are supplied")
        resolved_label = class_name if class_name is not None else label
        object.__setattr__(self, "center", tuple(float(v) for v in center))
        object.__setattr__(self, "size", tuple(float(v) for v in size))
        object.__setattr__(self, "yaw", float(wrap_angle(yaw)))
        object.__setattr__(self, "class_name", resolved_label)
        object.__setattr__(self, "label", resolved_label)

    @classmethod
    def from_row(
        cls,
        row: Any,
        *,
        z_mode: ZMode | str = "raw",
        ground_plane: GroundPlane | None = None,
        ground_clearance_m: float = 0.0,
        columns: Mapping[str, str] | None = None,
    ) -> OrientedBox:
        """Build an OBB from a pandas Series, namedtuple, or mapping.

        The default schema recognizes PACO's ``bbox_center_*``, ``bbox_d*`` and
        ``bbox_yaw`` columns, with shorter aliases such as ``x``, ``dx`` and
        ``yaw`` as fallbacks.  ``columns`` can map the semantic keys ``x``,
        ``y``, ``z``, ``dx``, ``dy``, ``dz``, ``yaw``, ``id`` and ``label`` to
        custom column names.

        ``z_mode='raw'`` preserves the row's center z coordinate.
        ``z_mode='ground_anchored'`` ignores that value and places the box
        bottom on ``ground_plane`` (plus ``ground_clearance_m``).  If no plane
        is supplied for ground anchoring, the horizontal plane ``z=0`` is used.
        """

        custom = dict(columns or {})

        def names(key: str, defaults: Sequence[str]) -> tuple[str, ...]:
            return ((custom[key],) if key in custom else ()) + tuple(defaults)

        x = float(_row_value(row, names("x", ("bbox_center_x", "center_x", "x"))))
        y = float(_row_value(row, names("y", ("bbox_center_y", "center_y", "y"))))
        dx = float(_row_value(row, names("dx", ("bbox_dx", "dx", "length"))))
        dy = float(_row_value(row, names("dy", ("bbox_dy", "dy", "width"))))
        dz = float(_row_value(row, names("dz", ("bbox_dz", "dz", "height"))))
        yaw = float(
            _row_value(row, names("yaw", ("bbox_yaw", "yaw", "heading")), default=0.0)
        )

        normalized_mode = str(z_mode).strip().lower().replace("-", "_")
        if normalized_mode in {"ground", "anchored", "grounded"}:
            normalized_mode = "ground_anchored"
        if normalized_mode == "raw":
            z = float(
                _row_value(row, names("z", ("bbox_center_z", "center_z", "z")))
            )
        elif normalized_mode == "ground_anchored":
            plane = ground_plane if ground_plane is not None else GroundPlane()
            clearance = float(ground_clearance_m)
            if not math.isfinite(clearance):
                raise ValueError("ground_clearance_m must be finite")
            z = float(plane.height_at(x, y)) + clearance + dz / 2.0
        else:
            raise ValueError("z_mode must be 'raw' or 'ground_anchored'")

        object_id = _row_value(
            row,
            names("id", ("track_id", "object_id", "id")),
            default=None,
        )
        label_value = _row_value(
            row,
            names("label", ("class_name", "label", "class")),
            default=None,
        )
        label = None if label_value is None else str(label_value)
        return cls(
            center=(x, y, z),
            size=(dx, dy, dz),
            yaw=yaw,
            object_id=object_id,
            class_name=label,
        )

    @property
    def center_array(self) -> FloatArray:
        """Center as a new ``float64`` NumPy array."""

        return np.asarray(self.center, dtype=np.float64)

    @property
    def size_array(self) -> FloatArray:
        """Dimensions as a new ``float64`` NumPy array."""

        return np.asarray(self.size, dtype=np.float64)

    @property
    def half_size(self) -> FloatArray:
        """Half-dimensions in local box coordinates."""

        return self.size_array / 2.0

    @property
    def top_center(self) -> FloatArray:
        """World coordinate of the center of the top face."""

        point = self.center_array
        point[2] += self.size[2] / 2.0
        return point

    @property
    def bottom_center(self) -> FloatArray:
        """World coordinate of the center of the bottom face."""

        point = self.center_array
        point[2] -= self.size[2] / 2.0
        return point

    def world_to_local(self, points: ArrayLike) -> FloatArray:
        """Transform one point or ``(..., 3)`` points into box-local coordinates."""

        values = np.asarray(points, dtype=np.float64)
        if values.shape == (3,):
            shifted = values - self.center_array
        elif values.ndim >= 2 and values.shape[-1] == 3:
            shifted = values - self.center_array
        else:
            raise ValueError("points must have shape (3,) or (..., 3)")
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        local = np.empty_like(shifted, dtype=np.float64)
        local[..., 0] = c * shifted[..., 0] + s * shifted[..., 1]
        local[..., 1] = -s * shifted[..., 0] + c * shifted[..., 1]
        local[..., 2] = shifted[..., 2]
        return local

    def direction_to_local(self, directions: ArrayLike) -> FloatArray:
        """Rotate one direction or ``(..., 3)`` directions into box-local axes."""

        values = np.asarray(directions, dtype=np.float64)
        if values.shape != (3,) and (values.ndim < 2 or values.shape[-1] != 3):
            raise ValueError("directions must have shape (3,) or (..., 3)")
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        local = np.empty_like(values, dtype=np.float64)
        local[..., 0] = c * values[..., 0] + s * values[..., 1]
        local[..., 1] = -s * values[..., 0] + c * values[..., 1]
        local[..., 2] = values[..., 2]
        return local

    def local_to_world(self, points: ArrayLike) -> FloatArray:
        """Transform one point or ``(..., 3)`` local points to world coordinates."""

        values = np.asarray(points, dtype=np.float64)
        if values.shape != (3,) and (values.ndim < 2 or values.shape[-1] != 3):
            raise ValueError("points must have shape (3,) or (..., 3)")
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        world = np.empty_like(values, dtype=np.float64)
        world[..., 0] = c * values[..., 0] - s * values[..., 1]
        world[..., 1] = s * values[..., 0] + c * values[..., 1]
        world[..., 2] = values[..., 2]
        return world + self.center_array

    def corners(self) -> FloatArray:
        """Return the eight box corners as an array with shape ``(8, 3)``."""

        hx, hy, hz = self.half_size
        local = np.asarray(
            [
                (sx * hx, sy * hy, sz * hz)
                for sz in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sx in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        return self.local_to_world(local)

    def contains(self, point: ArrayLike, *, atol: float = 1.0e-9) -> bool:
        """Return whether a world point lies inside or on the OBB."""

        local = self.world_to_local(_vector3(point, name="point"))
        return bool(np.all(np.abs(local) <= self.half_size + float(atol)))


@dataclass(frozen=True, slots=True)
class RayHit:
    """Intersection interval between a forward ray and an OBB.

    ``distance`` is the first nonnegative contact.  It is zero when the origin
    starts inside or on the box.  ``exit_distance`` is the far slab contact.
    Both are metric because :func:`ray_obb_intersection` normalizes directions.
    """

    distance: float
    exit_distance: float
    point: tuple[float, float, float]
    exit_point: tuple[float, float, float]
    started_inside: bool


def ray_obb_intersection(
    origin: ArrayLike,
    direction: ArrayLike,
    box: OrientedBox,
    *,
    max_distance: float = math.inf,
    epsilon: float = 1.0e-9,
) -> RayHit | None:
    """Intersect a forward ray with an upright OBB using the slab algorithm.

    Parallel rays are handled without division by zero.  A zero-length
    direction represents a degenerate point query: it returns a zero-distance
    hit only when the origin lies in the box.  When the origin starts inside,
    the first-contact distance is zero and ``exit_distance`` records where a
    nondegenerate ray leaves the box.

    Parameters
    ----------
    origin, direction:
        Three-dimensional world vectors.  ``direction`` need not be unit length.
    box:
        Upright oriented bounding box to test.
    max_distance:
        Optional finite ray length.  A contact beyond it is ignored.
    epsilon:
        Numerical tolerance for parallel axes and slab-boundary comparisons.
    """

    ray_origin = _vector3(origin, name="origin")
    ray_direction = _vector3(direction, name="direction")
    limit = float(max_distance)
    tolerance = float(epsilon)
    if math.isnan(limit) or limit < 0.0:
        raise ValueError("max_distance must be nonnegative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    direction_norm = float(np.linalg.norm(ray_direction))
    if direction_norm <= tolerance:
        if not box.contains(ray_origin, atol=tolerance):
            return None
        point = tuple(float(v) for v in ray_origin)
        return RayHit(0.0, 0.0, point, point, True)
    unit_direction = ray_direction / direction_norm

    local_origin = box.world_to_local(ray_origin)
    local_direction = box.direction_to_local(unit_direction)
    half_size = box.half_size
    started_inside = bool(np.all(np.abs(local_origin) <= half_size + tolerance))

    t_enter = -math.inf
    t_exit = math.inf
    for coordinate, component, half_extent in zip(
        local_origin, local_direction, half_size, strict=True
    ):
        if abs(float(component)) <= tolerance:
            if coordinate < -half_extent - tolerance or coordinate > half_extent + tolerance:
                return None
            continue
        first = (-half_extent - coordinate) / component
        second = (half_extent - coordinate) / component
        slab_enter = float(min(first, second))
        slab_exit = float(max(first, second))
        t_enter = max(t_enter, slab_enter)
        t_exit = min(t_exit, slab_exit)
        if t_enter > t_exit + tolerance:
            return None

    if t_exit < -tolerance:
        return None
    first_contact = 0.0 if started_inside else max(0.0, t_enter)
    if first_contact > limit + tolerance:
        return None
    far_contact = max(first_contact, t_exit)
    point = ray_origin + first_contact * unit_direction
    exit_point = ray_origin + far_contact * unit_direction
    return RayHit(
        distance=float(first_contact),
        exit_distance=float(far_contact),
        point=tuple(float(v) for v in point),
        exit_point=tuple(float(v) for v in exit_point),
        started_inside=started_inside,
    )


def ray_obb_intersection_distance(
    origin: ArrayLike,
    direction: ArrayLike,
    box: OrientedBox,
    *,
    max_distance: float = math.inf,
    epsilon: float = 1.0e-9,
) -> float | None:
    """Return only the first ray/OBB contact distance, or ``None`` on a miss."""

    hit = ray_obb_intersection(
        origin,
        direction,
        box,
        max_distance=max_distance,
        epsilon=epsilon,
    )
    return None if hit is None else hit.distance


def segment_obb_intersection(
    start: ArrayLike,
    end: ArrayLike,
    box: OrientedBox,
    *,
    epsilon: float = 1.0e-9,
) -> RayHit | None:
    """Intersect the closed line segment from ``start`` to ``end`` with an OBB."""

    segment_start = _vector3(start, name="start")
    segment_end = _vector3(end, name="end")
    delta = segment_end - segment_start
    length = float(np.linalg.norm(delta))
    return ray_obb_intersection(
        segment_start,
        delta,
        box,
        max_distance=length,
        epsilon=epsilon,
    )
