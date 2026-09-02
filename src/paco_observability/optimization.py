"""Decision helpers for weighted coverage and hotspot-rank stability.

The maximum-coverage routine treats rows as candidate deployments and columns
as demand elements.  Greedy selection is deterministic: when candidates have
the same marginal weighted gain, the lower original row index wins.

The rank-stability helpers operate on cluster-level hotspot contributions.  A
cluster can be a physical conflict pair, an event, or a temporal block.  This
keeps repeated frames and viewpoints together when quantifying how often a
hotspot remains highly ranked under resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GreedyCoverageStep:
    """One candidate addition in a greedy maximum-coverage trajectory."""

    step: int
    candidate_index: int
    candidate_id: Hashable
    marginal_gain: float
    cumulative_covered_weight: float
    coverage_fraction: float
    newly_covered_count: int
    newly_covered_indices: tuple[int, ...]


@dataclass(frozen=True)
class MaximumCoverageResult:
    """Result of deterministic greedy weighted maximum coverage.

    Attributes:
        selected_indices: Candidate row indices in selection order.
        selected_ids: Caller-provided candidate identifiers in selection order.
        marginal_gains: Weighted gain at each selection step.
        cumulative_gains: Cumulative covered weight after each selection.
        coverage_fraction: Final covered weight divided by total demand weight.
        covered_weight: Final weighted demand coverage.
        total_weight: Total demand weight in the optimization instance.
        covered_mask: Boolean mask over demand-element columns.
        trajectory: Full per-step audit trail.
    """

    selected_indices: tuple[int, ...]
    selected_ids: tuple[Hashable, ...]
    marginal_gains: tuple[float, ...]
    cumulative_gains: tuple[float, ...]
    coverage_fraction: float
    covered_weight: float
    total_weight: float
    covered_mask: BoolArray
    trajectory: tuple[GreedyCoverageStep, ...]

    @property
    def selection_trajectory(self) -> tuple[GreedyCoverageStep, ...]:
        """Alias for :attr:`trajectory`."""

        return self.trajectory


@dataclass(frozen=True)
class HotspotRankStability:
    """Bootstrap score and rank summaries for a collection of hotspots.

    Ranks are one-based. Undefined bootstrap scores, which can occur when a
    resample has zero denominator for a sparse hotspot, are assigned the worst
    rank and counted as failures to enter the top-k set. ``valid_fraction``
    makes the frequency of this condition explicit.
    """

    hotspot_ids: tuple[Hashable, ...]
    point_scores: FloatArray
    point_order: tuple[int, ...]
    score_ci_low: FloatArray
    score_ci_high: FloatArray
    median_rank: FloatArray
    rank_ci_low: FloatArray
    rank_ci_high: FloatArray
    top_k_probability: FloatArray
    valid_fraction: FloatArray
    bootstrap_scores: FloatArray
    bootstrap_ranks: IntArray
    top_k: int
    confidence_level: float
    higher_is_hotter: bool
    seed: int | None = None

    @property
    def ordered_hotspot_ids(self) -> tuple[Hashable, ...]:
        """Return hotspot identifiers in deterministic point-estimate order."""

        return tuple(self.hotspot_ids[index] for index in self.point_order)

    def records(self) -> list[dict[str, Any]]:
        """Return one serializable summary dictionary per hotspot."""

        return [
            {
                "hotspot_id": self.hotspot_ids[index],
                "point_score": float(self.point_scores[index]),
                "score_ci_low": float(self.score_ci_low[index]),
                "score_ci_high": float(self.score_ci_high[index]),
                "median_rank": float(self.median_rank[index]),
                "rank_ci_low": float(self.rank_ci_low[index]),
                "rank_ci_high": float(self.rank_ci_high[index]),
                "top_k_probability": float(self.top_k_probability[index]),
                "valid_fraction": float(self.valid_fraction[index]),
            }
            for index in range(len(self.hotspot_ids))
        ]


def _as_coverage_matrix(coverage: ArrayLike) -> BoolArray:
    """Validate and convert a candidate-by-element binary matrix."""

    array = np.asarray(coverage)
    if array.ndim != 2:
        raise ValueError(f"coverage must be two-dimensional; got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("coverage must contain at least one candidate row")
    if array.shape[1] == 0:
        raise ValueError("coverage must contain at least one demand-element column")
    if array.dtype.kind not in "bifuc":
        raise ValueError("coverage must be a boolean or numeric 0/1 matrix")
    if array.dtype.kind != "b":
        if not np.all(np.isfinite(array)):
            raise ValueError("coverage must contain only finite values")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("numeric coverage entries must be exactly 0 or 1")
    return array.astype(bool, copy=False)


def _as_weights(weights: ArrayLike | None, n_elements: int) -> FloatArray:
    """Validate nonnegative element weights with a positive total."""

    if weights is None:
        array = np.ones(n_elements, dtype=np.float64)
    else:
        try:
            array = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "weights must be convertible to a one-dimensional float array"
            ) from exc
        if array.ndim != 1:
            raise ValueError(f"weights must be one-dimensional; got shape {array.shape}")
        if len(array) != n_elements:
            raise ValueError(f"weights has length {len(array)}; expected {n_elements}")
        if not np.all(np.isfinite(array)):
            raise ValueError("weights must contain only finite values")
        if np.any(array < 0.0):
            raise ValueError("weights must contain only nonnegative values")
    if float(np.sum(array, dtype=np.float64)) <= 0.0:
        raise ValueError("weights must have a strictly positive total")
    return array


def _validate_ids(
    ids: Sequence[Hashable] | None,
    n_items: int,
    *,
    name: str,
) -> tuple[Hashable, ...]:
    """Validate unique hashable identifiers, or use integer indices."""

    if ids is None:
        return tuple(range(n_items))
    if isinstance(ids, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identifiers, not a string")
    result = tuple(ids)
    if len(result) != n_items:
        raise ValueError(f"{name} has length {len(result)}; expected {n_items}")
    try:
        unique_count = len(set(result))
    except TypeError as exc:
        raise ValueError(f"{name} entries must be hashable") from exc
    if unique_count != n_items:
        raise ValueError(f"{name} entries must be unique")
    return result


def _validate_positive_integer(name: str, value: int) -> int:
    """Return a validated integer of at least one."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def _validate_confidence_level(confidence_level: float) -> float:
    """Return a finite confidence level strictly between zero and one."""

    if isinstance(confidence_level, (bool, np.bool_)):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    try:
        value = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be strictly between 0 and 1") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return value


def _validate_seed(seed: int) -> int:
    """Return an integer random seed."""

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return int(seed)


def greedy_weighted_maximum_coverage(
    coverage: ArrayLike,
    weights: ArrayLike | None = None,
    budget: int = 1,
    *,
    candidate_ids: Sequence[Hashable] | None = None,
    stop_when_no_gain: bool = True,
) -> MaximumCoverageResult:
    """Select candidates by deterministic greedy weighted maximum coverage.

    Args:
        coverage: Binary matrix with shape ``(n_candidates, n_elements)``.
            ``coverage[i, j]`` indicates that candidate ``i`` covers demand
            element ``j``.
        weights: Nonnegative weight for each demand element. Equal weights are
            used when omitted.
        budget: Maximum number of candidates to select. Zero is allowed.
        candidate_ids: Optional unique identifiers for candidate rows.
        stop_when_no_gain: Stop before the budget is exhausted if every
            remaining candidate has zero marginal weighted gain. If false,
            exactly ``budget`` candidates are selected.

    Returns:
        A :class:`MaximumCoverageResult` with the selected candidates, marginal
        gains, final coverage fraction, covered mask, and complete trajectory.

    Notes:
        Ties are resolved by the lower original candidate row index. This rule
        is independent of hash order and makes repeated runs deterministic.
    """

    matrix = _as_coverage_matrix(coverage)
    element_weights = _as_weights(weights, matrix.shape[1])
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be an integer")
    normalized_budget = int(budget)
    if normalized_budget < 0 or normalized_budget > matrix.shape[0]:
        raise ValueError(f"budget must be between 0 and {matrix.shape[0]}; got {budget}")
    if not isinstance(stop_when_no_gain, (bool, np.bool_)):
        raise ValueError("stop_when_no_gain must be boolean")
    ids = _validate_ids(candidate_ids, matrix.shape[0], name="candidate_ids")

    total_weight = float(np.sum(element_weights, dtype=np.float64))
    covered = np.zeros(matrix.shape[1], dtype=bool)
    selected = np.zeros(matrix.shape[0], dtype=bool)
    selected_indices: list[int] = []
    selected_ids: list[Hashable] = []
    marginal_gains: list[float] = []
    cumulative_gains: list[float] = []
    trajectory: list[GreedyCoverageStep] = []
    covered_weight = 0.0

    for step_number in range(1, normalized_budget + 1):
        uncovered_weights = element_weights * (~covered)
        gains = matrix.astype(np.float64, copy=False) @ uncovered_weights
        gains[selected] = -np.inf
        candidate_index = int(np.argmax(gains))  # np.argmax returns the first tie.
        marginal_gain = float(gains[candidate_index])
        if bool(stop_when_no_gain) and marginal_gain <= 0.0:
            break

        newly_covered = matrix[candidate_index] & ~covered
        newly_covered_indices = tuple(int(index) for index in np.flatnonzero(newly_covered))
        selected[candidate_index] = True
        covered |= matrix[candidate_index]
        # Recompute from the covered mask to avoid cumulative floating-point drift.
        new_covered_weight = float(np.sum(element_weights[covered], dtype=np.float64))
        marginal_gain = new_covered_weight - covered_weight
        covered_weight = new_covered_weight
        fraction = covered_weight / total_weight

        selected_indices.append(candidate_index)
        selected_ids.append(ids[candidate_index])
        marginal_gains.append(marginal_gain)
        cumulative_gains.append(covered_weight)
        trajectory.append(
            GreedyCoverageStep(
                step=step_number,
                candidate_index=candidate_index,
                candidate_id=ids[candidate_index],
                marginal_gain=marginal_gain,
                cumulative_covered_weight=covered_weight,
                coverage_fraction=fraction,
                newly_covered_count=len(newly_covered_indices),
                newly_covered_indices=newly_covered_indices,
            )
        )

    covered_output = covered.copy()
    covered_output.setflags(write=False)
    return MaximumCoverageResult(
        selected_indices=tuple(selected_indices),
        selected_ids=tuple(selected_ids),
        marginal_gains=tuple(marginal_gains),
        cumulative_gains=tuple(cumulative_gains),
        coverage_fraction=covered_weight / total_weight,
        covered_weight=covered_weight,
        total_weight=total_weight,
        covered_mask=covered_output,
        trajectory=tuple(trajectory),
    )


def _as_score_matrix(name: str, values: ArrayLike, *, allow_nan: bool) -> FloatArray:
    """Validate a two-dimensional score or cluster-contribution matrix."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a two-dimensional float array") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values")
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _stable_rank_rows(scores: FloatArray, higher_is_hotter: bool) -> IntArray:
    """Compute one-based row-wise ranks with stable index-based tie breaking."""

    worst = -np.inf if higher_is_hotter else np.inf
    safe_scores = np.where(np.isfinite(scores), scores, worst)
    order_values = -safe_scores if higher_is_hotter else safe_scores
    order = np.argsort(order_values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int64)
    rows = np.arange(order.shape[0], dtype=np.intp)[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1, dtype=np.int64)
    return ranks


def _stable_point_order(scores: FloatArray, higher_is_hotter: bool) -> tuple[int, ...]:
    """Return a deterministic hotspot order for one score vector."""

    order_values = -scores if higher_is_hotter else scores
    return tuple(int(index) for index in np.argsort(order_values, kind="stable"))


def summarize_hotspot_rank_stability(
    bootstrap_scores: ArrayLike,
    *,
    point_scores: ArrayLike | None = None,
    hotspot_ids: Sequence[Hashable] | None = None,
    top_k: int = 3,
    confidence_level: float = 0.95,
    higher_is_hotter: bool = True,
    seed: int | None = None,
) -> HotspotRankStability:
    """Summarize hotspot rank stability from precomputed bootstrap scores.

    Args:
        bootstrap_scores: Matrix of shape ``(n_resamples, n_hotspots)``.
            NaN is allowed for an undefined hotspot score in a draw.
        point_scores: Optional score vector from the original data. If omitted,
            each hotspot's mean finite bootstrap score is used.
        hotspot_ids: Optional unique identifiers for hotspot columns.
        top_k: Rank threshold for the reported top-k probability.
        confidence_level: Equal-tail percentile interval level.
        higher_is_hotter: Rank larger scores first when true, or smaller scores
            first when false.
        seed: Optional provenance seed to retain in the returned summary.

    Returns:
        Score intervals, rank intervals, top-k probabilities, valid fractions,
        and the underlying score and rank draws.
    """

    scores = _as_score_matrix("bootstrap_scores", bootstrap_scores, allow_nan=True)
    if np.any(np.all(~np.isfinite(scores), axis=1)):
        raise ValueError("every bootstrap draw must have at least one finite hotspot score")
    if np.any(np.all(~np.isfinite(scores), axis=0)):
        raise ValueError("every hotspot must have at least one finite bootstrap score")
    n_hotspots = scores.shape[1]
    normalized_top_k = _validate_positive_integer("top_k", top_k)
    if normalized_top_k > n_hotspots:
        raise ValueError(f"top_k must not exceed the {n_hotspots} hotspots")
    confidence = _validate_confidence_level(confidence_level)
    if not isinstance(higher_is_hotter, (bool, np.bool_)):
        raise ValueError("higher_is_hotter must be boolean")
    higher = bool(higher_is_hotter)
    ids = _validate_ids(hotspot_ids, n_hotspots, name="hotspot_ids")

    if point_scores is None:
        original_scores = np.nanmean(scores, axis=0)
    else:
        try:
            original_scores = np.asarray(point_scores, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("point_scores must be convertible to a float vector") from exc
        if original_scores.ndim != 1 or len(original_scores) != n_hotspots:
            raise ValueError(f"point_scores must have shape ({n_hotspots},)")
        if not np.all(np.isfinite(original_scores)):
            raise ValueError("point_scores must contain only finite values")

    alpha_percent = 50.0 * (1.0 - confidence)
    score_ci_low = np.nanpercentile(scores, alpha_percent, axis=0, method="linear")
    score_ci_high = np.nanpercentile(scores, 100.0 - alpha_percent, axis=0, method="linear")
    ranks = _stable_rank_rows(scores, higher)
    # Undefined scores are not merely tied by their column order; they have no
    # estimable rank in that draw and are conservatively assigned the worst
    # possible rank. ``valid_fraction`` separately reports how often this
    # occurred.
    ranks[~np.isfinite(scores)] = n_hotspots
    rank_ci_low = np.percentile(ranks, alpha_percent, axis=0, method="linear")
    rank_ci_high = np.percentile(ranks, 100.0 - alpha_percent, axis=0, method="linear")
    median_rank = np.median(ranks, axis=0)
    finite = np.isfinite(scores)
    top_k_probability = np.mean((ranks <= normalized_top_k) & finite, axis=0)
    valid_fraction = np.mean(finite, axis=0)

    arrays = [
        original_scores,
        score_ci_low,
        score_ci_high,
        median_rank,
        rank_ci_low,
        rank_ci_high,
        top_k_probability,
        valid_fraction,
        scores,
        ranks,
    ]
    readonly_arrays: list[NDArray[Any]] = []
    for array in arrays:
        output = np.asarray(array).copy()
        output.setflags(write=False)
        readonly_arrays.append(output)

    normalized_seed: int | None
    if seed is None:
        normalized_seed = None
    else:
        normalized_seed = _validate_seed(seed)
    return HotspotRankStability(
        hotspot_ids=ids,
        point_scores=readonly_arrays[0],
        point_order=_stable_point_order(original_scores, higher),
        score_ci_low=readonly_arrays[1],
        score_ci_high=readonly_arrays[2],
        median_rank=readonly_arrays[3],
        rank_ci_low=readonly_arrays[4],
        rank_ci_high=readonly_arrays[5],
        top_k_probability=readonly_arrays[6],
        valid_fraction=readonly_arrays[7],
        bootstrap_scores=readonly_arrays[8],
        bootstrap_ranks=readonly_arrays[9],
        top_k=normalized_top_k,
        confidence_level=confidence,
        higher_is_hotter=higher,
        seed=normalized_seed,
    )


def bootstrap_hotspot_rank_stability(
    numerator: ArrayLike,
    denominator: ArrayLike | None = None,
    *,
    hotspot_ids: Sequence[Hashable] | None = None,
    n_resamples: int = 5_000,
    top_k: int = 3,
    confidence_level: float = 0.95,
    higher_is_hotter: bool = True,
    seed: int = 0,
) -> HotspotRankStability:
    """Cluster-bootstrap hotspot scores and summarize their rank stability.

    Rows are independent resampling clusters; columns are hotspots. For a
    ratio metric, provide weighted numerator and denominator contributions for
    every cluster/hotspot cell. The score in each draw is the ratio of the
    resampled sums. If ``denominator`` is omitted, the score is the mean
    resampled numerator contribution per cluster.

    Args:
        numerator: Matrix with shape ``(n_clusters, n_hotspots)``.
        denominator: Optional nonnegative matrix with the same shape. Each
            hotspot must have a positive denominator total in the original
            sample.
        hotspot_ids: Optional unique identifiers for hotspot columns.
        n_resamples: Number of cluster-bootstrap samples.
        top_k: Rank threshold for top-k selection probability.
        confidence_level: Equal-tail percentile interval level.
        higher_is_hotter: Rank larger scores first when true.
        seed: Deterministic NumPy random seed.

    Returns:
        A :class:`HotspotRankStability` containing score and rank uncertainty.

    Notes:
        The caller should place all dependent observations from a physical
        conflict pair or event into the same input row before calling this
        function. Numerator values may already contain TTC, separation, or VRU
        weights; the bootstrap does not redefine those scientific weights.
    """

    numerator_array = _as_score_matrix("numerator", numerator, allow_nan=False)
    n_clusters, n_hotspots = numerator_array.shape
    ids = _validate_ids(hotspot_ids, n_hotspots, name="hotspot_ids")
    resamples = _validate_positive_integer("n_resamples", n_resamples)
    normalized_seed = _validate_seed(seed)
    confidence = _validate_confidence_level(confidence_level)
    normalized_top_k = _validate_positive_integer("top_k", top_k)
    if normalized_top_k > n_hotspots:
        raise ValueError(f"top_k must not exceed the {n_hotspots} hotspots")
    if not isinstance(higher_is_hotter, (bool, np.bool_)):
        raise ValueError("higher_is_hotter must be boolean")

    denominator_array: FloatArray | None
    if denominator is None:
        denominator_array = None
        point_scores = np.mean(numerator_array, axis=0)
    else:
        denominator_array = _as_score_matrix("denominator", denominator, allow_nan=False)
        if denominator_array.shape != numerator_array.shape:
            raise ValueError(
                "denominator must have the same shape as numerator; "
                f"got {denominator_array.shape} and {numerator_array.shape}"
            )
        if np.any(denominator_array < 0.0):
            raise ValueError("denominator must contain only nonnegative values")
        denominator_total = np.sum(denominator_array, axis=0, dtype=np.float64)
        if np.any(denominator_total <= 0.0):
            bad = np.flatnonzero(denominator_total <= 0.0).tolist()
            raise ValueError(f"denominator totals must be positive for every hotspot; bad={bad}")
        point_scores = np.sum(numerator_array, axis=0, dtype=np.float64) / denominator_total

    rng = np.random.default_rng(normalized_seed)
    probabilities = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    # Batching avoids materializing an n_resamples x n_clusters x n_hotspots
    # tensor while retaining exact ordinary cluster-bootstrap multiplicities.
    # Keep the multinomial count matrix near or below roughly two million
    # entries. This matters when temporal blocks, rather than a few hundred
    # conflict pairs, are used as clusters.
    memory_bounded_batch = max(1, 2_000_000 // n_clusters)
    batch_size = max(1, min(512, resamples, memory_bounded_batch))
    score_chunks: list[FloatArray] = []
    n_collected = 0
    n_attempted = 0
    max_attempts = max(1_000, resamples * 100)
    while n_collected < resamples and n_attempted < max_attempts:
        draw_count = min(batch_size, max_attempts - n_attempted)
        counts = rng.multinomial(n_clusters, probabilities, size=draw_count)
        numerator_total = counts @ numerator_array
        if denominator_array is None:
            scores = numerator_total / n_clusters
        else:
            denominator_total = counts @ denominator_array
            scores = np.divide(
                numerator_total,
                denominator_total,
                out=np.full_like(numerator_total, np.nan, dtype=np.float64),
                where=denominator_total > 0.0,
            )
        # A draw with no estimable hotspot cannot contribute a ranking. Reject
        # that complete draw, while retaining partially estimable draws and
        # reporting their per-hotspot validity below.
        valid_rows = np.any(np.isfinite(scores), axis=1)
        if np.any(valid_rows):
            score_chunks.append(scores[valid_rows])
            n_collected += int(np.count_nonzero(valid_rows))
        n_attempted += draw_count

    if n_collected < resamples:
        raise ValueError(
            "too many hotspot bootstrap draws have zero denominator for every hotspot"
        )
    bootstrap_scores = np.concatenate(score_chunks, axis=0)[:resamples]

    return summarize_hotspot_rank_stability(
        bootstrap_scores,
        point_scores=point_scores,
        hotspot_ids=ids,
        top_k=normalized_top_k,
        confidence_level=confidence,
        higher_is_hotter=bool(higher_is_hotter),
        seed=normalized_seed,
    )


# A concise alias for code that already has bootstrap score draws.
hotspot_rank_stability = summarize_hotspot_rank_stability


__all__ = [
    "GreedyCoverageStep",
    "HotspotRankStability",
    "MaximumCoverageResult",
    "bootstrap_hotspot_rank_stability",
    "greedy_weighted_maximum_coverage",
    "hotspot_rank_stability",
    "summarize_hotspot_rank_stability",
]
