"""Moving-block bootstrap utilities for temporally dependent ratio metrics.

PACO metrics such as observability are ratios of accumulated counts.  The
statistically relevant estimator is therefore ``sum(numerator) /
sum(denominator)``, not the unweighted mean of per-frame ratios.  This module
implements a moving-block bootstrap that preserves that estimator while
resampling contiguous units of time.

The paired function uses exactly the same sampled blocks for both methods.  Its
delta is defined as ``method_b - method_a`` so that positive values denote an
improvement of B over A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapEstimate:
    """A point estimate and percentile interval from bootstrap draws.

    Attributes:
        estimate: Ratio-of-sums estimate on the original ordered observations.
        ci_low: Lower endpoint of the percentile confidence interval.
        ci_high: Upper endpoint of the percentile confidence interval.
        samples: One-dimensional array of valid bootstrap estimates.
        confidence_level: Confidence level used for the percentile interval.
        block_length: Number of ordered time units in each moving block.
        circular: Whether blocks were allowed to wrap around the record end.
        seed: Integer seed used to create the random number generator.
    """

    estimate: float
    ci_low: float
    ci_high: float
    samples: FloatArray
    confidence_level: float
    block_length: int
    circular: bool
    seed: int

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Return ``(ci_low, ci_high)`` for convenient unpacking."""

        return (self.ci_low, self.ci_high)

    @property
    def ci(self) -> tuple[float, float]:
        """Alias for :attr:`confidence_interval`."""

        return self.confidence_interval

    @property
    def point_estimate(self) -> float:
        """Alias for :attr:`estimate`."""

        return self.estimate


@dataclass(frozen=True)
class PairedBootstrapEstimate:
    """Paired moving-block estimates for methods A, B, and ``B - A``.

    All three sample arrays have the same length and bootstrap draw ordering.
    Consequently ``samples_delta`` is exactly ``samples_b - samples_a``.
    """

    estimate_a: float
    estimate_b: float
    estimate_delta: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    ci_delta: tuple[float, float]
    samples_a: FloatArray
    samples_b: FloatArray
    samples_delta: FloatArray
    confidence_level: float
    block_length: int
    circular: bool
    seed: int

    @property
    def delta(self) -> float:
        """Return the observed paired difference ``method_b - method_a``."""

        return self.estimate_delta

    @property
    def delta_confidence_interval(self) -> tuple[float, float]:
        """Return the percentile interval for ``method_b - method_a``."""

        return self.ci_delta


def _as_1d_float(name: str, values: ArrayLike, *, nonnegative: bool = False) -> FloatArray:
    """Convert an input to a finite, nonempty one-dimensional float array."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return array


def _as_time_array(time: ArrayLike, expected_length: int) -> NDArray[Any]:
    """Validate grouping labels used to aggregate repeated rows by time."""

    array = np.asarray(time)
    if array.ndim != 1:
        raise ValueError(f"time must be one-dimensional; got shape {array.shape}")
    if len(array) != expected_length:
        raise ValueError(
            f"time has length {len(array)}, but ratio components have length {expected_length}"
        )
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError("time must not contain NaN or infinite values")
    if array.dtype.kind in "mM" and np.any(np.isnat(array)):
        raise ValueError("time must not contain NaT values")
    if array.dtype.kind == "O":
        for value in array:
            if value is None:
                raise ValueError("time must not contain missing values")
            try:
                is_missing = bool(value != value)  # NaN-like objects are unequal to themselves.
            except (TypeError, ValueError):
                # Some custom scalar types do not define a scalar inequality.
                is_missing = False
            if is_missing:
                raise ValueError("time must not contain missing values")
    return array


def _time_inverse(time: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Return sorted unique time labels and each row's group index."""

    try:
        unique_time, inverse = np.unique(time, return_inverse=True)
    except TypeError as exc:
        raise ValueError("time values must be mutually orderable") from exc
    return unique_time, inverse.astype(np.intp, copy=False)


def aggregate_ratio_components_by_time(
    time: ArrayLike,
    numerator: ArrayLike,
    denominator: ArrayLike,
) -> tuple[NDArray[Any], FloatArray, FloatArray]:
    """Aggregate repeated observations into ordered numerator/denominator sums.

    This helper is useful when a frame contains many ego positions or objects.
    Rows sharing a time label are summed, and the returned labels are sorted in
    ascending order.  The result can be passed directly to
    :func:`moving_block_bootstrap_ratio`.

    Args:
        time: Frame identifiers or other sortable temporal labels.
        numerator: Per-row numerator contributions.
        denominator: Per-row denominator contributions. Values must be
            nonnegative.

    Returns:
        ``(ordered_time, numerator_by_time, denominator_by_time)``.
    """

    numerator_array = _as_1d_float("numerator", numerator)
    denominator_array = _as_1d_float("denominator", denominator, nonnegative=True)
    if len(numerator_array) != len(denominator_array):
        raise ValueError("numerator and denominator must have the same length")
    time_array = _as_time_array(time, len(numerator_array))
    unique_time, inverse = _time_inverse(time_array)
    n_groups = len(unique_time)
    numerator_sum = np.bincount(inverse, weights=numerator_array, minlength=n_groups)
    denominator_sum = np.bincount(inverse, weights=denominator_array, minlength=n_groups)
    return (
        unique_time,
        numerator_sum.astype(np.float64, copy=False),
        denominator_sum.astype(np.float64, copy=False),
    )


def ratio_of_sums(numerator: ArrayLike, denominator: ArrayLike) -> float:
    """Compute ``sum(numerator) / sum(denominator)`` with validation."""

    numerator_array = _as_1d_float("numerator", numerator)
    denominator_array = _as_1d_float("denominator", denominator, nonnegative=True)
    if len(numerator_array) != len(denominator_array):
        raise ValueError("numerator and denominator must have the same length")
    denominator_total = float(np.sum(denominator_array, dtype=np.float64))
    if denominator_total <= 0.0:
        raise ValueError("denominator must have a strictly positive total")
    return float(np.sum(numerator_array, dtype=np.float64) / denominator_total)


def _validate_bootstrap_options(
    n_time: int,
    block_length: int,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[int, int, float, int]:
    """Validate and normalize scalar bootstrap options."""

    if isinstance(block_length, (bool, np.bool_)) or not isinstance(
        block_length, (int, np.integer)
    ):
        raise ValueError("block_length must be an integer")
    if block_length < 1 or block_length > n_time:
        raise ValueError(f"block_length must be between 1 and {n_time}; got {block_length}")
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise ValueError("n_resamples must be an integer")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if isinstance(confidence_level, (bool, np.bool_)):
        raise ValueError("confidence_level must be a number strictly between 0 and 1")
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be a number strictly between 0 and 1") from exc
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return int(block_length), int(n_resamples), confidence, int(seed)


def _block_sums(values: FloatArray, length: int, circular: bool) -> FloatArray:
    """Return a sum for every admissible moving block of a fixed length."""

    n_time = len(values)
    if circular:
        extended = np.concatenate((values, values[: max(0, length - 1)]))
        starts = np.arange(n_time, dtype=np.intp)
    else:
        extended = values
        starts = np.arange(n_time - length + 1, dtype=np.intp)
    cumulative = np.concatenate(([0.0], np.cumsum(extended, dtype=np.float64)))
    return cumulative[starts + length] - cumulative[starts]


def _draw_ratio_samples(
    components: tuple[tuple[FloatArray, FloatArray], ...],
    *,
    block_length: int,
    n_resamples: int,
    circular: bool,
    rng: np.random.Generator,
) -> tuple[FloatArray, ...]:
    """Draw valid ratio samples, rejecting draws with a zero denominator."""

    n_time = len(components[0][0])
    n_blocks = int(np.ceil(n_time / block_length))
    remainder = n_time - (n_blocks - 1) * block_length
    full_sums = [
        (_block_sums(num, block_length, circular), _block_sums(den, block_length, circular))
        for num, den in components
    ]
    partial_sums = [
        (_block_sums(num, remainder, circular), _block_sums(den, remainder, circular))
        for num, den in components
    ]
    n_starts = len(full_sums[0][0])

    collected: list[list[FloatArray]] = [[] for _ in components]
    n_collected = 0
    n_attempted = 0
    max_attempts = max(1_000, n_resamples * 100)

    while n_collected < n_resamples and n_attempted < max_attempts:
        batch_size = min(max(n_resamples - n_collected, 64), max_attempts - n_attempted)
        starts = rng.integers(0, n_starts, size=(batch_size, n_blocks), endpoint=False)
        valid = np.ones(batch_size, dtype=bool)
        ratios: list[FloatArray] = []
        for (num_full, den_full), (num_partial, den_partial) in zip(
            full_sums, partial_sums, strict=True
        ):
            if n_blocks == 1:
                numerator_total = num_partial[starts[:, 0]]
                denominator_total = den_partial[starts[:, 0]]
            else:
                numerator_total = np.sum(num_full[starts[:, :-1]], axis=1)
                numerator_total += num_partial[starts[:, -1]]
                denominator_total = np.sum(den_full[starts[:, :-1]], axis=1)
                denominator_total += den_partial[starts[:, -1]]
            valid &= denominator_total > 0.0
            ratios.append(
                np.divide(
                    numerator_total,
                    denominator_total,
                    out=np.full(batch_size, np.nan, dtype=np.float64),
                    where=denominator_total > 0.0,
                )
            )

        valid_count = int(np.count_nonzero(valid))
        if valid_count:
            for destination, ratio in zip(collected, ratios, strict=True):
                destination.append(ratio[valid])
            n_collected += valid_count
        n_attempted += batch_size

    if n_collected < n_resamples:
        raise ValueError(
            "too many bootstrap draws have a zero denominator; aggregate to a coarser "
            "time unit or use a longer block"
        )

    output: list[FloatArray] = []
    for chunks in collected:
        samples = np.concatenate(chunks)[:n_resamples].astype(np.float64, copy=False)
        samples.setflags(write=False)
        output.append(samples)
    return tuple(output)


def _percentile_interval(samples: FloatArray, confidence_level: float) -> tuple[float, float]:
    """Compute a two-sided equal-tail percentile interval."""

    alpha_percent = 50.0 * (1.0 - confidence_level)
    low, high = np.percentile(
        samples,
        [alpha_percent, 100.0 - alpha_percent],
        method="linear",
    )
    return (float(low), float(high))


def moving_block_bootstrap_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    block_length: int,
    n_resamples: int = 5_000,
    confidence_level: float = 0.95,
    circular: bool = True,
    seed: int = 0,
    time: ArrayLike | None = None,
) -> BootstrapEstimate:
    """Bootstrap a ratio-of-sums estimator over contiguous time blocks.

    Args:
        numerator: Ordered per-time numerator contributions, or per-row
            contributions when ``time`` is supplied.
        denominator: Ordered nonnegative denominator contributions.
        block_length: Number of unique time units per moving block.
        n_resamples: Number of valid bootstrap draws to return.
        confidence_level: Two-sided percentile confidence level.
        circular: If true, a block may wrap from the record end to its start.
            If false, only blocks wholly contained in the record are sampled.
        seed: Deterministic NumPy random seed.
        time: Optional frame/time labels. Repeated labels are summed before the
            bootstrap and sorted in temporal-label order.

    Returns:
        A :class:`BootstrapEstimate` containing the point estimate, interval,
        and the bootstrap sample distribution.
    """

    numerator_array = _as_1d_float("numerator", numerator)
    denominator_array = _as_1d_float("denominator", denominator, nonnegative=True)
    if len(numerator_array) != len(denominator_array):
        raise ValueError("numerator and denominator must have the same length")
    if time is not None:
        _, numerator_array, denominator_array = aggregate_ratio_components_by_time(
            time, numerator_array, denominator_array
        )
    if not isinstance(circular, (bool, np.bool_)):
        raise ValueError("circular must be boolean")
    estimate = ratio_of_sums(numerator_array, denominator_array)
    block, resamples, confidence, normalized_seed = _validate_bootstrap_options(
        len(numerator_array), block_length, n_resamples, confidence_level, seed
    )
    rng = np.random.default_rng(normalized_seed)
    (samples,) = _draw_ratio_samples(
        ((numerator_array, denominator_array),),
        block_length=block,
        n_resamples=resamples,
        circular=bool(circular),
        rng=rng,
    )
    ci_low, ci_high = _percentile_interval(samples, confidence)
    return BootstrapEstimate(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        samples=samples,
        confidence_level=confidence,
        block_length=block,
        circular=bool(circular),
        seed=normalized_seed,
    )


def paired_moving_block_bootstrap_ratio(
    numerator_a: ArrayLike,
    denominator_a: ArrayLike,
    numerator_b: ArrayLike,
    denominator_b: ArrayLike,
    *,
    block_length: int,
    n_resamples: int = 5_000,
    confidence_level: float = 0.95,
    circular: bool = True,
    seed: int = 0,
    time: ArrayLike | None = None,
) -> PairedBootstrapEstimate:
    """Paired moving-block bootstrap for two ratio-of-sums estimators.

    The same moving blocks are used for A and B in every draw. The reported
    difference and its confidence interval are always ``B - A``.

    Args:
        numerator_a: Numerator contributions for method A.
        denominator_a: Nonnegative denominator contributions for method A.
        numerator_b: Numerator contributions for method B.
        denominator_b: Nonnegative denominator contributions for method B.
        block_length: Number of unique time units per moving block.
        n_resamples: Number of valid bootstrap draws to return.
        confidence_level: Two-sided percentile confidence level.
        circular: Whether moving blocks may wrap around the record boundary.
        seed: Deterministic NumPy random seed.
        time: Optional shared frame/time labels. Repeated labels are summed for
            each method before resampling.

    Returns:
        Point estimates, percentile intervals, and paired bootstrap samples for
        A, B, and ``B - A``.
    """

    num_a = _as_1d_float("numerator_a", numerator_a)
    den_a = _as_1d_float("denominator_a", denominator_a, nonnegative=True)
    num_b = _as_1d_float("numerator_b", numerator_b)
    den_b = _as_1d_float("denominator_b", denominator_b, nonnegative=True)
    lengths = {len(num_a), len(den_a), len(num_b), len(den_b)}
    if len(lengths) != 1:
        raise ValueError("all paired numerator and denominator arrays must have the same length")

    if time is not None:
        time_array = _as_time_array(time, len(num_a))
        _, inverse = _time_inverse(time_array)
        n_groups = int(np.max(inverse)) + 1
        num_a = np.bincount(inverse, weights=num_a, minlength=n_groups).astype(np.float64)
        den_a = np.bincount(inverse, weights=den_a, minlength=n_groups).astype(np.float64)
        num_b = np.bincount(inverse, weights=num_b, minlength=n_groups).astype(np.float64)
        den_b = np.bincount(inverse, weights=den_b, minlength=n_groups).astype(np.float64)

    if not isinstance(circular, (bool, np.bool_)):
        raise ValueError("circular must be boolean")

    estimate_a = ratio_of_sums(num_a, den_a)
    estimate_b = ratio_of_sums(num_b, den_b)
    block, resamples, confidence, normalized_seed = _validate_bootstrap_options(
        len(num_a), block_length, n_resamples, confidence_level, seed
    )
    rng = np.random.default_rng(normalized_seed)
    samples_a, samples_b = _draw_ratio_samples(
        ((num_a, den_a), (num_b, den_b)),
        block_length=block,
        n_resamples=resamples,
        circular=bool(circular),
        rng=rng,
    )
    samples_delta = (samples_b - samples_a).astype(np.float64, copy=False)
    samples_delta.setflags(write=False)
    return PairedBootstrapEstimate(
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        estimate_delta=estimate_b - estimate_a,
        ci_a=_percentile_interval(samples_a, confidence),
        ci_b=_percentile_interval(samples_b, confidence),
        ci_delta=_percentile_interval(samples_delta, confidence),
        samples_a=samples_a,
        samples_b=samples_b,
        samples_delta=samples_delta,
        confidence_level=confidence,
        block_length=block,
        circular=bool(circular),
        seed=normalized_seed,
    )


# Concise aliases retained for callers that do not need the estimator name in
# the function itself. Both aliases preserve ratio-of-sums semantics.
moving_block_bootstrap = moving_block_bootstrap_ratio
paired_moving_block_bootstrap = paired_moving_block_bootstrap_ratio


__all__ = [
    "BootstrapEstimate",
    "PairedBootstrapEstimate",
    "aggregate_ratio_components_by_time",
    "moving_block_bootstrap",
    "moving_block_bootstrap_ratio",
    "paired_moving_block_bootstrap",
    "paired_moving_block_bootstrap_ratio",
    "ratio_of_sums",
]
