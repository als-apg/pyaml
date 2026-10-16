"""
First-turn trajectory correction tool.

This module holds the :class:`Trajectory` tool and the records it publishes and
returns:

- :class:`Reason` and :class:`SpectrumKind`, two closed enums that serialise as
  their member names;
- :class:`TrajectoryLayout`, the read-only, value-compared facts of a loaded
  response matrix;
- :class:`PseudoInverse`, the result of :meth:`Trajectory.pseudo_inverse`;
- :class:`TrajectoryStep`, the result of :meth:`Trajectory.correct`.

Every record is a frozen, slotted dataclass. Name fields are tuples of ``str``.
Every array a record stores is its own copy and is read-only, so a returned record
can never reach or change the tool's state.
"""

import copy
import dataclasses
import logging
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ..common.exception import PyAMLException
from ..validation import DynamicValidation, register_schema
from .orbit_response_matrix_data import OrbitResponseMatrixData
from .tuning_tool import TuningTool

__all__ = ["PseudoInverse", "Reason", "SpectrumKind", "Trajectory", "TrajectoryLayout", "TrajectoryStep"]

logger = logging.getLogger(__name__)

PYAMLCLASS = "Trajectory"


class Reason(StrEnum):
    """
    Why a :class:`TrajectoryStep` has the step it has.

    The reason is the first match, in this order, of NO_BEAM, NO_INVERSE,
    NO_UPSTREAM, NO_UPSTREAM_H / NO_UPSTREAM_V, ON_REFERENCE and OK. Each member's
    value equals its name, so ``json.dumps`` writes the name.

    Attributes
    ----------
    NO_BEAM
        No BPM has beam (every entry of ``x`` is NaN).
    NO_INVERSE
        Every inverse weight is zero (``n_discard == L``).
    NO_UPSTREAM
        Neither plane has a live corrector at or before the last BPM with beam.
    NO_UPSTREAM_H
        The horizontal plane has no live corrector at or before the last BPM with beam.
    NO_UPSTREAM_V
        The vertical plane has no live corrector at or before the last BPM with beam.
    ON_REFERENCE
        The trajectory already equals the reference.
    OK
        A correction step was computed.
    """

    NO_BEAM = "NO_BEAM"
    NO_INVERSE = "NO_INVERSE"
    NO_UPSTREAM = "NO_UPSTREAM"
    NO_UPSTREAM_H = "NO_UPSTREAM_H"
    NO_UPSTREAM_V = "NO_UPSTREAM_V"
    ON_REFERENCE = "ON_REFERENCE"
    OK = "OK"


class SpectrumKind(StrEnum):
    """
    Kind of one position of the singular spectrum of a :class:`PseudoInverse`.

    A position with two causes takes the first of STRUCTURAL > RANK_DISCARDED >
    DISCARDED > LIVE. Each member's value equals its name, so ``json.dumps`` writes
    the name.

    Attributes
    ----------
    LIVE
        A live singular value whose inverse weight is applied.
    RANK_DISCARDED
        A live singular value dropped by the rank guard (``alpha == 0`` only).
    DISCARDED
        A live singular value zeroed by ``n_discard``.
    STRUCTURAL
        A structural zero: a position past the live singular values, caused by
        correctors with no response.
    """

    LIVE = "LIVE"
    RANK_DISCARDED = "RANK_DISCARDED"
    DISCARDED = "DISCARDED"
    STRUCTURAL = "STRUCTURAL"


_STATUS: dict[Reason, str] = {
    Reason.NO_BEAM: "no BPM with beam",
    Reason.NO_INVERSE: "every singular value is discarded; no correction is possible",
    Reason.NO_UPSTREAM: "no corrector upstream of the last BPM with beam",
    Reason.NO_UPSTREAM_H: "no horizontal corrector upstream of the last BPM with beam",
    Reason.NO_UPSTREAM_V: "no vertical corrector upstream of the last BPM with beam",
    Reason.ON_REFERENCE: "already on reference",
    Reason.OK: "",
}


def _names(value: Any, field: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of str, raising TypeError on any other element."""
    names = tuple(value)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{field} must hold str names, got {type(name).__name__} {name!r}")
    return names


def _frozen_array(value: Any, field: str, ndim: int, dtype: Any = float) -> np.ndarray:
    """Return a read-only copy of ``value`` with the given rank and dtype."""
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional, got shape {array.shape}")
    array.flags.writeable = False
    return array


def _set(record: Any, field: str, value: Any) -> None:
    object.__setattr__(record, field, value)


@dataclass(frozen=True, slots=True)
class TrajectoryLayout:
    """
    Read-only facts of a loaded trajectory response matrix.

    Two layouts are equal when every name field is equal. All counts are derived
    from the names.

    Parameters
    ----------
    bpm_names_h : tuple[str, ...]
        BPM names of the horizontal row half, in row order.
    bpm_names_v : tuple[str, ...]
        BPM names of the vertical row half, in row order.
    hcorr_names : tuple[str, ...]
        Horizontal corrector names, in column order.
    vcorr_names : tuple[str, ...]
        Vertical corrector names, in column order.
    no_response_names : tuple[str, ...]
        Names of the correctors whose response column is zero, in column order.

    Attributes
    ----------
    n_bpm : int
        Number of BPMs per plane.
    n_h, n_v : int
        Number of horizontal and vertical correctors.
    n_cor : int
        Number of correctors, ``n_h + n_v``.
    n_live : int
        Number of correctors with a response.
    L : int
        Length of the singular spectrum, ``min(2 * n_bpm, n_cor)``.
    k_live : int
        Number of live singular values, ``min(2 * n_bpm, n_live)``.
    n_struct : int
        Number of structural zeros, ``L - k_live``.
    """

    bpm_names_h: tuple[str, ...]
    bpm_names_v: tuple[str, ...]
    hcorr_names: tuple[str, ...]
    vcorr_names: tuple[str, ...]
    no_response_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _set(self, field.name, _names(getattr(self, field.name), field.name))

    @property
    def n_bpm(self) -> int:
        return len(self.bpm_names_h)

    @property
    def n_h(self) -> int:
        return len(self.hcorr_names)

    @property
    def n_v(self) -> int:
        return len(self.vcorr_names)

    @property
    def n_cor(self) -> int:
        return self.n_h + self.n_v

    @property
    def n_live(self) -> int:
        return self.n_cor - len(self.no_response_names)

    @property
    def L(self) -> int:
        return min(2 * self.n_bpm, self.n_cor)

    @property
    def k_live(self) -> int:
        return min(2 * self.n_bpm, self.n_live)

    @property
    def n_struct(self) -> int:
        return self.L - self.k_live


@dataclass(frozen=True, slots=True, eq=False)
class PseudoInverse:
    """
    Guarded, damped pseudo-inverse of the trajectory response matrix.

    ``singular_values``, ``inverse_weights`` and ``spectrum_kind`` are index-aligned
    and have length ``L``. Every array is a read-only copy. Records compare by
    identity.

    Parameters
    ----------
    matrix : numpy.ndarray
        The ``n_cor x 2*n_bpm`` inverse. Rows of correctors with no response are
        exactly zero.
    singular_values : numpy.ndarray
        The live singular values in descending order, then the structural zeros.
    inverse_weights : numpy.ndarray
        The diagonal applied to each singular direction, gain included; zero at
        every structural, rank-discarded and discarded position.
    spectrum_kind : tuple[SpectrumKind, ...]
        The kind of each position.
    rank_discarded : int
        Number of live singular values the rank guard dropped (``alpha == 0`` only).
    no_response_correctors : numpy.ndarray
        Column indices of the correctors with no response.
    no_response_names : tuple[str, ...]
        Their names, as in the layout.
    alpha : float
        Tikhonov parameter used.
    n_discard : int
        Number of trailing spectrum positions zeroed.
    gain : float
        Gain applied.
    """

    matrix: np.ndarray
    singular_values: np.ndarray
    inverse_weights: np.ndarray
    spectrum_kind: tuple[SpectrumKind, ...]
    rank_discarded: int
    no_response_correctors: np.ndarray
    no_response_names: tuple[str, ...]
    alpha: float
    n_discard: int
    gain: float

    def __post_init__(self) -> None:
        _set(self, "matrix", _frozen_array(self.matrix, "matrix", 2))
        _set(self, "singular_values", _frozen_array(self.singular_values, "singular_values", 1))
        _set(self, "inverse_weights", _frozen_array(self.inverse_weights, "inverse_weights", 1))
        _set(self, "spectrum_kind", tuple(SpectrumKind(kind) for kind in self.spectrum_kind))
        _set(
            self,
            "no_response_correctors",
            _frozen_array(self.no_response_correctors, "no_response_correctors", 1, np.intp),
        )
        _set(self, "no_response_names", _names(self.no_response_names, "no_response_names"))
        lengths = {len(self.singular_values), len(self.inverse_weights), len(self.spectrum_kind)}
        if len(lengths) != 1:
            raise ValueError(
                "singular_values, inverse_weights and spectrum_kind must have one length, got "
                f"{len(self.singular_values)}, {len(self.inverse_weights)}, {len(self.spectrum_kind)}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryStep:
    """
    Corrector step computed by :meth:`Trajectory.correct`.

    The increment to apply is ``-step``. Records compare by identity.

    Parameters
    ----------
    step : numpy.ndarray
        Read-only step in response-matrix column order: all horizontal correctors,
        then all vertical ones.
    names : tuple[str, ...]
        Corrector names, aligned with ``step``.
    n_h : int
        Number of horizontal correctors; ``step[:n_h]`` is the horizontal plane.
    last_bpm : int or None
        Position of the last BPM with beam in the first half of ``bpm_index``, named
        by ``layout.bpm_names_h``; None when no BPM has beam.
    zeroed_h, zeroed_v : tuple[str, ...]
        Correctors of each plane whose step was zeroed by threading, in column order.
    pseudo_inverse : PseudoInverse
        The inverse the step was solved with.
    reason : Reason
        Why the step is what it is.

    Attributes
    ----------
    step_h, step_v : numpy.ndarray
        Read-only horizontal and vertical parts of ``step``.
    status : str
        Sentence describing ``reason``; empty for OK.
    all_zero : bool
        True when every entry of ``step`` is zero.
    """

    step: np.ndarray
    names: tuple[str, ...]
    n_h: int
    last_bpm: Optional[int]
    zeroed_h: tuple[str, ...]
    zeroed_v: tuple[str, ...]
    pseudo_inverse: PseudoInverse
    reason: Reason

    def __post_init__(self) -> None:
        _set(self, "step", _frozen_array(self.step, "step", 1))
        _set(self, "names", _names(self.names, "names"))
        _set(self, "zeroed_h", _names(self.zeroed_h, "zeroed_h"))
        _set(self, "zeroed_v", _names(self.zeroed_v, "zeroed_v"))
        _set(self, "reason", Reason(self.reason))
        if len(self.names) != len(self.step):
            raise ValueError(f"names has {len(self.names)} entries, step has {len(self.step)}")

    @property
    def step_h(self) -> np.ndarray:
        return self.step[: self.n_h]

    @property
    def step_v(self) -> np.ndarray:
        return self.step[self.n_h :]

    @property
    def status(self) -> str:
        return _STATUS[self.reason]

    @property
    def all_zero(self) -> bool:
        return not np.any(self.step)


@dataclass(frozen=True, slots=True, eq=False)
class _RMState:
    """
    Private, immutable state of a loaded response matrix.

    Built completely and assigned to the tool in one step by ``load()``, its only
    writer; the solve reads ``matrix`` and ``live`` from it.

    Parameters
    ----------
    response_matrix : OrbitResponseMatrixData
        A private deep copy of the loaded data.
    layout : TrajectoryLayout
        The published facts.
    matrix : numpy.ndarray
        Read-only ``2*n_bpm x n_cor`` response matrix.
    live : numpy.ndarray
        Read-only boolean mask over columns, True where the corrector responds.
    """

    response_matrix: OrbitResponseMatrixData
    layout: TrajectoryLayout
    matrix: np.ndarray
    live: np.ndarray

    def __post_init__(self) -> None:
        _set(self, "matrix", _frozen_array(self.matrix, "matrix", 2))
        _set(self, "live", _frozen_array(self.live, "live", 1, bool))


@dataclass(frozen=True, slots=True)
class _Settings:
    """
    Resolved, checked solver settings.

    Only :func:`_resolve_settings` builds one, so every instance holds values that
    passed its checks, as Python scalars.

    Parameters
    ----------
    alpha : float
        Tikhonov parameter, finite, ``>= 0``, with ``alpha * alpha`` finite.
    n_discard : int
        Number of trailing spectrum positions to zero, in ``[0, L]``.
    gain : float
        Gain, finite and ``> 0``.
    """

    alpha: float
    n_discard: int
    gain: float


def _unwrap(value: Any) -> Any:
    """Return the scalar held by a 0-d ndarray (as ``np.load`` gives a stored scalar), else ``value``."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value[()]
    return value


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _real(value: Any, field: str) -> float:
    """Return ``value`` as a finite float, raising PyAMLException on a non-real or non-finite value."""
    value = _unwrap(value)
    if _is_bool(value) or not isinstance(value, numbers.Real):
        raise PyAMLException(f"{field} must be a real number, got {type(value).__name__} {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise PyAMLException(f"{field} must be finite, got {result!r}")
    return result


def _resolve_settings(
    alpha: Any,
    n_discard: Any,
    gain: Any,
    L: Optional[int],
    defaults: Optional[_Settings] = None,
) -> _Settings:
    """
    Check and resolve the solver settings ``alpha``, ``n_discard`` and ``gain``.

    This is the one resolution routine of :class:`Trajectory`, used on every path:
    at construction (where pyAML's validation layer may already have coerced the
    values), at construction under ``validate=False``, at every ``load()`` (to
    re-check the configured ``n_discard`` against the new ``L``) and at every call
    that takes per-call overrides. It is a pure function of its arguments.

    Parameters
    ----------
    alpha : real or None
        Tikhonov parameter: a finite real ``>= 0`` whose square is finite.
    n_discard : int or None
        Number of trailing spectrum positions to zero: an integer in ``[0, L]``.
    gain : real or None
        Gain: a finite real ``> 0``.
    L : int or None
        Length of the singular spectrum of the loaded response matrix. None when no
        response matrix is loaded; only ``n_discard >= 0`` is checked then, and
        ``load()`` checks the upper bound.
    defaults : _Settings, optional
        The configured settings. A None argument takes its value from here; the
        result is checked against ``L`` either way. Without ``defaults`` (at
        construction) a None argument is an error.

    Returns
    -------
    _Settings
        The checked settings as Python ``float``, ``int`` and ``float``.

    Raises
    ------
    PyAMLException
        When a setting is None without a configured value, has the wrong type, or is
        out of range. The message names the setting.

    Notes
    -----
    ``bool`` and ``numpy.bool_`` are rejected for every setting. NumPy integer and
    floating scalars, and 0-d arrays holding one (as ``np.load`` returns a stored
    scalar), are accepted and converted to Python scalars. ``n_discard`` must be an
    integer type: ``1.0`` is rejected, not truncated.
    """
    if defaults is not None:
        alpha = defaults.alpha if alpha is None else alpha
        n_discard = defaults.n_discard if n_discard is None else n_discard
        gain = defaults.gain if gain is None else gain
    for field, value in (("alpha", alpha), ("n_discard", n_discard), ("gain", gain)):
        if value is None:
            raise PyAMLException(f"{field} must be given, got None")

    alpha_value = _real(alpha, "alpha")
    if alpha_value < 0.0:
        raise PyAMLException(f"alpha must be >= 0, got {alpha_value!r}")
    if not math.isfinite(alpha_value * alpha_value):
        raise PyAMLException(f"alpha is too large: alpha**2 is not finite for alpha={alpha_value!r}")
    alpha_value += 0.0  # -0.0 -> 0.0

    n_discard = _unwrap(n_discard)
    if _is_bool(n_discard) or not isinstance(n_discard, numbers.Integral):
        raise PyAMLException(f"n_discard must be an integer, got {type(n_discard).__name__} {n_discard!r}")
    n_discard_value = int(n_discard)
    if n_discard_value < 0 or (L is not None and n_discard_value > L):
        bound = "[0, L]" if L is None else f"[0, {L}]"
        raise PyAMLException(f"n_discard must be an integer in {bound}, got {n_discard_value}")

    gain_value = _real(gain, "gain")
    if gain_value <= 0.0:
        raise PyAMLException(f"gain must be > 0, got {gain_value!r}")

    return _Settings(alpha=alpha_value, n_discard=n_discard_value, gain=gain_value)


# ---------------------------------------------------------------------------
# Response-matrix entry point
# ---------------------------------------------------------------------------

_RM_FIELDS = ("matrix", "observable_names", "variable_names", "rf_response", "variable_planes", "observable_planes")
_RM_REQUIRED = ("matrix", "observable_names")


def _rm_fields(response_matrix: Any, source: str) -> dict[str, Any]:
    """
    Return the six ``OrbitResponseMatrixData`` fields of an instance or a mapping.

    A mapping must hold ``matrix`` and ``observable_names``; the other four fields
    default to None. Any other key, a build key such as ``type`` included, is
    rejected.
    """
    if isinstance(response_matrix, OrbitResponseMatrixData):
        return {field: getattr(response_matrix, field) for field in _RM_FIELDS}
    if isinstance(response_matrix, Mapping):
        unknown = sorted(str(key) for key in response_matrix if key not in _RM_FIELDS)
        if unknown:
            raise PyAMLException(f"{source}: unknown response-matrix keys {unknown}; expected only {list(_RM_FIELDS)}")
        missing = [field for field in _RM_REQUIRED if field not in response_matrix]
        if missing:
            raise PyAMLException(f"{source}: missing response-matrix keys {missing}")
        return {field: response_matrix.get(field) for field in _RM_FIELDS}
    raise PyAMLException(
        f"{source}: expected an OrbitResponseMatrixData file, mapping or instance, got {type(response_matrix).__name__}"
    )


def _str_list(value: Any, field: str, source: str) -> list[str]:
    """Return ``value`` as a list of str, raising PyAMLException naming ``source`` otherwise."""
    if value is None:
        raise PyAMLException(f"{source}: {field} is missing")
    if not isinstance(value, (list, tuple)):
        raise PyAMLException(f"{source}: {field} must be a list of str, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise PyAMLException(f"{source}: {field} must hold str, got {type(item).__name__} {item!r}")
    return list(value)


def _plane_major(planes: list[str], field: str, source: str) -> int:
    """Check that ``planes`` is every 'H' then every 'V'; return the number of 'H' entries."""
    bad = sorted({plane for plane in planes if plane not in ("H", "V")})
    if bad:
        raise PyAMLException(f"{source}: {field} must hold only 'H' and 'V', got {bad}")
    n_h = planes.count("H")
    if planes != ["H"] * n_h + ["V"] * (len(planes) - n_h):
        raise PyAMLException(f"{source}: {field} must be plane-major (every 'H' before every 'V')")
    return n_h


def _duplicates(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for name in names:
        (duplicated if name in seen else seen).add(name)
    return sorted(duplicated)


def _matrix(rows: Any, source: str) -> np.ndarray:
    """Return ``rows`` as a finite, rectangular, non-empty float matrix, raising PyAMLException otherwise."""
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise PyAMLException(f"{source}: matrix must be a non-empty list of rows")
    for row in rows:
        if not isinstance(row, (list, tuple)):
            raise PyAMLException(f"{source}: every matrix row must be a list, got {type(row).__name__}")
        for value in row:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise PyAMLException(f"{source}: matrix entries must be real numbers, got {type(value).__name__}")
    lengths = sorted({len(row) for row in rows})
    if len(lengths) != 1 or lengths[0] == 0:
        raise PyAMLException(f"{source}: matrix must be rectangular and non-empty, got row lengths {lengths}")
    matrix = np.array(rows, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise PyAMLException(f"{source}: matrix must be finite")
    return matrix


def _build_state(fields: dict[str, Any], source: str, settings: _Settings) -> _RMState:
    """
    Check every fact of a response matrix once and build the complete private state.

    Raises PyAMLException naming ``source`` on the first failed check, a configured
    ``n_discard`` larger than the new ``L`` included. Nothing of ``fields`` is
    aliased by the returned state.
    """
    observable_names = _str_list(fields["observable_names"], "observable_names", source)
    variable_names = _str_list(fields["variable_names"], "variable_names", source)
    observable_planes = _str_list(fields["observable_planes"], "observable_planes", source)
    variable_planes = _str_list(fields["variable_planes"], "variable_planes", source)
    if fields["rf_response"] is not None:
        raise PyAMLException(f"{source}: a trajectory response matrix must not carry an rf_response")
    matrix = _matrix(fields["matrix"], source)

    n_rows, n_cols = matrix.shape
    if not n_rows == len(observable_names) == len(observable_planes):
        raise PyAMLException(
            f"{source}: matrix has {n_rows} rows but {len(observable_names)} observable_names "
            f"and {len(observable_planes)} observable_planes"
        )
    if not n_cols == len(variable_names) == len(variable_planes):
        raise PyAMLException(
            f"{source}: matrix has {n_cols} columns but {len(variable_names)} variable_names "
            f"and {len(variable_planes)} variable_planes"
        )

    n_bpm = _plane_major(observable_planes, "observable_planes", source)
    if 2 * n_bpm != n_rows:
        raise PyAMLException(
            f"{source}: observable_planes must have equal row counts per plane, got {n_bpm} 'H' and {n_rows - n_bpm} 'V'"
        )
    n_h = _plane_major(variable_planes, "variable_planes", source)

    bpm_names_h, bpm_names_v = observable_names[:n_bpm], observable_names[n_bpm:]
    for part, names in (("H row half", bpm_names_h), ("V row half", bpm_names_v), ("variable_names", variable_names)):
        duplicated = _duplicates(names)
        if duplicated:
            raise PyAMLException(f"{source}: names must be unique within the {part}, duplicated {duplicated}")

    live = np.any(matrix != 0.0, axis=0)
    if not np.any(live):
        raise PyAMLException(f"{source}: no corrector has a response (every matrix column is zero)")

    layout = TrajectoryLayout(
        bpm_names_h=bpm_names_h,
        bpm_names_v=bpm_names_v,
        hcorr_names=variable_names[:n_h],
        vcorr_names=variable_names[n_h:],
        no_response_names=[name for name, responds in zip(variable_names, live, strict=True) if not responds],
    )
    try:
        _resolve_settings(None, None, None, L=layout.L, defaults=settings)
    except PyAMLException as exc:
        raise PyAMLException(f"{source}: {exc.message}") from exc

    data = OrbitResponseMatrixData(
        matrix=matrix.tolist(),
        observable_names=observable_names,
        variable_names=variable_names,
        rf_response=None,
        variable_planes=variable_planes,
        observable_planes=observable_planes,
        validate=False,
    )
    return _RMState(response_matrix=data, layout=layout, matrix=matrix, live=live)


# ---------------------------------------------------------------------------
# Pseudo-inverse
# ---------------------------------------------------------------------------

_EPS = float(np.finfo(float).eps)


def _pseudo_inverse(state: _RMState, settings: _Settings) -> PseudoInverse:
    """
    Compute the guarded, damped pseudo-inverse of a loaded response matrix.

    A pure function of ``state`` and ``settings``. The correctors with no response
    (zero columns) are removed before the SVD and re-embedded as exact-zero rows,
    so they are a structural fact at every ``alpha``. The spectrum has the
    reference implementation's length ``L = min(2*n_bpm, n_cor)``: the ``k_live``
    live singular values in descending order, then ``n_struct = L - k_live``
    structural zeros. The weights are formed in the reference order: ``s / (s**2 +
    alpha**2)`` for ``alpha > 0``, else ``1 / s`` under the rank guard ``s <=
    max(2*n_bpm, n_cor) * eps * s_max``; then times ``gain``; then the structural
    zeros are appended and the last ``n_discard`` positions of the full-length
    vector are zeroed.

    Raises
    ------
    PyAMLException
        When, after the cut, a weight is not finite, or every weight is zero while
        ``n_discard < L`` (``n_discard == L`` is the reference's own all-zero
        diagonal and returns the all-zero inverse).
    """
    layout = state.layout
    alpha, n_discard, gain = settings.alpha, settings.n_discard, settings.gain
    L, k_live, n_struct = layout.L, layout.k_live, layout.n_struct

    live_columns = np.flatnonzero(state.live)
    u, s, vt = np.linalg.svd(state.matrix[:, live_columns], full_matrices=False)
    s = s[:k_live]
    u, vt = u[:, :k_live], vt[:k_live, :]

    rank_dropped = np.zeros(k_live, dtype=bool)
    if alpha > 0.0:
        weights = s / (s * s + alpha * alpha)
    else:
        if k_live:
            rank_dropped = s <= max(2 * layout.n_bpm, layout.n_cor) * _EPS * s[0]
        weights = np.zeros(k_live)
        with np.errstate(divide="ignore", over="ignore"):
            weights[~rank_dropped] = 1.0 / s[~rank_dropped]
    rank_discarded = int(np.count_nonzero(rank_dropped))

    with np.errstate(over="ignore", invalid="ignore"):
        weights = weights * gain
    weights = np.concatenate([weights, np.zeros(n_struct)])
    cut = np.zeros(L, dtype=bool)
    if n_discard:
        cut[L - n_discard :] = True
    weights[cut] = 0.0

    if not np.all(np.isfinite(weights)):
        raise PyAMLException(f"the inverse weights are not finite for alpha={alpha!r}, n_discard={n_discard}, gain={gain!r}")
    if n_discard < L and not np.any(weights):
        raise PyAMLException(
            f"every inverse weight is zero for alpha={alpha!r}, n_discard={n_discard}, gain={gain!r} "
            f"(only n_discard == L = {L} discards the whole spectrum)"
        )

    kinds = []
    for i in range(L):
        if i >= k_live:
            kinds.append(SpectrumKind.STRUCTURAL)
        elif rank_dropped[i]:
            kinds.append(SpectrumKind.RANK_DISCARDED)
        elif cut[i]:
            kinds.append(SpectrumKind.DISCARDED)
        else:
            kinds.append(SpectrumKind.LIVE)

    matrix = np.zeros((layout.n_cor, 2 * layout.n_bpm))
    matrix[live_columns, :] = (vt.T * weights[:k_live]) @ u.T
    singular_values = np.concatenate([s, np.zeros(n_struct)])

    return PseudoInverse(
        matrix=matrix,
        singular_values=singular_values,
        inverse_weights=weights,
        spectrum_kind=kinds,
        rank_discarded=rank_discarded,
        no_response_correctors=np.flatnonzero(~state.live),
        no_response_names=layout.no_response_names,
        alpha=alpha,
        n_discard=n_discard,
        gain=gain,
    )


# ---------------------------------------------------------------------------
# Correction step
# ---------------------------------------------------------------------------

# Reasons under which a plane's step is zeroed and listed whole, whatever the threading.
_WHOLE_PLANE_H = frozenset({Reason.NO_BEAM, Reason.NO_INVERSE, Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_H})
_WHOLE_PLANE_V = frozenset({Reason.NO_BEAM, Reason.NO_INVERSE, Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_V})


def _float_array(value: Any, field: str, shape: tuple[int, ...], source: str) -> np.ndarray:
    """
    Return a float copy of ``value`` with exactly ``shape``.

    NaN is allowed (it carries meaning for readings, references and nothing else);
    ±Inf anywhere, a non-numeric entry or any other shape raises PyAMLException
    naming ``field``.
    """
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise PyAMLException(f"{source}: {field} must be an array of real numbers ({exc})") from exc
    if array.shape != shape:
        raise PyAMLException(f"{source}: {field} must have shape {shape} for the loaded response matrix, got {array.shape}")
    if np.any(np.isinf(array)):
        raise PyAMLException(f"{source}: {field} must not hold +Inf or -Inf")
    return array


def _index_array(value: Any, field: str, length: int, source: str) -> np.ndarray:
    """Return ``value`` as ``length`` finite, strictly increasing indices, raising PyAMLException otherwise."""
    array = _float_array(value, field, (length,), source)
    if np.any(np.isnan(array)):
        raise PyAMLException(f"{source}: {field} must not hold NaN")
    if np.any(np.diff(array) <= 0.0):
        raise PyAMLException(f"{source}: {field} must be strictly increasing")
    return array


def _bpm_index(value: Any, n_bpm: int, source: str) -> np.ndarray:
    """
    Return the first half of the sidecar's ``bpm_index``.

    The sidecar carries one entry per response-matrix row, ``2 * n_bpm`` in all;
    both halves must be equal and strictly increasing.
    """
    array = _float_array(value, "bpm_index", (2 * n_bpm,), source)
    if np.any(np.isnan(array)):
        raise PyAMLException(f"{source}: bpm_index must not hold NaN")
    first, second = array[:n_bpm], array[n_bpm:]
    if not np.array_equal(first, second):
        raise PyAMLException(f"{source}: both halves of bpm_index (one entry per response-matrix row) must be equal")
    if np.any(np.diff(first) <= 0.0):
        raise PyAMLException(f"{source}: bpm_index must be strictly increasing within each half")
    return first


def _correct(
    state: _RMState,
    pinv: PseudoInverse,
    x: np.ndarray,
    y: np.ndarray,
    reference: np.ndarray,
    bpm_index: np.ndarray,
    hcorr_index: np.ndarray,
    vcorr_index: np.ndarray,
) -> TrajectoryStep:
    """
    Solve one checked set of inputs and thread the result; a pure function.

    ``step = pinv.matrix @ dR`` with ``dR = R - R0`` and NaN set to 0. The last
    BPM with beam is the last non-NaN entry of ``x``; in each plane every
    corrector whose index exceeds that BPM's index is zeroed (the reference
    implementation's ``<=``). The reason is the first match of NO_BEAM,
    NO_INVERSE, NO_UPSTREAM, NO_UPSTREAM_H / NO_UPSTREAM_V, ON_REFERENCE, OK; under
    NO_BEAM, NO_INVERSE and NO_UPSTREAM both planes are zeroed whole, under
    NO_UPSTREAM_H / NO_UPSTREAM_V that plane is.
    """
    layout = state.layout
    n_h, n_v = layout.n_h, layout.n_v

    d_r = np.concatenate([x, y]) - reference.reshape(-1)
    d_r[np.isnan(d_r)] = 0.0
    step = pinv.matrix @ d_r

    with_beam = np.flatnonzero(~np.isnan(x))
    if with_beam.size:
        last_bpm: Optional[int] = int(with_beam[-1])
        last_index = bpm_index[last_bpm]
        past_h = hcorr_index > last_index
        past_v = vcorr_index > last_index
    else:
        last_bpm = None
        past_h = np.ones(n_h, dtype=bool)
        past_v = np.ones(n_v, dtype=bool)
    upstream_h = bool(np.any(state.live[:n_h] & ~past_h))
    upstream_v = bool(np.any(state.live[n_h:] & ~past_v))

    if last_bpm is None:
        reason = Reason.NO_BEAM
    elif not np.any(pinv.inverse_weights):
        reason = Reason.NO_INVERSE
    elif not upstream_h and not upstream_v:
        reason = Reason.NO_UPSTREAM
    elif not upstream_h:
        reason = Reason.NO_UPSTREAM_H
    elif not upstream_v:
        reason = Reason.NO_UPSTREAM_V
    elif not np.any(d_r):
        reason = Reason.ON_REFERENCE
    else:
        reason = Reason.OK

    zero_h = np.ones(n_h, dtype=bool) if reason in _WHOLE_PLANE_H else past_h
    zero_v = np.ones(n_v, dtype=bool) if reason in _WHOLE_PLANE_V else past_v
    step[np.concatenate([zero_h, zero_v])] = 0.0

    return TrajectoryStep(
        step=step,
        names=layout.hcorr_names + layout.vcorr_names,
        n_h=n_h,
        last_bpm=last_bpm,
        zeroed_h=tuple(name for name, zeroed in zip(layout.hcorr_names, zero_h, strict=True) if zeroed),
        zeroed_v=tuple(name for name, zeroed in zip(layout.vcorr_names, zero_v, strict=True) if zeroed),
        pseudo_inverse=pinv,
        reason=reason,
    )


def _read_file(path: Union[str, Path], settings: _Settings) -> _RMState:
    """Read and check the response-matrix file at ``path``; every failure is a PyAMLException naming it."""
    source = str(path)
    try:
        data = OrbitResponseMatrixData.load(source)
    except Exception as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise PyAMLException(f"{source}: cannot read the response matrix: {message}") from exc
    if not isinstance(data, OrbitResponseMatrixData):
        raise PyAMLException(f"{source}: the file holds a {type(data).__name__}, expected an OrbitResponseMatrixData")
    return _build_state(_rm_fields(data, source), source, settings)


@register_schema
class Trajectory(TuningTool, DynamicValidation):
    """
    First-turn trajectory correction from a trajectory response matrix.

    The tool is a response matrix plus default ``alpha``, ``n_discard`` and
    ``gain``. It reads no hardware and needs no peer. Every fact of the response
    matrix is checked once, when it is loaded, and published as :attr:`layout`.
    :meth:`load` is the only method that changes the tool, and it replaces the
    response matrix atomically; :meth:`pseudo_inverse` and :meth:`correct` are
    pure functions of the tool and their arguments, each returning one frozen
    record that carries the settings it used.

    Parameters
    ----------
    name : str
        Name of the tool.
    response_matrix : str, Path, mapping or OrbitResponseMatrixData
        Path to an :class:`~.OrbitResponseMatrixData` file, a mapping of its six
        fields, or an instance. Rows are BPM readings, every horizontal one then
        every vertical one, with equal counts; columns are correctors, every
        horizontal one then every vertical one. It carries no RF response.
    alpha : float
        Tikhonov parameter, a finite real ``>= 0``.
    n_discard : int
        Number of trailing singular-spectrum positions to zero, in ``[0, L]``.
    gain : float
        Gain, a finite real ``> 0``.

    Attributes
    ----------
    layout : TrajectoryLayout or None
        Facts of the loaded response matrix; None when none is loaded.
    response_matrix : OrbitResponseMatrixData or None
        A copy of the loaded response matrix; None when none is loaded.

    Methods
    -------
    load(load_path)
        Replace the response matrix with the one in a file.
    pseudo_inverse(alpha=None, n_discard=None, gain=None)
        The guarded, damped pseudo-inverse of the response matrix.
    correct(x, y, reference=None, *, bpm_index, hcorr_index, vcorr_index, ...)
        The corrector step for one set of first-turn BPM readings.

    Configuration
    -------------
    .. code-block:: yaml

        - type: pyaml.tuning_tools.trajectory
          name: TRAJECTORY
          response_matrix: ${path:trajectory_rm.json}
          alpha: 0.0
          n_discard: 0
          gain: 1.0

    Notes
    -----
    **The x-row rule.** Threading follows the reference implementation (the ALS
    ``calcCMstep`` routine): the last BPM with beam is the last non-NaN entry of
    ``x``, and ``y`` never decides it. In each plane every corrector whose lattice
    index exceeds that BPM's index is zeroed; a corrector at the BPM's own index
    is kept. An *upstream corrector* is a live corrector (one with a response) at
    or before that BPM.

    **The reason order.** A step's reason is the first match, in the fixed order
    NO_BEAM, NO_INVERSE, NO_UPSTREAM, NO_UPSTREAM_H / NO_UPSTREAM_V, ON_REFERENCE, OK,
    of: no BPM with beam (every entry of ``x`` NaN); every inverse weight zero
    (``n_discard == L``); no upstream corrector in either plane; none in that
    plane; the NaN-cleaned deviation identically zero; otherwise OK. Under NO_BEAM,
    NO_INVERSE and NO_UPSTREAM both planes are zeroed whole, under NO_UPSTREAM_H /
    NO_UPSTREAM_V that plane is, and otherwise exactly the correctors past the
    last BPM with beam; ``zeroed_h`` and ``zeroed_v`` list them for every reason.
    An all-zero step never carries OK; a non-OK reason does not imply an all-zero
    step (NO_UPSTREAM_H / NO_UPSTREAM_V correct the other plane), so a caller
    refuses to apply on ``all_zero``, never on the reason.

    **The four divergences.** The tool equals the reference implementation
    (``SCgetPinv`` and ``calcCMstep``) on every parity case of the frozen goldens
    and differs from it in exactly four frozen cases:

    (a) ``dead_last_plain``, rank-deficient at ``alpha == 0`` (an inverse
        divergence, reason OK): the reference applies no rank guard, so its
        inverse is not finite; the tool applies the guard, equals MATLAB
        ``pinv``, and gives correctors with no response an exact-zero step at
        every ``alpha``.
    (b) ``first_bpm_only``, beam only at the first BPM, before any corrector
        (verdict only): the tool answers NO_UPSTREAM with an all-zero step; the
        reference zeroes nothing and its step is roundoff, because the deviation
        lies outside the response matrix's range.
    (c) ``one_plane_upstream``, one plane without a live corrector at or before
        the last BPM with beam (verdict only in that plane): the tool zeroes that
        plane under NO_UPSTREAM_H / NO_UPSTREAM_V; the other plane is parity.
    (d) ``no_beam``, no BPM with beam (a step divergence): the tool answers
        NO_BEAM and zeroes every corrector; the reference falls back to the last
        BPM and zeroes no live corrector, so its vertical step is non-zero.

    ``n_discard == L`` (NO_INVERSE) is parity, not a divergence: the reference's
    diagonal is all zero there too.

    **The error rule.** A bad constructor argument or bad response-matrix content
    raises at construction, ``PyAMLConfigException`` from pyAML's validation layer
    (and from ``Factory.build``) and ``PyAMLException`` otherwise. A path that is
    not a file at construction logs a warning and leaves :attr:`layout` None, so
    every call raises ``PyAMLException``. A missing or bad file at :meth:`load`
    raises ``PyAMLException`` naming the path and leaves the tool unchanged. A
    bad per-call argument of :meth:`pseudo_inverse` or :meth:`correct` (a setting
    of the wrong type or out of range, a reading or index array whose shape does
    not match :attr:`layout`, +Inf or -Inf anywhere, a non-increasing index array,
    unequal ``bpm_index`` halves) raises ``PyAMLException`` naming the argument. A
    mapping holding any key other than the six ``OrbitResponseMatrixData``
    fields, a build key such as ``type`` included, is rejected.
    """

    def __init__(
        self,
        name: str,
        response_matrix: Union[str, Path, OrbitResponseMatrixData],
        alpha: float = 0.0,
        n_discard: int = 0,
        gain: float = 1.0,
    ):
        super().__init__(name)
        self._settings = _resolve_settings(alpha, n_discard, gain, L=None)
        self._state: Optional[_RMState] = None
        if isinstance(response_matrix, (str, Path)):
            if not Path(response_matrix).is_file():
                logger.warning(f"{name}: response matrix {response_matrix} is not a file; no response matrix is loaded")
                return
            self._state = _read_file(response_matrix, self._settings)
        else:
            source = f"{name}: response_matrix"
            self._state = _build_state(_rm_fields(response_matrix, source), source, self._settings)

    def load(self, load_path: Path) -> None:
        """
        Replace the response matrix with the one in a file.

        The new matrix is checked completely before it replaces the old one, so a
        failed load leaves the tool unchanged.

        Parameters
        ----------
        load_path : Path
            Path to an :class:`~.OrbitResponseMatrixData` file.

        Raises
        ------
        PyAMLException
            When the file is missing, cannot be read, or fails a check, a configured
            ``n_discard`` larger than the new ``L`` included. The message names the
            path.
        """
        self._state = _read_file(load_path, self._settings)

    def pseudo_inverse(
        self,
        alpha: Optional[float] = None,
        n_discard: Optional[int] = None,
        gain: Optional[float] = None,
    ) -> PseudoInverse:
        """
        Compute the guarded, damped pseudo-inverse of the response matrix.

        A pure function of the loaded response matrix, the configured settings and
        the overrides: nothing is stored on the tool, and the returned record is
        immutable. Correctors with no response are exact-zero rows of the inverse
        at every ``alpha``.

        Parameters
        ----------
        alpha : float, optional
            Tikhonov parameter, a finite real ``>= 0``; None means the configured
            value. At ``alpha == 0`` the singular values below the rank guard
            ``max(2*n_bpm, n_cor) * eps * s_max`` are dropped instead.
        n_discard : int, optional
            Number of trailing positions of the length-``L`` spectrum to zero, in
            ``[0, L]``; None means the configured value. The structural zeros are
            the last positions, so a discard consumes them before any live value.
        gain : float, optional
            Gain multiplying every weight, a finite real ``> 0``; None means the
            configured value.

        Returns
        -------
        PseudoInverse
            The ``n_cor x 2*n_bpm`` inverse with its spectrum, the kind of each
            position, the no-response correctors and the settings used.

        Raises
        ------
        PyAMLException
            When no response matrix is loaded; when a setting has the wrong type or
            is out of range; when a weight is not finite after the cut; or when
            every weight is zero while ``n_discard < L`` (``n_discard == L`` returns
            the all-zero inverse).
        """
        if self._state is None:
            raise PyAMLException(f"{self.name}: no response matrix is loaded")
        settings = _resolve_settings(alpha, n_discard, gain, L=self._state.layout.L, defaults=self._settings)
        return _pseudo_inverse(self._state, settings)

    def correct(
        self,
        x: Any,
        y: Any,
        reference: Any = None,
        *,
        bpm_index: Any,
        hcorr_index: Any,
        vcorr_index: Any,
        alpha: Optional[float] = None,
        n_discard: Optional[int] = None,
        gain: Optional[float] = None,
    ) -> TrajectoryStep:
        """
        Compute the corrector step for one set of first-turn BPM readings.

        A pure function of the loaded response matrix, the configured settings and
        the arguments: nothing is stored on the tool, the arguments are not
        modified, and the returned record is immutable. The solve is one joint
        full-matrix product, ``step = pseudo_inverse.matrix @ dR`` with ``dR = R -
        R0`` (all horizontal readings then all vertical ones) and NaN set to 0;
        the caller applies ``-step``.

        Parameters
        ----------
        x, y : array_like
            Horizontal and vertical readings of the ``n_bpm`` BPMs, in the row
            order of the response matrix; NaN means no beam at that BPM.
        reference : array_like or None
            ``(x_ref, y_ref)`` of shape ``(2, n_bpm)``; NaN means no reference at
            that BPM. None means a zero reference.
        bpm_index : array_like
            The sidecar's ``bpm_index``, verbatim: one entry per response-matrix
            row, ``2 * n_bpm`` in all, both halves equal and strictly increasing.
        hcorr_index, vcorr_index : array_like
            The lattice index of each horizontal and vertical corrector, in column
            order, strictly increasing, of lengths ``n_h`` and ``n_v``.
        alpha, n_discard, gain : optional
            Per-call overrides, as for :meth:`pseudo_inverse`; None means the
            configured value.

        Returns
        -------
        TrajectoryStep
            The step in column order with the corrector names, ``n_h``, the
            position of the last BPM with beam, the correctors zeroed per plane,
            the :class:`PseudoInverse` solved with and the reason.

        Raises
        ------
        PyAMLException
            When no response matrix is loaded; when a setting is rejected as in
            :meth:`pseudo_inverse`; when a reading, the reference or an index
            array has a shape that does not match :attr:`layout`, holds +Inf or
            -Inf (or NaN in an index array), is not strictly increasing, or when
            the two halves of ``bpm_index`` differ. The message names the
            argument.

        Notes
        -----
        Threading follows the reference implementation: the last BPM with beam is
        the last non-NaN entry of ``x`` (``y`` never decides it), and in each
        plane every corrector whose index exceeds that BPM's index is zeroed; a
        corrector at the BPM's own index is kept. An upstream corrector is a live
        corrector at or before that BPM. The reason is the first match of
        NO_BEAM, NO_INVERSE, NO_UPSTREAM, NO_UPSTREAM_H / NO_UPSTREAM_V,
        ON_REFERENCE, OK; see the class notes for the predicates and for where
        the result deliberately differs from the reference implementation.
        """
        if self._state is None:
            raise PyAMLException(f"{self.name}: no response matrix is loaded")
        state = self._state
        layout = state.layout
        settings = _resolve_settings(alpha, n_discard, gain, L=layout.L, defaults=self._settings)
        source = f"{self.name}: correct"
        n_bpm = layout.n_bpm
        x_values = _float_array(x, "x", (n_bpm,), source)
        y_values = _float_array(y, "y", (n_bpm,), source)
        if reference is None:
            reference_values = np.zeros((2, n_bpm))
        else:
            reference_values = _float_array(reference, "reference", (2, n_bpm), source)
        bpm = _bpm_index(bpm_index, n_bpm, source)
        hcorr = _index_array(hcorr_index, "hcorr_index", layout.n_h, source)
        vcorr = _index_array(vcorr_index, "vcorr_index", layout.n_v, source)
        pinv = _pseudo_inverse(state, settings)
        return _correct(state, pinv, x_values, y_values, reference_values, bpm, hcorr, vcorr)

    @property
    def layout(self) -> Optional[TrajectoryLayout]:
        """Facts of the loaded response matrix; None when none is loaded."""
        return None if self._state is None else self._state.layout

    @property
    def response_matrix(self) -> Optional[OrbitResponseMatrixData]:
        """A copy of the loaded response matrix; None when none is loaded."""
        return None if self._state is None else copy.deepcopy(self._state.response_matrix)
