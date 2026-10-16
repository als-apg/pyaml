"""Tests for :mod:`pyaml.tuning_tools.trajectory`.

Every test that needs the frozen goldens file carries ``golden`` in its name, so
``pytest -k "not golden"`` runs the rest without it. The module clears pyAML's
element registry itself, because it must also run without the repository's
``conftest.py``.
"""

import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pyaml.common.exception import PyAMLConfigException, PyAMLException
from pyaml.configuration.factory import Factory
from pyaml.tuning_tools.orbit_response_matrix_data import OrbitResponseMatrixData
from pyaml.tuning_tools.trajectory import (
    PseudoInverse,
    Reason,
    SpectrumKind,
    Trajectory,
    TrajectoryLayout,
    TrajectoryStep,
    _resolve_settings,
    _Settings,
)


@pytest.fixture(autouse=True)
def _clear_element_registry():
    Factory.clear()
    yield
    Factory.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _layout(n_bpm=4, n_h=3, n_v=3, dead=()):
    return TrajectoryLayout(
        bpm_names_h=[f"BPM{i}" for i in range(n_bpm)],
        bpm_names_v=[f"BPM{i}" for i in range(n_bpm)],
        hcorr_names=[f"HCM{i}" for i in range(n_h)],
        vcorr_names=[f"VCM{i}" for i in range(n_v)],
        no_response_names=list(dead),
    )


def _pinv(n_cor=6, n_obs=8, length=6, **overrides):
    kwargs = dict(
        matrix=np.arange(n_cor * n_obs, dtype=float).reshape(n_cor, n_obs),
        singular_values=np.linspace(3.0, 1.0, length),
        inverse_weights=1.0 / np.linspace(3.0, 1.0, length),
        spectrum_kind=[SpectrumKind.LIVE] * length,
        rank_discarded=0,
        no_response_correctors=[],
        no_response_names=[],
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )
    kwargs.update(overrides)
    return PseudoInverse(**kwargs)


def _step(step=(1.0, 0.0, -2.0, 0.5, 0.0, 0.0), reason=Reason.OK, n_h=3, **overrides):
    kwargs = dict(
        step=np.asarray(step, dtype=float),
        names=("HCM0", "HCM1", "HCM2", "VCM0", "VCM1", "VCM2"),
        n_h=n_h,
        last_bpm=3,
        zeroed_h=(),
        zeroed_v=(),
        pseudo_inverse=_pinv(),
        reason=reason,
    )
    kwargs.update(overrides)
    return TrajectoryStep(**kwargs)


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


def test_reason_members_are_the_closed_set():
    assert [m.name for m in Reason] == [
        "NO_BEAM",
        "NO_INVERSE",
        "NO_UPSTREAM",
        "NO_UPSTREAM_H",
        "NO_UPSTREAM_V",
        "ON_REFERENCE",
        "OK",
    ]


def test_spectrum_kind_members_are_the_closed_set():
    assert [m.name for m in SpectrumKind] == ["LIVE", "RANK_DISCARDED", "DISCARDED", "STRUCTURAL"]


@pytest.mark.parametrize("enum_cls", [Reason, SpectrumKind])
def test_enums_serialise_as_their_names(enum_cls):
    for member in enum_cls:
        assert member.name == member.value
        text = json.dumps(member)
        assert text == f'"{member.name}"'
        assert json.loads(text) == member.name == member.value
        assert enum_cls(json.loads(text)) is member
    assert json.loads(json.dumps(list(enum_cls))) == [m.name for m in enum_cls]


# ---------------------------------------------------------------------------
# TrajectoryLayout
# ---------------------------------------------------------------------------


def test_layout_stores_name_fields_as_tuples_of_str():
    layout = _layout(dead=["HCM2"])
    for field in ("bpm_names_h", "bpm_names_v", "hcorr_names", "vcorr_names", "no_response_names"):
        value = getattr(layout, field)
        assert isinstance(value, tuple)
        assert all(isinstance(name, str) for name in value)
    assert layout.no_response_names == ("HCM2",)


def test_layout_rejects_non_str_names():
    with pytest.raises(TypeError):
        TrajectoryLayout(
            bpm_names_h=["B0"],
            bpm_names_v=["B0"],
            hcorr_names=[1],
            vcorr_names=["V0"],
            no_response_names=[],
        )


def test_layout_counts_are_derived_from_names():
    layout = _layout(n_bpm=4, n_h=3, n_v=2, dead=["HCM2"])
    assert layout.n_bpm == 4
    assert layout.n_h == 3
    assert layout.n_v == 2
    assert layout.n_cor == 5
    assert layout.n_live == 4
    assert layout.L == min(2 * 4, 5) == 5
    assert layout.k_live == min(2 * 4, 4) == 4
    assert layout.n_struct == 1


@pytest.mark.parametrize(
    "n_bpm, n_h, n_v, dead",
    [
        (16, 12, 12, ()),
        (16, 12, 12, ("HCM11", "VCM11")),
        (2, 3, 3, ("HCM0",)),  # n_cor > 2*n_bpm: one dead column leaves no structural zero
        (2, 3, 3, ("HCM0", "HCM1", "VCM0")),
        (72, 72, 72, ()),
        (3, 1, 1, ("VCM0",)),
    ],
)
def test_layout_identities(n_bpm, n_h, n_v, dead):
    layout = _layout(n_bpm=n_bpm, n_h=n_h, n_v=n_v, dead=dead)
    assert layout.n_cor == n_h + n_v
    assert layout.n_live == layout.n_cor - len(dead)
    assert layout.L == min(2 * layout.n_bpm, layout.n_cor)
    assert layout.n_struct == layout.L - layout.k_live
    assert layout.k_live == min(2 * layout.n_bpm, layout.n_live)
    assert 0 <= layout.n_struct <= layout.L


def test_layout_n_cor_above_two_n_bpm_has_no_structural_zero():
    layout = _layout(n_bpm=2, n_h=3, n_v=3, dead=("HCM0",))
    assert layout.L == 4
    assert layout.k_live == 4
    assert layout.n_struct == 0


def test_layout_is_value_compared_and_hashable():
    a = _layout(dead=["HCM2"])
    b = _layout(dead=("HCM2",))
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_layout_with_a_different_dead_column_is_unequal():
    a = _layout(dead=["HCM2"])
    b = _layout(dead=["VCM0", "HCM2"])
    assert a.bpm_names_h == b.bpm_names_h
    assert a.hcorr_names == b.hcorr_names and a.vcorr_names == b.vcorr_names
    assert a != b
    assert a.n_live != b.n_live


def test_layout_with_different_names_is_unequal():
    assert _layout(n_bpm=4) != _layout(n_bpm=5)


def test_layout_is_frozen_and_slotted():
    layout = _layout()
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.hcorr_names = ("X",)
    assert not hasattr(layout, "__dict__")


# ---------------------------------------------------------------------------
# PseudoInverse
# ---------------------------------------------------------------------------


def test_pseudo_inverse_arrays_are_read_only():
    pinv = _pinv(no_response_correctors=[5], no_response_names=["VCM2"])
    for field in ("matrix", "singular_values", "inverse_weights", "no_response_correctors"):
        array = getattr(pinv, field)
        assert isinstance(array, np.ndarray)
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        pinv.matrix[0, 0] = 1.0
    with pytest.raises(ValueError):
        pinv.singular_values[0] = 1.0
    with pytest.raises(ValueError):
        pinv.inverse_weights[:] = 0.0
    with pytest.raises(ValueError):
        pinv.no_response_correctors[0] = 0


def test_pseudo_inverse_does_not_alias_its_inputs():
    matrix = np.ones((6, 8))
    weights = np.ones(6)
    pinv = _pinv(matrix=matrix, inverse_weights=weights)
    matrix[0, 0] = 99.0
    weights[0] = 99.0
    assert pinv.matrix[0, 0] == 1.0
    assert pinv.inverse_weights[0] == 1.0
    assert matrix.flags.writeable and weights.flags.writeable


def test_pseudo_inverse_fields_and_types():
    pinv = _pinv(
        spectrum_kind=["LIVE"] * 4 + [SpectrumKind.DISCARDED, "STRUCTURAL"],
        no_response_correctors=[5],
        no_response_names=["VCM2"],
        rank_discarded=0,
        alpha=0.5,
        n_discard=2,
        gain=0.7,
    )
    assert [f.name for f in dataclasses.fields(PseudoInverse)] == [
        "matrix",
        "singular_values",
        "inverse_weights",
        "spectrum_kind",
        "rank_discarded",
        "no_response_correctors",
        "no_response_names",
        "alpha",
        "n_discard",
        "gain",
    ]
    assert pinv.spectrum_kind == (SpectrumKind.LIVE,) * 4 + (SpectrumKind.DISCARDED, SpectrumKind.STRUCTURAL)
    assert all(isinstance(kind, SpectrumKind) for kind in pinv.spectrum_kind)
    assert json.loads(json.dumps(list(pinv.spectrum_kind)))[-1] == "STRUCTURAL"
    assert pinv.no_response_names == ("VCM2",)
    assert pinv.no_response_correctors.tolist() == [5]
    assert (pinv.alpha, pinv.n_discard, pinv.gain) == (0.5, 2, 0.7)


def test_pseudo_inverse_spectrum_arrays_are_index_aligned():
    with pytest.raises(ValueError):
        _pinv(inverse_weights=np.ones(5))
    with pytest.raises(ValueError):
        _pinv(spectrum_kind=[SpectrumKind.LIVE] * 5)


def test_pseudo_inverse_rejects_wrong_array_ranks():
    with pytest.raises(ValueError):
        _pinv(matrix=np.ones(6))
    with pytest.raises(ValueError):
        _pinv(singular_values=np.ones((6, 1)), inverse_weights=np.ones((6, 1)))


def test_pseudo_inverse_is_frozen_slotted_and_identity_compared():
    a = _pinv()
    b = _pinv()
    assert a == a
    assert a != b  # eq=False: identity, and no ambiguous ndarray truth value
    hash(a)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.gain = 2.0
    assert not hasattr(a, "__dict__")


# ---------------------------------------------------------------------------
# TrajectoryStep
# ---------------------------------------------------------------------------


def test_step_fields():
    assert [f.name for f in dataclasses.fields(TrajectoryStep)] == [
        "step",
        "names",
        "n_h",
        "last_bpm",
        "zeroed_h",
        "zeroed_v",
        "pseudo_inverse",
        "reason",
    ]


def test_step_array_is_read_only_and_not_aliased():
    raw = np.array([1.0, 0.0, -2.0, 0.5, 0.0, 0.0])
    step = _step(step=raw)
    raw[0] = 42.0
    assert step.step[0] == 1.0
    assert not step.step.flags.writeable
    with pytest.raises(ValueError):
        step.step[0] = 3.0
    with pytest.raises(ValueError):
        step.step_h[0] = 3.0
    with pytest.raises(ValueError):
        step.step_v[0] = 3.0


def test_step_plane_views():
    step = _step()
    np.testing.assert_array_equal(step.step_h, [1.0, 0.0, -2.0])
    np.testing.assert_array_equal(step.step_v, [0.5, 0.0, 0.0])
    assert step.names[: step.n_h] == ("HCM0", "HCM1", "HCM2")


def test_step_plane_views_follow_n_h():
    step = _step(n_h=2)
    np.testing.assert_array_equal(step.step_h, [1.0, 0.0])
    np.testing.assert_array_equal(step.step_v, [-2.0, 0.5, 0.0, 0.0])


def test_step_all_zero():
    assert not _step().all_zero
    assert _step(step=np.zeros(6), reason=Reason.NO_BEAM, last_bpm=None).all_zero
    assert _step(step=-np.zeros(6), reason=Reason.ON_REFERENCE).all_zero


def test_step_name_fields_are_tuples():
    step = _step(names=["HCM0", "HCM1", "HCM2", "VCM0", "VCM1", "VCM2"], zeroed_h=["HCM2"], zeroed_v=["VCM2"])
    assert step.names == ("HCM0", "HCM1", "HCM2", "VCM0", "VCM1", "VCM2")
    assert step.zeroed_h == ("HCM2",)
    assert step.zeroed_v == ("VCM2",)


def test_step_reason_is_coerced_to_the_enum():
    step = _step(reason="ON_REFERENCE", step=np.zeros(6))
    assert step.reason is Reason.ON_REFERENCE


def test_step_rejects_names_of_the_wrong_length():
    with pytest.raises(ValueError):
        _step(names=("HCM0", "HCM1"))


@pytest.mark.parametrize(
    "reason, status",
    [
        (Reason.NO_BEAM, "no BPM with beam"),
        (Reason.NO_INVERSE, "every singular value is discarded; no correction is possible"),
        (Reason.NO_UPSTREAM, "no corrector upstream of the last BPM with beam"),
        (Reason.NO_UPSTREAM_H, "no horizontal corrector upstream of the last BPM with beam"),
        (Reason.NO_UPSTREAM_V, "no vertical corrector upstream of the last BPM with beam"),
        (Reason.ON_REFERENCE, "already on reference"),
        (Reason.OK, ""),
    ],
)
def test_step_status_is_derived_from_reason(reason, status):
    assert _step(reason=reason).status == status


def test_step_is_frozen_slotted_and_identity_compared():
    a = _step()
    b = _step()
    assert a != b
    hash(a)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.reason = Reason.NO_BEAM
    assert not hasattr(a, "__dict__")


def test_step_serialises_reason_and_status():
    step = _step(reason=Reason.NO_UPSTREAM_V)
    payload = json.loads(json.dumps({"reason": step.reason, "status": step.status, "all_zero": step.all_zero}))
    assert payload == {
        "reason": "NO_UPSTREAM_V",
        "status": "no vertical corrector upstream of the last BPM with beam",
        "all_zero": False,
    }


# ---------------------------------------------------------------------------
# settings resolution
# ---------------------------------------------------------------------------
#
# ``_resolve_settings`` is the one routine every path goes through: construction,
# construction under ``validate=False``, and every per-call override. It is a pure
# function of its arguments and ``L``, so it is tested here before the class exists.

_DEFAULTS = _Settings(alpha=0.5, n_discard=2, gain=0.8)


def test_resolve_returns_python_scalars_of_the_documented_types():
    settings = _resolve_settings(0.25, 3, 0.5, L=6)
    assert settings == _Settings(alpha=0.25, n_discard=3, gain=0.5)
    assert type(settings.alpha) is float
    assert type(settings.n_discard) is int
    assert type(settings.gain) is float


def test_resolve_settings_record_is_frozen():
    settings = _resolve_settings(0.0, 0, 1.0, L=6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.alpha = 1.0


def test_resolve_accepts_ints_for_the_real_fields():
    settings = _resolve_settings(0, 0, 1, L=6)
    assert settings == _Settings(alpha=0.0, n_discard=0, gain=1.0)
    assert type(settings.alpha) is float and type(settings.gain) is float


def test_resolve_none_per_call_means_the_configured_value():
    assert _resolve_settings(None, None, None, L=6, defaults=_DEFAULTS) == _DEFAULTS
    assert _resolve_settings(0.1, None, None, L=6, defaults=_DEFAULTS) == _Settings(0.1, 2, 0.8)
    assert _resolve_settings(None, 5, None, L=6, defaults=_DEFAULTS) == _Settings(0.5, 5, 0.8)
    assert _resolve_settings(None, None, 2.0, L=6, defaults=_DEFAULTS) == _Settings(0.5, 2, 2.0)


@pytest.mark.parametrize("field", ["alpha", "n_discard", "gain"])
def test_resolve_none_without_defaults_raises_naming_the_field(field):
    # Construction has no configured value to fall back on.
    kwargs = {"alpha": 0.0, "n_discard": 0, "gain": 1.0, field: None}
    with pytest.raises(PyAMLException, match=field):
        _resolve_settings(**kwargs, L=6)


def test_resolve_rechecks_configured_defaults_against_l():
    # A configured n_discard that no longer fits L (load() shrank it) raises even
    # when the call passes None.
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(None, None, None, L=1, defaults=_DEFAULTS)


@pytest.mark.parametrize("alpha", [0.0, -0.0, 1e-12, 3.5, 1e150])
def test_resolve_alpha_in_range(alpha):
    resolved = _resolve_settings(alpha, 0, 1.0, L=6).alpha
    assert resolved == alpha
    assert math.copysign(1.0, resolved) == 1.0


@pytest.mark.parametrize("alpha", [-1e-300, -1.0, math.inf, -math.inf, math.nan, 1e200, 1e155], ids=lambda v: repr(v))
def test_resolve_alpha_out_of_range_raises_naming_alpha(alpha):
    with pytest.raises(PyAMLException, match="alpha"):
        _resolve_settings(alpha, 0, 1.0, L=6)


def test_resolve_alpha_whose_square_overflows_raises():
    with pytest.raises(PyAMLException, match="alpha"):
        _resolve_settings(1e200, 0, 1.0, L=6)


@pytest.mark.parametrize("n_discard", [0, 1, 5, 6])
def test_resolve_n_discard_in_range(n_discard):
    assert _resolve_settings(0.0, n_discard, 1.0, L=6).n_discard == n_discard


@pytest.mark.parametrize("n_discard", [-1, 7, 100])
def test_resolve_n_discard_out_of_range_raises_naming_n_discard(n_discard):
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(0.0, n_discard, 1.0, L=6)


def test_resolve_n_discard_equal_to_l_is_allowed_and_l_plus_one_is_not():
    assert _resolve_settings(0.0, 4, 1.0, L=4).n_discard == 4
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(0.0, 5, 1.0, L=4)


def test_resolve_without_l_checks_only_the_lower_bound():
    # L is None while no response matrix is loaded; load() re-checks the bound.
    assert _resolve_settings(0.0, 1000, 1.0, L=None).n_discard == 1000
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(0.0, -1, 1.0, L=None)


@pytest.mark.parametrize("gain", [1e-300, 0.5, 1.0, 10.0])
def test_resolve_gain_in_range(gain):
    assert _resolve_settings(0.0, 0, gain, L=6).gain == gain


@pytest.mark.parametrize("gain", [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan], ids=lambda v: repr(v))
def test_resolve_gain_out_of_range_raises_naming_gain(gain):
    with pytest.raises(PyAMLException, match="gain"):
        _resolve_settings(0.0, 0, gain, L=6)


@pytest.mark.parametrize(
    "field, value",
    [
        ("alpha", "0.1"),
        ("alpha", True),
        ("alpha", 1 + 0j),
        ("alpha", [0.1]),
        ("alpha", np.array([0.1])),
        ("n_discard", 1.0),
        ("n_discard", 1.5),
        ("n_discard", "1"),
        ("n_discard", True),
        ("n_discard", np.bool_(True)),
        ("n_discard", np.float64(1.0)),
        ("gain", "1"),
        ("gain", False),
        ("gain", np.bool_(True)),
        ("gain", object()),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, str) else repr(v),
)
def test_resolve_wrong_type_raises_naming_the_field(field, value):
    kwargs = {"alpha": 0.0, "n_discard": 0, "gain": 1.0, field: value}
    with pytest.raises(PyAMLException, match=field):
        _resolve_settings(**kwargs, L=6)


def test_resolve_wrong_type_is_also_rejected_as_a_per_call_override():
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(None, 1.0, None, L=6, defaults=_DEFAULTS)


def test_resolve_accepts_numpy_scalars_and_returns_python_scalars():
    settings = _resolve_settings(np.float64(0.25), np.int64(3), np.float32(0.5), L=6)
    assert settings == _Settings(alpha=0.25, n_discard=3, gain=0.5)
    assert type(settings.alpha) is float
    assert type(settings.n_discard) is int
    assert type(settings.gain) is float


def test_resolve_accepts_zero_d_arrays_as_read_from_an_npz():
    # np.load(...)[key] of a stored scalar is a 0-d ndarray, not a numpy scalar.
    settings = _resolve_settings(np.array(0.25), np.array(3, dtype=np.int64), np.array(2.0), L=6)
    assert settings == _Settings(alpha=0.25, n_discard=3, gain=2.0)
    assert type(settings.n_discard) is int


def test_resolve_numpy_scalars_are_range_checked():
    with pytest.raises(PyAMLException, match="n_discard"):
        _resolve_settings(0.0, np.int64(7), 1.0, L=6)
    with pytest.raises(PyAMLException, match="alpha"):
        _resolve_settings(np.float64(np.nan), 0, 1.0, L=6)
    with pytest.raises(PyAMLException, match="gain"):
        _resolve_settings(0.0, 0, np.array(0.0), L=6)


def test_resolve_is_pure():
    defaults = _Settings(alpha=0.5, n_discard=2, gain=0.8)
    alpha = np.array(0.25)
    first = _resolve_settings(alpha, None, None, L=6, defaults=defaults)
    second = _resolve_settings(alpha, None, None, L=6, defaults=defaults)
    assert first == second
    assert defaults == _Settings(alpha=0.5, n_discard=2, gain=0.8)
    assert alpha == 0.25


# ---------------------------------------------------------------------------
# construction, the RM entry point and load()
#
# The tool is pure: it reads no hardware and needs no holder or peer, so these
# tests construct it directly (and through Factory.build from a config dict).
# Every response-matrix file they read is written under tmp_path by _rm_fields
# and _write_rm below; the module reads no other file but the case table and the
# goldens npz. Every Factory.build call uses a fresh element name, and every
# Factory-route assertion matches a message naming the rejected field or the RM
# path, so a duplicate-name rejection can never satisfy it.
# ---------------------------------------------------------------------------

_TOOL_TYPE = "pyaml.tuning_tools.trajectory"


def _rm_fields(n_bpm=4, n_h=3, n_v=3, dead=(), scale=1.0):
    """The six OrbitResponseMatrixData fields of a small causal RM; ``dead`` names zero columns."""
    hcorr = [f"HCM{i}" for i in range(n_h)]
    vcorr = [f"VCM{i}" for i in range(n_v)]
    rng = np.random.default_rng(7)
    matrix = scale * rng.uniform(0.5, 1.5, size=(2 * n_bpm, n_h + n_v))
    for j, name in enumerate(hcorr + vcorr):
        if name in dead:
            matrix[:, j] = 0.0
    return {
        "matrix": matrix.tolist(),
        "observable_names": [f"BPM{i}:x" for i in range(n_bpm)] + [f"BPM{i}:y" for i in range(n_bpm)],
        "variable_names": hcorr + vcorr,
        "rf_response": None,
        "variable_planes": ["H"] * n_h + ["V"] * n_v,
        "observable_planes": ["H"] * n_bpm + ["V"] * n_bpm,
    }


def _write_rm(path, fields=None, type_key="pyaml.tuning_tools.orbit_response_matrix_data", **kwargs):
    body = dict(_rm_fields(**kwargs) if fields is None else fields)
    if body.get("rf_response") is None:
        body.pop("rf_response", None)
    if type_key is not None:
        body["type"] = type_key
    path.write_text(json.dumps(body))
    return path


def _expected_layout(n_bpm=4, n_h=3, n_v=3, dead=()):
    return TrajectoryLayout(
        bpm_names_h=[f"BPM{i}:x" for i in range(n_bpm)],
        bpm_names_v=[f"BPM{i}:y" for i in range(n_bpm)],
        hcorr_names=[f"HCM{i}" for i in range(n_h)],
        vcorr_names=[f"VCM{i}" for i in range(n_v)],
        no_response_names=list(dead),
    )


def _config(name, response_matrix, **settings):
    return {"type": _TOOL_TYPE, "name": name, "response_matrix": response_matrix, **settings}


def _snapshot(tool):
    rm = tool.response_matrix
    return tool.layout, None if rm is None else json.dumps(dataclasses.asdict(rm))


def test_trajectory_construction_forms_give_the_same_layout(tmp_path):
    path = _write_rm(tmp_path / "rm.json", dead=("VCM2",))
    fields = _rm_fields(dead=("VCM2",))
    instance = OrbitResponseMatrixData(**fields)
    tools = [
        Trajectory("from_str", str(path)),
        Trajectory("from_path", path),
        Trajectory("from_mapping", fields),
        Trajectory("from_validated_instance", instance),
        Trajectory("from_instance", instance, validate=False),
        Factory.build(_config("from_factory", str(path))),
    ]
    expected = _expected_layout(dead=("VCM2",))
    for tool in tools:
        assert isinstance(tool, Trajectory)
        assert tool.layout == expected
        assert tool.response_matrix.matrix == fields["matrix"]
        assert tool.response_matrix.variable_names == fields["variable_names"]
        assert tool.response_matrix.observable_names == fields["observable_names"]
    assert expected.n_live == 5 and expected.L == 6 and expected.k_live == 5 and expected.n_struct == 1


def test_trajectory_factory_build_registers_the_tool_under_its_name(tmp_path):
    path = _write_rm(tmp_path / "rm.json")
    tool = Factory.build(_config("registered_tool", str(path)))
    assert tool.get_name() == "registered_tool"
    with pytest.raises(PyAMLConfigException, match="registered_tool"):
        Factory.build(_config("registered_tool", str(path)))


def test_trajectory_factory_build_passes_the_settings(tmp_path):
    path = _write_rm(tmp_path / "rm.json")
    tool = Factory.build(_config("factory_settings", str(path), alpha=0.5, n_discard=2, gain=0.7))
    assert tool._settings == _Settings(alpha=0.5, n_discard=2, gain=0.7)


def test_trajectory_defaults_are_alpha_zero_no_discard_unit_gain(tmp_path):
    tool = Trajectory("defaults", _write_rm(tmp_path / "rm.json"))
    assert tool._settings == _Settings(alpha=0.0, n_discard=0, gain=1.0)


def test_trajectory_mapping_with_a_build_key_is_rejected():
    # The implementer's choice, pinned here: a mapping may hold only the six
    # OrbitResponseMatrixData fields. On the validated path pydantic rejects the
    # extra key; under validate=False the tool's own check rejects it, naming it.
    fields = dict(_rm_fields(), type="pyaml.tuning_tools.orbit_response_matrix_data")
    with pytest.raises(PyAMLConfigException, match="response_matrix"):
        Trajectory("build_key_validated", fields)
    with pytest.raises(PyAMLException, match="type"):
        Trajectory("build_key_unvalidated", fields, validate=False)


def test_trajectory_mapping_may_omit_the_optional_none_field():
    fields = _rm_fields()
    del fields["rf_response"]
    assert Trajectory("no_rf_key", fields, validate=False).layout == _expected_layout()


def test_trajectory_rejects_an_rm_argument_of_another_type():
    with pytest.raises(PyAMLException, match="response_matrix"):
        Trajectory("bad_rm_type", 3, validate=False)


def _copy_fields(fields):
    return json.loads(json.dumps(fields))


def test_trajectory_does_not_alias_the_instance_or_mapping_it_was_built_from():
    fields = _rm_fields()
    instance = OrbitResponseMatrixData(**_copy_fields(fields), validate=False)
    from_instance = Trajectory("alias_instance", instance, validate=False)
    mapping = _copy_fields(fields)
    from_mapping = Trajectory("alias_mapping", mapping, validate=False)
    before = [_snapshot(from_instance), _snapshot(from_mapping)]

    for target in (instance.matrix, mapping["matrix"]):
        target[0][0] += 1.0
        target.append([0.0] * 6)
    for names in (instance.variable_names, mapping["variable_names"]):
        names[0] = "CHANGED"

    assert [_snapshot(from_instance), _snapshot(from_mapping)] == before
    assert from_instance.layout == _expected_layout()


def test_trajectory_response_matrix_is_a_copy(tmp_path):
    tool = Trajectory("rm_copy", _write_rm(tmp_path / "rm.json"))
    first = tool.response_matrix
    original = first.matrix[0][0]
    first.matrix[0][0] += 1
    first.variable_names[0] = "CHANGED"
    second = tool.response_matrix
    assert second is not first
    assert second.matrix[0][0] == original
    assert second.variable_names[0] == "HCM0"


@pytest.mark.parametrize("key", ["bpm_array_name", "hcorr_array_name", "vcorr_array_name"])
def test_trajectory_config_with_an_array_name_is_rejected(tmp_path, key):
    path = _write_rm(tmp_path / "rm.json")
    with pytest.raises(PyAMLConfigException, match=key):
        Factory.build(_config(f"array_name_{key}", str(path), **{key: "BPM"}))


@pytest.mark.parametrize("field", ["alpha", "n_discard", "gain"])
def test_trajectory_none_setting_at_construction_raises_config_exception(tmp_path, field):
    path = _write_rm(tmp_path / "rm.json")
    with pytest.raises(PyAMLConfigException, match=field):
        Trajectory(f"none_{field}", str(path), **{field: None})


_RANGE_REJECTIONS = [
    ("alpha", -1.0),
    ("alpha", 1e200),
    ("alpha", math.inf),
    ("alpha", math.nan),
    ("n_discard", -1),
    ("n_discard", 7),
    ("gain", 0.0),
    ("gain", -1.0),
    ("gain", math.inf),
    ("gain", math.nan),
]


@pytest.mark.parametrize("field,value", _RANGE_REJECTIONS, ids=lambda v: repr(v))
@pytest.mark.parametrize("validate", [True, False])
def test_trajectory_range_rejection_at_direct_construction(tmp_path, field, value, validate):
    path = _write_rm(tmp_path / "rm.json")  # L = 6
    with pytest.raises(PyAMLException, match=field):
        Trajectory(f"range_{field}", str(path), validate=validate, **{field: value})


@pytest.mark.parametrize("field,value", _RANGE_REJECTIONS, ids=lambda v: repr(v))
def test_trajectory_range_rejection_through_factory_build(tmp_path, field, value):
    path = _write_rm(tmp_path / "rm.json")
    with pytest.raises(PyAMLConfigException, match=field):
        Factory.build(_config(f"factory_range_{field}_{value!r}", str(path), **{field: value}))


def test_trajectory_range_rejection_without_a_file_still_checks_the_lower_bounds(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(PyAMLException, match="n_discard"):
        Trajectory("lower_bound", missing, n_discard=-1)
    # The upper bound needs L, which a missing file does not give.
    assert Trajectory("upper_bound_unknown", missing, n_discard=100).layout is None


def test_trajectory_n_discard_up_to_l_is_accepted(tmp_path):
    tool = Trajectory("n_discard_l", _write_rm(tmp_path / "rm.json"), n_discard=6)
    assert tool.layout.L == 6


def test_trajectory_n_discard_above_l_raises_at_a_load_that_shrinks_l(tmp_path):
    tool = Trajectory("shrink", _write_rm(tmp_path / "big.json"), n_discard=6)
    small = _write_rm(tmp_path / "small.json", n_bpm=2, n_h=1, n_v=1)  # L = 2
    before = _snapshot(tool)
    with pytest.raises(PyAMLException, match="n_discard") as info:
        tool.load(small)
    assert str(small) in info.value.message
    assert _snapshot(tool) == before


def _interleaved(fields):
    n = len(fields["observable_planes"]) // 2
    fields["observable_planes"] = ["H", "V"] * n


def _unequal_rows(fields):
    n = len(fields["observable_planes"])
    fields["observable_planes"] = ["H"] * (n // 2 + 1) + ["V"] * (n // 2 - 1)


def _ragged(fields):
    fields["matrix"][1] = fields["matrix"][1][:-1]


def _non_finite(fields):
    fields["matrix"][2][3] = math.nan


def _shape_mismatch(fields):
    fields["variable_names"] = fields["variable_names"] + ["HCM_extra"]


def _rf_response(fields):
    fields["rf_response"] = [0.0] * len(fields["observable_names"])


def _no_response(fields):
    fields["matrix"] = [[0.0] * len(row) for row in fields["matrix"]]


def _duplicated_row_name(fields):
    fields["observable_names"][1] = fields["observable_names"][0]


def _duplicated_column_name(fields):
    fields["variable_names"][4] = fields["variable_names"][3]


def _missing_planes(fields):
    del fields["variable_planes"]


def _write_malformed(path, case):
    if case == "response_matrix_data_type":
        fields = _rm_fields()
        body = {k: fields[k] for k in ("matrix", "observable_names", "variable_names")}
        return _write_rm(path, body, type_key="pyaml.tuning_tools.response_matrix_data")
    if case == "type_3":
        return _write_rm(path, type_key=3)
    if case == "class_path_x":
        body = dict(_rm_fields(), class_path="x")
        return _write_rm(path, body, type_key=None)
    fields = _rm_fields()
    _MUTATIONS[case](fields)
    return _write_rm(path, fields)


_MUTATIONS = {
    "interleaved_planes": _interleaved,
    "unequal_row_counts": _unequal_rows,
    "ragged_matrix": _ragged,
    "non_finite_matrix": _non_finite,
    "shape_mismatched_to_names": _shape_mismatch,
    "rf_response_present": _rf_response,
    "no_responding_corrector": _no_response,
    "duplicated_row_name": _duplicated_row_name,
    "duplicated_column_name": _duplicated_column_name,
    "planes_missing": _missing_planes,
}

_MALFORMED = sorted(_MUTATIONS) + ["response_matrix_data_type", "type_3", "class_path_x"]

# The check each malformed file must trip, so no case passes on another's failure.
_REJECTED_BY = {
    "interleaved_planes": "plane-major",
    "unequal_row_counts": "equal row counts",
    "ragged_matrix": "rectangular",
    "non_finite_matrix": "finite",
    "shape_mismatched_to_names": "variable_names",
    "rf_response_present": "rf_response",
    "no_responding_corrector": "no corrector has a response",
    "duplicated_row_name": "H row half",
    "duplicated_column_name": "unique within the variable_names",
    "planes_missing": "variable_planes is missing",
    "response_matrix_data_type": "ResponseMatrixData",
    "type_3": "cannot read",
    "class_path_x": "cannot read",
}


@pytest.mark.parametrize("case", _MALFORMED)
def test_trajectory_malformed_rm_raises_at_construction_and_load(tmp_path, case):
    bad = _write_malformed(tmp_path / f"{case}.json", case)
    with pytest.raises(PyAMLException, match=_REJECTED_BY[case]) as info:
        Trajectory(f"malformed_{case}", bad)
    assert str(bad) in info.value.message
    with pytest.raises(PyAMLException, match=_REJECTED_BY[case]):
        Trajectory(f"malformed_{case}_unvalidated", str(bad), validate=False)

    tool = Trajectory(f"good_{case}", _write_rm(tmp_path / "good.json"))
    before = _snapshot(tool)
    with pytest.raises(PyAMLException, match=_REJECTED_BY[case]) as info:
        tool.load(bad)
    assert str(bad) in info.value.message
    assert _snapshot(tool) == before


def test_trajectory_malformed_rm_through_factory_build_names_the_path(tmp_path):
    bad = _write_malformed(tmp_path / "dup.json", "duplicated_column_name")
    with pytest.raises(PyAMLConfigException, match="dup.json"):
        Factory.build(_config("factory_malformed", str(bad)))


@pytest.mark.parametrize("case", sorted(_MUTATIONS))
def test_trajectory_malformed_mapping_raises_at_construction(case):
    fields = _rm_fields()
    _MUTATIONS[case](fields)
    with pytest.raises(PyAMLException, match=_REJECTED_BY[case]):
        Trajectory(f"malformed_mapping_{case}", fields, validate=False)


def test_trajectory_duplicate_names_across_row_halves_are_allowed():
    fields = _rm_fields()
    fields["observable_names"] = [f"BPM{i}" for i in range(4)] * 2
    tool = Trajectory("shared_bpm_names", fields)
    assert tool.layout.bpm_names_h == tool.layout.bpm_names_v


def test_trajectory_missing_path_at_construction_warns_and_leaves_no_rm(tmp_path, caplog):
    missing = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING, logger="pyaml.tuning_tools.trajectory"):
        tool = Trajectory("missing", str(missing))
    assert tool.layout is None
    assert tool.response_matrix is None
    assert any(str(missing) in record.getMessage() for record in caplog.records)


def test_trajectory_directory_path_at_construction_is_not_a_file(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="pyaml.tuning_tools.trajectory"):
        tool = Trajectory("directory", tmp_path)
    assert tool.layout is None
    assert any(str(tmp_path) in record.getMessage() for record in caplog.records)


def test_trajectory_missing_path_through_factory_build_warns(tmp_path, caplog):
    missing = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING, logger="pyaml.tuning_tools.trajectory"):
        tool = Factory.build(_config("factory_missing", str(missing)))
    assert tool.layout is None
    assert any(str(missing) in record.getMessage() for record in caplog.records)


def test_trajectory_load_of_a_missing_path_raises_without_warning(tmp_path, caplog):
    tool = Trajectory("load_missing", _write_rm(tmp_path / "rm.json"))
    before = _snapshot(tool)
    missing = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING, logger="pyaml.tuning_tools.trajectory"):
        with pytest.raises(PyAMLException) as info:
            tool.load(missing)
    assert str(missing) in info.value.message
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert _snapshot(tool) == before


def test_trajectory_load_of_a_missing_path_on_a_tool_without_rm_changes_nothing(tmp_path):
    tool = Trajectory("load_missing_empty", tmp_path / "absent.json")
    with pytest.raises(PyAMLException):
        tool.load(tmp_path / "still_absent.json")
    assert tool.layout is None and tool.response_matrix is None


def test_trajectory_second_load_changes_layout_and_rm_together(tmp_path):
    tool = Trajectory("reload", _write_rm(tmp_path / "first.json"))
    first_layout, first_rm = tool.layout, tool.response_matrix
    second = _write_rm(tmp_path / "second.json", n_bpm=5, n_h=2, n_v=4, dead=("HCM1",), scale=2.0)
    tool.load(second)
    assert tool.layout == _expected_layout(n_bpm=5, n_h=2, n_v=4, dead=("HCM1",))
    assert tool.layout != first_layout
    rm = tool.response_matrix
    assert rm.matrix == _rm_fields(n_bpm=5, n_h=2, n_v=4, dead=("HCM1",), scale=2.0)["matrix"]
    assert rm.matrix != first_rm.matrix
    assert len(rm.matrix) == 2 * tool.layout.n_bpm
    assert len(rm.variable_names) == tool.layout.n_cor


def test_trajectory_load_fills_a_tool_built_without_rm(tmp_path):
    tool = Trajectory("late_load", tmp_path / "absent.json", n_discard=6)
    tool.load(_write_rm(tmp_path / "rm.json"))
    assert tool.layout == _expected_layout()


def test_trajectory_load_accepts_a_str_path(tmp_path):
    tool = Trajectory("load_str", _write_rm(tmp_path / "rm.json"))
    tool.load(str(_write_rm(tmp_path / "other.json", n_bpm=3)))
    assert tool.layout.n_bpm == 3


def test_trajectory_attached_copy_shares_no_writable_state(tmp_path):
    tool = Trajectory("attach_src", _write_rm(tmp_path / "rm.json"))
    clone = copy.copy(tool)
    clone.load(_write_rm(tmp_path / "other.json", n_bpm=3))
    assert tool.layout == _expected_layout()
    assert clone.layout.n_bpm == 3


# ---------------------------------------------------------------------------
# pseudo_inverse()
#
# Designed matrices: _designed_fields gives column c one non-zero entry, at row c,
# so the singular values are the column magnitudes and the inverse is known by
# hand (a zero magnitude is a dead column). Random RMs from _rm_fields serve the
# count identities. Nothing here reads the goldens.
# ---------------------------------------------------------------------------

_EPS = np.finfo(float).eps


def _designed_fields(h_values, v_values, n_bpm):
    """Fields of a 2*n_bpm x (n_h + n_v) RM whose column c has one entry at row c."""
    values = list(h_values) + list(v_values)
    assert len(values) <= 2 * n_bpm
    matrix = np.zeros((2 * n_bpm, len(values)))
    for c, value in enumerate(values):
        matrix[c, c] = value
    fields = _rm_fields(n_bpm=n_bpm, n_h=len(h_values), n_v=len(v_values))
    fields["matrix"] = matrix.tolist()
    return fields


def _designed(name, h_values, v_values, n_bpm=2, **settings):
    return Trajectory(name, _designed_fields(h_values, v_values, n_bpm), **settings)


def _counts(pinv):
    return {kind: pinv.spectrum_kind.count(kind) for kind in SpectrumKind}


def _assert_count_identities(pinv, layout, n_discard):
    counts = _counts(pinv)
    assert len(pinv.singular_values) == len(pinv.inverse_weights) == len(pinv.spectrum_kind) == layout.L
    assert counts[SpectrumKind.STRUCTURAL] == layout.n_struct
    assert counts[SpectrumKind.RANK_DISCARDED] == pinv.rank_discarded
    assert counts[SpectrumKind.DISCARDED] == max(n_discard - layout.n_struct - pinv.rank_discarded, 0)
    assert counts[SpectrumKind.LIVE] == layout.L - sum(v for k, v in counts.items() if k is not SpectrumKind.LIVE)
    # Structural zeros come last, then rank-discarded and discarded positions precede them.
    assert pinv.spectrum_kind[layout.k_live :] == (SpectrumKind.STRUCTURAL,) * layout.n_struct
    live = np.array([kind is SpectrumKind.LIVE for kind in pinv.spectrum_kind])
    assert np.all(pinv.inverse_weights[~live] == 0.0)
    assert np.all(pinv.inverse_weights[live] != 0.0)
    assert np.all(np.diff(pinv.singular_values) <= 0.0)
    assert np.all(pinv.singular_values[layout.k_live :] == 0.0)


def _full_snapshot(tool):
    state = tool._state
    if state is None:
        return None
    return (state.layout, np.copy(state.matrix), np.copy(state.live), json.dumps(dataclasses.asdict(state.response_matrix)))


def _assert_same_snapshot(before, after):
    assert (before is None) == (after is None)
    if before is None:
        return
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1]) and before[1].tobytes() == after[1].tobytes()
    assert np.array_equal(before[2], after[2])
    assert before[3] == after[3]


def test_pseudo_inverse_without_rm_raises(tmp_path):
    tool = Trajectory("pinv_no_rm", tmp_path / "absent.json")
    with pytest.raises(PyAMLException, match="response matrix"):
        tool.pseudo_inverse()


def test_pseudo_inverse_construction_forms_give_the_same_matrix(tmp_path):
    path = _write_rm(tmp_path / "rm.json", dead=("VCM2",))
    fields = _rm_fields(dead=("VCM2",))
    tools = [
        Trajectory("pinv_form_str", str(path)),
        Trajectory("pinv_form_path", path),
        Trajectory("pinv_form_mapping", fields),
        Trajectory("pinv_form_instance", OrbitResponseMatrixData(**fields, validate=False), validate=False),
        Factory.build(_config("pinv_form_factory", str(path))),
    ]
    reference = tools[0].pseudo_inverse()
    for tool in tools[1:]:
        pinv = tool.pseudo_inverse()
        assert np.array_equal(pinv.matrix, reference.matrix)
        assert np.array_equal(pinv.singular_values, reference.singular_values)
        assert np.array_equal(pinv.inverse_weights, reference.inverse_weights)
        assert pinv.spectrum_kind == reference.spectrum_kind


def test_pseudo_inverse_returns_the_record_type_with_read_only_arrays():
    pinv = _designed("pinv_record", [4.0, 2.0], [1.0, 0.5]).pseudo_inverse()
    assert isinstance(pinv, PseudoInverse)
    for field in ("matrix", "singular_values", "inverse_weights", "no_response_correctors"):
        array = getattr(pinv, field)
        assert isinstance(array, np.ndarray)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[...] = 0
    assert pinv.matrix.shape == (4, 4)
    assert pinv.matrix.dtype == float


def test_pseudo_inverse_plain_inverse_of_a_designed_matrix():
    pinv = _designed("pinv_plain", [4.0, 2.0], [1.0, 0.5]).pseudo_inverse()
    assert np.allclose(pinv.singular_values, [4.0, 2.0, 1.0, 0.5], rtol=1e-13, atol=0.0)
    assert np.allclose(pinv.inverse_weights, [0.25, 0.5, 1.0, 2.0], rtol=1e-13, atol=0.0)
    assert pinv.spectrum_kind == (SpectrumKind.LIVE,) * 4
    assert pinv.rank_discarded == 0
    assert np.allclose(pinv.matrix, np.diag([0.25, 0.5, 1.0, 2.0]), rtol=1e-13, atol=1e-15)
    assert (pinv.alpha, pinv.n_discard, pinv.gain) == (0.0, 0, 1.0)


def test_pseudo_inverse_n_discard_zero_equals_the_plain_inverse(tmp_path):
    tool = Trajectory("pinv_n0", _write_rm(tmp_path / "rm.json"))
    matrix = np.array(tool.response_matrix.matrix)
    pinv = tool.pseudo_inverse(n_discard=0)
    expected = np.linalg.pinv(matrix, rcond=max(matrix.shape) * _EPS)
    assert np.allclose(pinv.matrix, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())
    assert np.allclose(pinv.matrix @ matrix, np.eye(6), atol=1e-10)


def test_pseudo_inverse_tikhonov_weights_in_matlab_order():
    alpha = 1.5
    pinv = _designed("pinv_tikhonov", [4.0, 2.0], [1.0, 0.5], alpha=alpha).pseudo_inverse()
    s = np.array([4.0, 2.0, 1.0, 0.5])
    expected = s / (s * s + alpha * alpha)
    assert np.allclose(pinv.inverse_weights, expected, rtol=1e-13, atol=0.0)
    assert np.allclose(pinv.matrix, np.diag(expected), rtol=1e-13, atol=1e-15)
    assert pinv.spectrum_kind == (SpectrumKind.LIVE,) * 4
    assert pinv.alpha == alpha


def test_pseudo_inverse_gain_scales_the_weights_and_the_matrix():
    plain = _designed("pinv_gain_1", [4.0, 2.0], [1.0, 0.5]).pseudo_inverse()
    scaled = _designed("pinv_gain_2", [4.0, 2.0], [1.0, 0.5]).pseudo_inverse(gain=2.0)
    assert np.allclose(scaled.inverse_weights, 2.0 * plain.inverse_weights, rtol=1e-15, atol=0.0)
    assert np.allclose(scaled.matrix, 2.0 * plain.matrix, rtol=1e-15, atol=0.0)
    assert np.array_equal(scaled.singular_values, plain.singular_values)
    assert scaled.gain == 2.0 and plain.gain == 1.0


def test_pseudo_inverse_gain_multiplies_after_tikhonov():
    alpha, gain = 0.7, 3.0
    pinv = _designed("pinv_alpha_gain", [4.0, 2.0], [1.0, 0.5]).pseudo_inverse(alpha=alpha, gain=gain)
    s = np.array([4.0, 2.0, 1.0, 0.5])
    assert np.allclose(pinv.inverse_weights, gain * s / (s * s + alpha * alpha), rtol=1e-13, atol=0.0)


def test_pseudo_inverse_dead_column_is_a_structural_zero_and_an_exact_zero_row():
    # L = 4, n_live = 3: one structural zero at the end of the spectrum.
    tool = _designed("pinv_dead", [4.0, 2.0], [1.0, 0.0])
    assert tool.layout.no_response_names == ("VCM1",)
    for alpha in (0.0, 0.3):
        pinv = tool.pseudo_inverse(alpha=alpha)
        assert len(pinv.singular_values) == tool.layout.L == 4
        assert pinv.spectrum_kind == (SpectrumKind.LIVE,) * 3 + (SpectrumKind.STRUCTURAL,)
        assert pinv.singular_values[3] == 0.0 and pinv.inverse_weights[3] == 0.0
        assert np.allclose(pinv.singular_values[:3], [4.0, 2.0, 1.0], rtol=1e-13, atol=0.0)
        assert pinv.matrix[3].tobytes() == np.zeros(4).tobytes()  # exact +0.0
        assert np.all(np.isfinite(pinv.matrix))
        assert pinv.rank_discarded == 0
        assert list(pinv.no_response_correctors) == [3]
        assert pinv.no_response_correctors.dtype == np.intp
        assert pinv.no_response_names == tool.layout.no_response_names
        _assert_count_identities(pinv, tool.layout, 0)


def test_pseudo_inverse_dead_rows_are_exact_zero_on_a_random_rm_at_every_alpha(tmp_path):
    dead = ("HCM1", "VCM2")
    tool = Trajectory("pinv_dead_random", _write_rm(tmp_path / "rm.json", n_bpm=4, n_h=3, n_v=3, dead=dead))
    for alpha in (0.0, 1e-3, 2.5):
        pinv = tool.pseudo_inverse(alpha=alpha)
        assert list(pinv.no_response_correctors) == [1, 5]
        assert pinv.no_response_names == dead
        assert pinv.matrix[[1, 5]].tobytes() == np.zeros((2, 8)).tobytes()
        assert np.all(pinv.matrix[[0, 2, 3, 4]] != 0.0)
        _assert_count_identities(pinv, tool.layout, 0)


def test_pseudo_inverse_discard_cuts_structural_zeros_before_live_values():
    tool = _designed("pinv_cut_order", [4.0, 2.0], [1.0, 0.0])  # L = 4, k_live = 3, n_struct = 1
    one = tool.pseudo_inverse(n_discard=1)
    assert one.spectrum_kind == (SpectrumKind.LIVE,) * 3 + (SpectrumKind.STRUCTURAL,)
    assert np.allclose(one.inverse_weights, [0.25, 0.5, 1.0, 0.0], rtol=1e-13, atol=0.0)
    assert np.allclose(one.matrix, tool.pseudo_inverse(n_discard=0).matrix, rtol=1e-15, atol=0.0)
    two = tool.pseudo_inverse(n_discard=2)
    assert two.spectrum_kind == (SpectrumKind.LIVE, SpectrumKind.LIVE, SpectrumKind.DISCARDED, SpectrumKind.STRUCTURAL)
    assert np.allclose(two.inverse_weights, [0.25, 0.5, 0.0, 0.0], rtol=1e-13, atol=0.0)
    assert np.allclose(two.matrix, np.diag([0.25, 0.5, 0.0, 0.0]), rtol=1e-13, atol=1e-15)
    assert two.singular_values[2] != 0.0  # the value stays in the spectrum, only its weight is cut
    for pinv, n_discard in ((one, 1), (two, 2)):
        _assert_count_identities(pinv, tool.layout, n_discard)
        assert pinv.n_discard == n_discard


def test_pseudo_inverse_discard_cuts_the_smallest_live_values_first():
    tool = _designed("pinv_cut_live", [4.0, 2.0], [1.0, 0.5])
    pinv = tool.pseudo_inverse(n_discard=3)
    assert pinv.spectrum_kind == (SpectrumKind.LIVE,) + (SpectrumKind.DISCARDED,) * 3
    assert np.allclose(pinv.inverse_weights, [0.25, 0.0, 0.0, 0.0], rtol=1e-13, atol=0.0)
    assert np.allclose(pinv.singular_values, [4.0, 2.0, 1.0, 0.5], rtol=1e-13, atol=0.0)
    _assert_count_identities(pinv, tool.layout, 3)


def test_pseudo_inverse_n_discard_equal_to_l_gives_the_all_zero_inverse_without_raising():
    tool = _designed("pinv_all_cut", [4.0, 2.0], [1.0, 0.0])
    pinv = tool.pseudo_inverse(n_discard=tool.layout.L)
    assert not np.any(pinv.matrix)
    assert not np.any(pinv.inverse_weights)
    assert pinv.spectrum_kind == (SpectrumKind.DISCARDED,) * 3 + (SpectrumKind.STRUCTURAL,)
    assert np.allclose(pinv.singular_values[:3], [4.0, 2.0, 1.0], rtol=1e-13, atol=0.0)
    _assert_count_identities(pinv, tool.layout, 4)


def test_pseudo_inverse_n_discard_equal_to_l_configured_at_construction(tmp_path):
    tool = Trajectory("pinv_all_cut_cfg", _write_rm(tmp_path / "rm.json"), n_discard=6)
    pinv = tool.pseudo_inverse()
    assert not np.any(pinv.matrix)
    assert pinv.n_discard == 6
    assert set(pinv.spectrum_kind) == {SpectrumKind.DISCARDED}


def test_pseudo_inverse_more_correctors_than_rows_has_no_structural_zero(tmp_path):
    # 4 rows, 6 columns, one dead: L = 4 = k_live, so the dead column is no spectrum position.
    tool = Trajectory("pinv_wide", _write_rm(tmp_path / "rm.json", n_bpm=2, n_h=3, n_v=3, dead=("HCM2",)))
    assert tool.layout.L == 4 and tool.layout.n_struct == 0
    plain = tool.pseudo_inverse()
    assert len(plain.singular_values) == 4
    assert SpectrumKind.STRUCTURAL not in plain.spectrum_kind
    assert plain.spectrum_kind == (SpectrumKind.LIVE,) * 4
    assert plain.matrix[2].tobytes() == np.zeros(4).tobytes()
    cut = tool.pseudo_inverse(n_discard=1)
    assert cut.spectrum_kind == (SpectrumKind.LIVE,) * 3 + (SpectrumKind.DISCARDED,)
    assert cut.inverse_weights[3] == 0.0 and cut.singular_values[3] != 0.0
    assert cut.matrix[2].tobytes() == np.zeros(4).tobytes()
    for pinv, n_discard in ((plain, 0), (cut, 1)):
        _assert_count_identities(pinv, tool.layout, n_discard)


def test_pseudo_inverse_rank_guard_drops_a_collinear_pair_at_alpha_zero():
    fields = _designed_fields([4.0, 2.0], [1.0, 1.0], n_bpm=2)
    matrix = np.array(fields["matrix"])
    matrix[:, 3] = matrix[:, 2]  # VCM1 duplicates VCM0: live, but rank-deficient
    fields["matrix"] = matrix.tolist()
    tool = Trajectory("pinv_collinear", fields)
    assert tool.layout.no_response_names == () and tool.layout.n_struct == 0
    pinv = tool.pseudo_inverse()
    assert pinv.rank_discarded == 1
    assert pinv.spectrum_kind == (SpectrumKind.LIVE,) * 3 + (SpectrumKind.RANK_DISCARDED,)
    assert pinv.singular_values[3] <= 4 * _EPS * pinv.singular_values[0]
    assert pinv.inverse_weights[3] == 0.0
    assert np.all(np.isfinite(pinv.matrix))
    expected = np.linalg.pinv(matrix, rcond=max(matrix.shape) * _EPS)
    assert np.allclose(pinv.matrix, expected, rtol=1e-12, atol=1e-13)
    _assert_count_identities(pinv, tool.layout, 0)
    # The guard is applied only at alpha = 0: at alpha > 0 the pair is a tiny live weight.
    damped = tool.pseudo_inverse(alpha=0.5)
    assert damped.rank_discarded == 0
    assert damped.spectrum_kind[3] is SpectrumKind.LIVE
    assert np.all(np.isfinite(damped.matrix))


def test_pseudo_inverse_rank_discarded_counts_under_the_cut():
    fields = _designed_fields([4.0, 2.0], [1.0, 1.0], n_bpm=2)
    matrix = np.array(fields["matrix"])
    matrix[:, 3] = matrix[:, 2]
    fields["matrix"] = matrix.tolist()
    tool = Trajectory("pinv_collinear_cut", fields)
    pinv = tool.pseudo_inverse(n_discard=2)
    assert pinv.rank_discarded == 1
    assert pinv.spectrum_kind == (SpectrumKind.LIVE, SpectrumKind.LIVE, SpectrumKind.DISCARDED, SpectrumKind.RANK_DISCARDED)
    _assert_count_identities(pinv, tool.layout, 2)
    assert np.allclose(pinv.inverse_weights, [0.25, 0.5, 0.0, 0.0], rtol=1e-13, atol=0.0)


def test_pseudo_inverse_underflowing_gain_raises_on_the_all_zero_vector():
    tool = _designed("pinv_underflow", [4.0, 2.0], [1.0, 0.5], alpha=1e150)
    # s / (s^2 + alpha^2) ~ 1e-150, times 1e-200 underflows to exactly zero everywhere.
    with pytest.raises(PyAMLException, match="zero"):
        tool.pseudo_inverse(n_discard=0, gain=1e-200)


def test_pseudo_inverse_non_finite_weight_raises():
    tool = _designed("pinv_overflow", [4.0, 2.0], [1.0, 0.5])
    with pytest.raises(PyAMLException, match="finite"):
        tool.pseudo_inverse(gain=1e308)  # 1/0.5 * 1e308 overflows
    # The same gain is fine once the overflowing position is discarded.
    pinv = tool.pseudo_inverse(gain=1e308, n_discard=2)
    assert np.all(np.isfinite(pinv.matrix))


def test_pseudo_inverse_non_finite_check_happens_after_the_cut_not_before():
    tool = _designed("pinv_overflow_cut", [4.0, 2.0], [1.0, 0.5])
    # Before the cut the last weight is 1/0.5 * 1e308 = inf; after it, the vector is finite.
    pinv = tool.pseudo_inverse(gain=1e308, n_discard=1)
    assert pinv.inverse_weights[3] == 0.0
    assert np.all(np.isfinite(pinv.inverse_weights)) and np.all(np.isfinite(pinv.matrix))
    assert pinv.spectrum_kind[3] is SpectrumKind.DISCARDED


@pytest.mark.parametrize("field,value", _RANGE_REJECTIONS, ids=lambda v: repr(v))
def test_pseudo_inverse_per_call_range_rejection_raises(tmp_path, field, value):
    tool = Trajectory("pinv_range", _write_rm(tmp_path / "rm.json"))  # L = 6
    before = _full_snapshot(tool)
    with pytest.raises(PyAMLException, match=field):
        tool.pseudo_inverse(**{field: value})
    _assert_same_snapshot(before, _full_snapshot(tool))


@pytest.mark.parametrize("field,value", [("alpha", "0"), ("n_discard", 1.0), ("gain", True)], ids=lambda v: repr(v))
def test_pseudo_inverse_per_call_wrong_type_raises(tmp_path, field, value):
    tool = Trajectory("pinv_type", _write_rm(tmp_path / "rm.json"))
    with pytest.raises(PyAMLException, match=field):
        tool.pseudo_inverse(**{field: value})


def test_pseudo_inverse_none_per_call_reproduces_the_configured_result(tmp_path):
    tool = Trajectory("pinv_none", _write_rm(tmp_path / "rm.json", dead=("HCM0",)), alpha=0.4, n_discard=2, gain=1.5)
    configured = tool.pseudo_inverse()
    for pinv in (tool.pseudo_inverse(None, None, None), tool.pseudo_inverse(alpha=0.4, n_discard=2, gain=1.5)):
        assert (pinv.alpha, pinv.n_discard, pinv.gain) == (0.4, 2, 1.5)
        assert pinv.matrix.tobytes() == configured.matrix.tobytes()
        assert pinv.singular_values.tobytes() == configured.singular_values.tobytes()
        assert pinv.inverse_weights.tobytes() == configured.inverse_weights.tobytes()
        assert pinv.spectrum_kind == configured.spectrum_kind
        assert pinv.rank_discarded == configured.rank_discarded


def test_pseudo_inverse_per_call_override_does_not_change_the_configured_settings(tmp_path):
    tool = Trajectory("pinv_override", _write_rm(tmp_path / "rm.json"), alpha=0.0, n_discard=0, gain=1.0)
    configured = tool.pseudo_inverse()
    override = tool.pseudo_inverse(alpha=2.0, n_discard=3, gain=0.5)
    assert (override.alpha, override.n_discard, override.gain) == (2.0, 3, 0.5)
    assert not np.array_equal(override.matrix, configured.matrix)
    again = tool.pseudo_inverse()
    assert (again.alpha, again.n_discard, again.gain) == (0.0, 0, 1.0)
    assert again.matrix.tobytes() == configured.matrix.tobytes()


def test_pseudo_inverse_accepts_numpy_scalars_per_call(tmp_path):
    tool = Trajectory("pinv_np_scalars", _write_rm(tmp_path / "rm.json"))
    pinv = tool.pseudo_inverse(alpha=np.float64(0.25), n_discard=np.int64(1), gain=np.float32(2.0))
    assert (pinv.alpha, pinv.n_discard, pinv.gain) == (0.25, 1, 2.0)
    assert type(pinv.alpha) is float and type(pinv.n_discard) is int and type(pinv.gain) is float


def test_pseudo_inverse_is_pure_and_deterministic(tmp_path):
    tool = Trajectory("pinv_pure", _write_rm(tmp_path / "rm.json", dead=("VCM1",)), alpha=0.1, n_discard=1)
    before = _full_snapshot(tool)
    first = tool.pseudo_inverse()
    second = tool.pseudo_inverse(alpha=3.0, n_discard=4, gain=0.1)
    third = tool.pseudo_inverse()
    _assert_same_snapshot(before, _full_snapshot(tool))
    assert first is not third
    assert first.matrix.tobytes() == third.matrix.tobytes()
    assert first.inverse_weights.tobytes() == third.inverse_weights.tobytes()
    assert second.matrix.tobytes() != first.matrix.tobytes()
    assert tool.layout == _expected_layout(dead=("VCM1",))
    # Returned arrays share no memory with the tool's state.
    assert not np.shares_memory(first.matrix, tool._state.matrix)


def test_pseudo_inverse_echoes_the_layout_no_response_correctors(tmp_path):
    dead = ("HCM2", "VCM0")
    tool = Trajectory("pinv_echo", _write_rm(tmp_path / "rm.json", dead=dead))
    pinv = tool.pseudo_inverse(alpha=0.2)
    assert pinv.no_response_names == tool.layout.no_response_names == dead
    assert list(pinv.no_response_correctors) == [2, 3]
    names = tool.layout.hcorr_names + tool.layout.vcorr_names
    assert tuple(names[i] for i in pinv.no_response_correctors) == dead


@pytest.mark.parametrize(
    "n_bpm,n_h,n_v,dead,alpha,n_discard",
    [
        (4, 3, 3, (), 0.0, 0),
        (4, 3, 3, (), 0.0, 6),
        (4, 3, 3, ("HCM0",), 0.0, 0),
        (4, 3, 3, ("HCM0", "VCM2"), 0.0, 1),
        (4, 3, 3, ("HCM0", "VCM2"), 0.0, 3),
        (4, 3, 3, ("HCM0", "VCM2"), 0.7, 3),
        (2, 3, 3, ("HCM2",), 0.0, 0),
        (2, 3, 3, ("HCM2",), 0.0, 1),
        (2, 3, 3, ("HCM2", "VCM2", "VCM1"), 0.0, 2),
        (2, 3, 3, ("HCM2", "VCM2", "VCM1"), 0.3, 4),
        (3, 1, 2, ("HCM0",), 0.0, 1),
        (3, 1, 2, ("HCM0",), 1e-3, 0),
        (5, 2, 4, ("VCM3",), 0.0, 2),
        (5, 2, 4, ("VCM3",), 0.0, 6),
    ],
)
def test_pseudo_inverse_count_identities_on_constructed_cases(tmp_path, n_bpm, n_h, n_v, dead, alpha, n_discard):
    tool = Trajectory(
        f"pinv_counts_{n_bpm}_{n_h}_{n_v}_{len(dead)}_{alpha}_{n_discard}",
        _write_rm(tmp_path / "rm.json", n_bpm=n_bpm, n_h=n_h, n_v=n_v, dead=dead),
    )
    layout = tool.layout
    pinv = tool.pseudo_inverse(alpha=alpha, n_discard=n_discard)
    _assert_count_identities(pinv, layout, n_discard)
    assert pinv.matrix.shape == (layout.n_cor, 2 * layout.n_bpm)
    assert np.all(np.isfinite(pinv.matrix))
    dead_idx = [i for i, name in enumerate(layout.hcorr_names + layout.vcorr_names) if name in dead]
    assert list(pinv.no_response_correctors) == dead_idx
    for i in dead_idx:
        assert pinv.matrix[i].tobytes() == np.zeros(2 * layout.n_bpm).tobytes()
    assert (pinv.alpha, pinv.n_discard, pinv.gain) == (alpha, n_discard, 1.0)
    if n_discard == layout.L:
        assert not np.any(pinv.matrix)
    else:
        assert np.any(pinv.matrix)


# ---------------------------------------------------------------------------
# correct(): the correction step
#
# The index arrays below put BPM i at 10 * (i + 1) and corrector i of a plane at
# offset + 10 * i, so "beam to BPM j" keeps exactly the correctors whose index is
# at or before 10 * (j + 1); an offset above 10 puts every corrector after the
# first BPM. The RMs from _rm_fields are dense, so an upstream live corrector's
# step is non-zero whenever the deviation is. Nothing here reads the goldens.
# ---------------------------------------------------------------------------


def _indices(n_bpm=4, n_h=3, n_v=3, h_offset=5, v_offset=7):
    bpm = [10 * (i + 1) for i in range(n_bpm)]
    return {
        "bpm_index": bpm + bpm,
        "hcorr_index": [h_offset + 10 * i for i in range(n_h)],
        "vcorr_index": [v_offset + 10 * i for i in range(n_v)],
    }


def _readings(n_bpm=4, to_bpm=None, seed=3):
    """Finite x, y with NaN past BPM ``to_bpm`` (None keeps beam at every BPM)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 1.0, n_bpm)
    y = rng.uniform(0.2, 1.0, n_bpm)
    if to_bpm is not None:
        x[to_bpm + 1 :] = np.nan
        y[to_bpm + 1 :] = np.nan
    return x, y


def _tool(tmp_path, name, **kwargs):
    settings = {key: kwargs.pop(key) for key in ("alpha", "n_discard", "gain") if key in kwargs}
    return Trajectory(name, _write_rm(tmp_path / f"{name}.json", **kwargs), **settings)


def _deviation(x, y, reference=None):
    """The NaN-cleaned dR the tool solves with."""
    if reference is None:
        reference = np.zeros((2, len(x)))
    d_r = np.concatenate([x, y]) - np.asarray(reference, dtype=float).reshape(-1)
    d_r[np.isnan(d_r)] = 0.0
    return d_r


def _positions(step, names):
    all_names = step.names
    return [all_names.index(name) for name in names]


def _matlab_positional_zeroing(indices, last_index):
    """Positions the reference zeroes on a sorted array: everything after the last position with index <= last_index."""
    kept = [i for i, index in enumerate(indices) if index <= last_index]
    assert kept, "the positional rule is only compared where a corrector is at or before the last BPM with beam"
    return list(range(kept[-1] + 1, len(indices)))


def test_correct_without_rm_raises(tmp_path):
    tool = Trajectory("correct_no_rm", str(tmp_path / "missing.json"))
    x, y = _readings()
    with pytest.raises(PyAMLException, match="no response matrix"):
        tool.correct(x, y, **_indices())


@pytest.mark.parametrize("field,value", _RANGE_REJECTIONS, ids=lambda v: repr(v))
def test_correct_per_call_range_rejection_raises(tmp_path, field, value):
    tool = _tool(tmp_path, "correct_range")
    x, y = _readings()
    before = _full_snapshot(tool)
    with pytest.raises(PyAMLException, match=field):
        tool.correct(x, y, **_indices(), **{field: value})
    _assert_same_snapshot(before, _full_snapshot(tool))


@pytest.mark.parametrize("field,value", [("alpha", "0"), ("n_discard", 1.0), ("gain", True)], ids=lambda v: repr(v))
def test_correct_per_call_wrong_type_raises(tmp_path, field, value):
    tool = _tool(tmp_path, "correct_type")
    x, y = _readings()
    with pytest.raises(PyAMLException, match=field):
        tool.correct(x, y, **_indices(), **{field: value})


_BAD_INPUTS = [
    ("x_short", {"x": [0.0] * 3}),
    ("x_long", {"x": [0.0] * 5}),
    ("x_2d", {"x": [[0.0] * 4]}),
    ("x_scalar", {"x": 0.0}),
    ("x_text", {"x": ["a"] * 4}),
    ("x_inf", {"x": [0.0, math.inf, 0.0, 0.0]}),
    ("y_short", {"y": [0.0] * 3}),
    ("y_2d", {"y": [[0.0] * 2] * 2}),
    ("y_neg_inf", {"y": [0.0, 0.0, -math.inf, 0.0]}),
    ("reference_1d", {"reference": [0.0] * 4}),
    ("reference_three_rows", {"reference": [[0.0] * 4] * 3}),
    ("reference_wrong_length", {"reference": ([0.0] * 5, [0.0] * 5)}),
    ("reference_transposed", {"reference": [[0.0, 0.0]] * 4}),
    ("reference_ragged", {"reference": ([0.0] * 4, [0.0] * 3)}),
    ("reference_inf", {"reference": ([0.0, 0.0, 0.0, math.inf], [0.0] * 4)}),
    ("reference_neg_inf", {"reference": ([0.0] * 4, [-math.inf, 0.0, 0.0, 0.0])}),
    ("bpm_index_one_half", {"bpm_index": [10, 20, 30, 40]}),
    ("bpm_index_too_long", {"bpm_index": [10, 20, 30, 40] * 2 + [50]}),
    ("bpm_index_unequal_halves", {"bpm_index": [10, 20, 30, 40, 10, 20, 30, 41]}),
    ("bpm_index_repeated", {"bpm_index": [10, 20, 20, 40] * 2}),
    ("bpm_index_decreasing", {"bpm_index": [10, 30, 20, 40] * 2}),
    ("bpm_index_nan", {"bpm_index": [10, 20, 30, math.nan] * 2}),
    ("bpm_index_inf", {"bpm_index": [10, 20, 30, math.inf] * 2}),
    ("bpm_index_2d", {"bpm_index": [[10, 20, 30, 40]] * 2}),
    ("bpm_index_text", {"bpm_index": ["a"] * 8}),
    ("hcorr_index_short", {"hcorr_index": [5, 15]}),
    ("hcorr_index_long", {"hcorr_index": [5, 15, 25, 35]}),
    ("hcorr_index_repeated", {"hcorr_index": [5, 5, 25]}),
    ("hcorr_index_decreasing", {"hcorr_index": [25, 15, 5]}),
    ("hcorr_index_nan", {"hcorr_index": [5, math.nan, 25]}),
    ("hcorr_index_text", {"hcorr_index": ["a", "b", "c"]}),
    ("vcorr_index_short", {"vcorr_index": [7]}),
    ("vcorr_index_decreasing", {"vcorr_index": [7, 27, 17]}),
    ("vcorr_index_neg_inf", {"vcorr_index": [-math.inf, 17, 27]}),
    ("vcorr_index_2d", {"vcorr_index": [[7, 17, 27]]}),
]


@pytest.mark.parametrize("case,overrides", _BAD_INPUTS, ids=[case for case, _ in _BAD_INPUTS])
def test_correct_rejects_bad_inputs_naming_the_argument(tmp_path, case, overrides):
    tool = _tool(tmp_path, f"correct_bad_{case}")
    x, y = _readings()
    kwargs = {"x": x, "y": y, "reference": None, **_indices(), **overrides}
    before = _full_snapshot(tool)
    with pytest.raises(PyAMLException, match=next(iter(overrides))):
        tool.correct(kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("reference"), **kwargs)
    _assert_same_snapshot(before, _full_snapshot(tool))


def test_correct_layout_mismatch_names_the_argument(tmp_path):
    tool = _tool(tmp_path, "correct_layout", n_bpm=3, n_h=2, n_v=4)
    x, y = _readings(n_bpm=3)
    good = _indices(n_bpm=3, n_h=2, n_v=4)
    assert tool.correct(x, y, **good).reason is Reason.OK
    with pytest.raises(PyAMLException, match="hcorr_index"):
        tool.correct(x, y, **{**good, "hcorr_index": [5, 15, 25]})
    with pytest.raises(PyAMLException, match="vcorr_index"):
        tool.correct(x, y, **{**good, "vcorr_index": [7, 17, 27]})
    with pytest.raises(PyAMLException, match="bpm_index"):
        tool.correct(x, y, **{**good, "bpm_index": _indices()["bpm_index"]})
    with pytest.raises(PyAMLException, match="x"):
        tool.correct(_readings()[0], y, **good)
    # A load that changes the layout changes what is accepted.
    tool.load(_write_rm(tmp_path / "other.json"))
    x4, y4 = _readings()
    with pytest.raises(PyAMLException, match="x"):
        tool.correct(x, y, **_indices())
    assert tool.correct(x4, y4, **_indices()).reason is Reason.OK


def test_correct_solves_the_designed_matrix_exactly():
    tool = _designed("correct_designed", [2.0], [4.0], n_bpm=1)
    step = tool.correct([0.5], [1.0], bpm_index=[3, 3], hcorr_index=[3], vcorr_index=[1])
    assert isinstance(step, TrajectoryStep)
    assert step.reason is Reason.OK and step.status == ""
    assert step.step.tolist() == [0.25, 0.25]
    assert step.step_h.tolist() == [0.25] and step.step_v.tolist() == [0.25]
    assert step.names == ("HCM0", "VCM0") and step.n_h == 1
    assert step.last_bpm == 0 and type(step.last_bpm) is int
    assert step.zeroed_h == () and step.zeroed_v == ()
    assert not step.all_zero
    # A reference is subtracted first; a gain scales the inverse.
    step = tool.correct([0.5], [1.0], ([0.1], [0.2]), bpm_index=[3, 3], hcorr_index=[3], vcorr_index=[1])
    assert step.step.tolist() == pytest.approx([0.2, 0.2], rel=1e-15)
    step = tool.correct([0.5], [1.0], bpm_index=[3, 3], hcorr_index=[3], vcorr_index=[1], gain=0.5)
    assert step.step.tolist() == [0.125, 0.125]
    assert step.pseudo_inverse.gain == 0.5


def test_correct_step_is_the_joint_solve_with_nan_set_to_zero(tmp_path):
    tool = _tool(tmp_path, "correct_solve", dead=("HCM1",))
    x, y = _readings()
    reference = (np.array([0.1, math.nan, 0.3, 0.4]), np.array([math.nan, 0.2, 0.3, 0.4]))
    step = tool.correct(x, y, reference, **_indices(), alpha=0.3)
    pinv = tool.pseudo_inverse(alpha=0.3)
    expected = pinv.matrix @ _deviation(x, y, reference)
    assert step.step.tobytes() == expected.tobytes()
    assert step.pseudo_inverse.matrix.tobytes() == pinv.matrix.tobytes()
    assert step.reason is Reason.OK and step.zeroed_h == () and step.zeroed_v == ()
    assert step.step[1] == 0.0  # the dead corrector's row of the inverse is exactly zero
    assert np.all(step.step[[0, 2, 3, 4, 5]] != 0.0)


def test_correct_nan_in_reference_means_no_reference_at_that_bpm(tmp_path):
    tool = _tool(tmp_path, "correct_ref_nan")
    x, y = _readings()
    x_ref, y_ref = _readings(seed=11)
    with_nan = (np.where(np.arange(4) == 1, np.nan, x_ref), np.where(np.arange(4) == 2, np.nan, y_ref))
    equal_there = (np.where(np.arange(4) == 1, x, x_ref), np.where(np.arange(4) == 2, y, y_ref))
    a = tool.correct(x, y, with_nan, **_indices())
    b = tool.correct(x, y, equal_there, **_indices())
    assert a.step.tobytes() == b.step.tobytes()
    assert a.reason is b.reason is Reason.OK
    assert a.last_bpm == b.last_bpm == 3


def test_correct_reference_none_equals_an_explicit_zero_reference(tmp_path):
    tool = _tool(tmp_path, "correct_ref_none")
    x, y = _readings(to_bpm=2)
    base = tool.correct(x, y, **_indices())
    for reference in ((np.zeros(4), np.zeros(4)), np.zeros((2, 4)), [[0.0] * 4, [0.0] * 4]):
        other = tool.correct(x, y, reference, **_indices())
        assert other.step.tobytes() == base.step.tobytes()
        assert other.reason is base.reason is Reason.OK
        assert (other.last_bpm, other.zeroed_h, other.zeroed_v) == (base.last_bpm, base.zeroed_h, base.zeroed_v)


def test_correct_accepts_lists_tuples_and_arrays_alike(tmp_path):
    tool = _tool(tmp_path, "correct_forms")
    x, y = _readings(to_bpm=1)
    idx = _indices()
    base = tool.correct(x, y, **idx)
    as_lists = tool.correct(x.tolist(), tuple(y.tolist()), **idx)
    as_arrays = tool.correct(x, y, **{key: np.array(value) for key, value in idx.items()})
    as_floats = tool.correct(x, y, **{key: np.array(value, dtype=float) for key, value in idx.items()})
    for other in (as_lists, as_arrays, as_floats):
        assert other.step.tobytes() == base.step.tobytes()
        assert (other.reason, other.last_bpm, other.zeroed_h, other.zeroed_v) == (
            base.reason,
            base.last_bpm,
            base.zeroed_h,
            base.zeroed_v,
        )


@pytest.mark.parametrize("to_bpm", [0, 1, 2, 3])
def test_correct_threading_zeroes_exactly_the_correctors_past_the_last_bpm_with_beam(tmp_path, to_bpm):
    tool = _tool(tmp_path, f"correct_thread_{to_bpm}")
    idx = _indices()
    x, y = _readings(to_bpm=to_bpm)
    step = tool.correct(x, y, **idx)
    last_index = idx["bpm_index"][to_bpm]
    expected_h = tuple(f"HCM{i}" for i, index in enumerate(idx["hcorr_index"]) if index > last_index)
    expected_v = tuple(f"VCM{i}" for i, index in enumerate(idx["vcorr_index"]) if index > last_index)
    assert step.last_bpm == to_bpm
    assert (step.zeroed_h, step.zeroed_v) == (expected_h, expected_v)
    assert step.reason is Reason.OK and not step.all_zero
    full = step.pseudo_inverse.matrix @ _deviation(x, y)
    zeroed = _positions(step, step.zeroed_h + step.zeroed_v)
    kept = [i for i in range(len(step.step)) if i not in zeroed]
    assert np.all(step.step[zeroed] == 0.0)
    assert step.step[kept].tobytes() == full[kept].tobytes()
    assert np.all(step.step[kept] != 0.0)
    # On sorted index arrays the reference's positional zeroing coincides with the index comparison.
    assert _positions(step, step.zeroed_h) == _matlab_positional_zeroing(idx["hcorr_index"], last_index)
    assert [i - step.n_h for i in _positions(step, step.zeroed_v)] == _matlab_positional_zeroing(
        idx["vcorr_index"], last_index
    )


def test_correct_a_corrector_at_the_last_bpm_index_is_kept(tmp_path):
    tool = _tool(tmp_path, "correct_tie")
    idx = _indices(h_offset=10, v_offset=20)  # HCM i at the index of BPM i; VCM i at the index of BPM i+1
    x, y = _readings(to_bpm=1)
    step = tool.correct(x, y, **idx)
    assert step.last_bpm == 1
    assert step.zeroed_h == ("HCM2",)
    assert step.zeroed_v == ("VCM1", "VCM2")
    assert step.step[1] != 0.0 and step.step[3] != 0.0


def test_correct_last_bpm_follows_the_x_row_only(tmp_path):
    tool = _tool(tmp_path, "correct_x_row")
    idx = _indices()
    x, y = _readings()
    y_lost = y.copy()
    y_lost[2:] = np.nan
    step = tool.correct(x, y_lost, **idx)
    assert step.last_bpm == 3 and step.zeroed_h == () and step.zeroed_v == ()
    assert step.reason is Reason.OK
    assert step.step.tobytes() == (step.pseudo_inverse.matrix @ _deviation(x, y_lost)).tobytes()
    x_lost = x.copy()
    x_lost[2:] = np.nan
    step = tool.correct(x_lost, y, **idx)
    assert step.last_bpm == 1 and step.zeroed_h == ("HCM2",) and step.zeroed_v == ("VCM2",)
    # Beam lost and regained: the last non-NaN entry of x decides.
    x_gap = x.copy()
    x_gap[1] = np.nan
    step = tool.correct(x_gap, y, **idx)
    assert step.last_bpm == 3 and step.zeroed_h == () and step.zeroed_v == ()
    assert step.step.tobytes() == (step.pseudo_inverse.matrix @ _deviation(x_gap, y)).tobytes()


def test_correct_no_beam(tmp_path):
    tool = _tool(tmp_path, "correct_no_beam", dead=("VCM2",))
    x = np.full(4, np.nan)
    _, y = _readings()
    reference = (np.zeros(4), y + 0.5)
    step = tool.correct(x, y, reference, **_indices(), alpha=0.1)
    assert step.reason is Reason.NO_BEAM
    assert step.status == "no BPM with beam"
    assert step.last_bpm is None
    assert step.all_zero
    assert step.zeroed_h == tool.layout.hcorr_names and step.zeroed_v == tool.layout.vcorr_names
    # The reference's fallback would keep every corrector: the unthreaded solve moves the V plane.
    assert np.any(step.pseudo_inverse.matrix @ _deviation(x, y, reference))
    # All-NaN x and y, with and without a reference, read the same.
    for reference in (None, (np.full(4, np.nan), np.full(4, np.nan))):
        step = tool.correct(x, np.full(4, np.nan), reference, **_indices())
        assert step.reason is Reason.NO_BEAM and step.all_zero and step.last_bpm is None


def test_correct_no_inverse_at_n_discard_equal_to_l(tmp_path):
    tool = _tool(tmp_path, "correct_no_inverse")
    L = tool.layout.L
    x, y = _readings()
    step = tool.correct(x, y, **_indices(), n_discard=L)
    assert step.reason is Reason.NO_INVERSE
    assert step.status == "every singular value is discarded; no correction is possible"
    assert step.all_zero
    assert step.last_bpm == 3
    assert step.zeroed_h == tool.layout.hcorr_names and step.zeroed_v == tool.layout.vcorr_names
    assert step.pseudo_inverse.n_discard == L and not np.any(step.pseudo_inverse.inverse_weights)
    configured = _tool(tmp_path, "correct_no_inverse_configured", n_discard=L)
    step = configured.correct(x, y, **_indices())
    assert step.reason is Reason.NO_INVERSE and step.all_zero
    assert configured.correct(x, y, **_indices(), n_discard=L - 1).reason is Reason.OK


def test_correct_no_beam_precedes_no_inverse(tmp_path):
    tool = _tool(tmp_path, "correct_no_beam_first")
    _, y = _readings()
    step = tool.correct(np.full(4, np.nan), y, **_indices(), n_discard=tool.layout.L)
    assert step.reason is Reason.NO_BEAM and step.all_zero and step.last_bpm is None


def test_correct_beam_only_at_the_first_bpm_before_any_corrector_is_no_upstream(tmp_path):
    tool = _tool(tmp_path, "correct_no_upstream")
    idx = _indices(h_offset=15, v_offset=17)
    x, y = _readings(to_bpm=0)
    step = tool.correct(x, y, **idx)
    assert step.reason is Reason.NO_UPSTREAM
    assert step.status == "no corrector upstream of the last BPM with beam"
    assert step.last_bpm == 0
    assert step.all_zero
    assert step.zeroed_h == tool.layout.hcorr_names and step.zeroed_v == tool.layout.vcorr_names
    # Zero deviation at that BPM reads NO_UPSTREAM, never ON_REFERENCE.
    on_zero = tool.correct(np.where(np.isnan(x), np.nan, 0.0), np.where(np.isnan(y), np.nan, 0.0), **idx)
    assert on_zero.reason is Reason.NO_UPSTREAM and on_zero.all_zero
    on_ref = tool.correct(x, y, (x, y), **idx)
    assert on_ref.reason is Reason.NO_UPSTREAM and on_ref.all_zero
    # NO_INVERSE precedes NO_UPSTREAM.
    assert tool.correct(x, y, **idx, n_discard=tool.layout.L).reason is Reason.NO_INVERSE
    # One more BPM with beam reaches the first corrector of each plane.
    x2, y2 = _readings(to_bpm=1)
    reached = tool.correct(x2, y2, **idx)
    assert reached.reason is Reason.OK and reached.zeroed_h == ("HCM1", "HCM2") and reached.zeroed_v == ("VCM1", "VCM2")


def test_correct_dead_corrector_at_or_before_the_beam_is_not_upstream(tmp_path):
    tool = _tool(tmp_path, "correct_dead_upstream_h", dead=("HCM0",))
    idx = _indices()  # HCM0 at 5 and VCM0 at 7 are both before BPM0 at 10
    x, y = _readings(to_bpm=0)
    step = tool.correct(x, y, **idx)
    assert step.reason is Reason.NO_UPSTREAM_H
    assert step.status == "no horizontal corrector upstream of the last BPM with beam"
    assert step.zeroed_h == tool.layout.hcorr_names
    assert step.zeroed_v == ("VCM1", "VCM2")
    assert not np.any(step.step_h) and step.step_v[0] != 0.0 and not np.any(step.step_v[1:])
    assert not step.all_zero
    both = _tool(tmp_path, "correct_dead_upstream_both", dead=("HCM0", "VCM0"))
    step = both.correct(x, y, **idx)
    assert step.reason is Reason.NO_UPSTREAM and step.all_zero
    assert step.zeroed_h == both.layout.hcorr_names and step.zeroed_v == both.layout.vcorr_names


def test_correct_no_upstream_v(tmp_path):
    tool = _tool(tmp_path, "correct_no_upstream_v")
    idx = _indices(h_offset=5, v_offset=17)
    x, y = _readings(to_bpm=0)
    # reference == (x, y) with no upstream VCM: NO_UPSTREAM_V and all_zero.
    step = tool.correct(x, y, (x, y), **idx)
    assert step.reason is Reason.NO_UPSTREAM_V
    assert step.status == "no vertical corrector upstream of the last BPM with beam"
    assert step.all_zero
    assert step.last_bpm == 0
    assert step.zeroed_h == ("HCM1", "HCM2") and step.zeroed_v == tool.layout.vcorr_names
    # With a deviation the H plane is corrected and the V plane is zeroed.
    step = tool.correct(x, y, **idx)
    assert step.reason is Reason.NO_UPSTREAM_V and not step.all_zero
    assert step.step_h[0] != 0.0 and not np.any(step.step_h[1:]) and not np.any(step.step_v)
    assert step.zeroed_h == ("HCM1", "HCM2") and step.zeroed_v == tool.layout.vcorr_names


def test_correct_plane_without_correctors_has_no_upstream_corrector(tmp_path):
    tool = _tool(tmp_path, "correct_no_vcm", n_v=0)
    x, y = _readings()
    step = tool.correct(x, y, **_indices(n_v=0))
    assert step.reason is Reason.NO_UPSTREAM_V
    assert step.zeroed_h == () and step.zeroed_v == ()
    assert step.step_v.shape == (0,) and step.n_h == 3
    assert not step.all_zero and np.all(step.step_h != 0.0)


def test_correct_on_reference(tmp_path):
    tool = _tool(tmp_path, "correct_on_reference")
    idx = _indices()
    x, y = _readings()
    step = tool.correct(x, y, (x, y), **idx)
    assert step.reason is Reason.ON_REFERENCE
    assert step.status == "already on reference"
    assert step.all_zero and step.last_bpm == 3 and step.zeroed_h == () and step.zeroed_v == ()
    zeros = np.zeros(4)
    assert tool.correct(zeros, zeros, **idx).reason is Reason.ON_REFERENCE
    # Partial beam with an upstream corrector per plane: still ON_REFERENCE, with the tail zeroed.
    x2, y2 = _readings(to_bpm=1)
    step = tool.correct(x2, y2, (x2, y2), **idx)
    assert step.reason is Reason.ON_REFERENCE and step.all_zero
    assert step.last_bpm == 1 and step.zeroed_h == ("HCM2",) and step.zeroed_v == ("VCM2",)
    # A deviation anywhere with beam is OK.
    assert tool.correct(x2 + 0.1, y2, (x2, y2), **idx).reason is Reason.OK


def test_correct_ok_step(tmp_path):
    tool = _tool(tmp_path, "correct_ok", dead=("HCM2",))
    x, y = _readings()
    step = tool.correct(x, y, **_indices())
    assert step.reason is Reason.OK and step.status == ""
    assert not step.all_zero
    assert step.zeroed_h == () and step.zeroed_v == ()
    assert step.last_bpm == 3
    assert step.names == tool.layout.hcorr_names + tool.layout.vcorr_names
    assert step.n_h == tool.layout.n_h
    assert step.step[2] == 0.0  # dead corrector, exact-zero row
    assert np.all(step.step[[0, 1, 3, 4, 5]] != 0.0)


_STATUS_TEXT = {
    Reason.NO_BEAM: "no BPM with beam",
    Reason.NO_INVERSE: "every singular value is discarded; no correction is possible",
    Reason.NO_UPSTREAM: "no corrector upstream of the last BPM with beam",
    Reason.NO_UPSTREAM_H: "no horizontal corrector upstream of the last BPM with beam",
    Reason.NO_UPSTREAM_V: "no vertical corrector upstream of the last BPM with beam",
    Reason.ON_REFERENCE: "already on reference",
}


def _expected_reason_and_zeroed(layout, live, x, d_r, weights, idx):
    """The ruled predicates and the card's zeroed-list rule, written independently of the tool."""
    n_h = layout.n_h
    h_index, v_index = np.array(idx["hcorr_index"]), np.array(idx["vcorr_index"])
    beam = np.flatnonzero(~np.isnan(x))
    if beam.size == 0:
        past_h, past_v = np.ones(n_h, bool), np.ones(layout.n_v, bool)
    else:
        last_index = idx["bpm_index"][beam[-1]]
        past_h, past_v = h_index > last_index, v_index > last_index
    upstream_h = np.any(live[:n_h] & ~past_h)
    upstream_v = np.any(live[n_h:] & ~past_v)
    if beam.size == 0:
        reason = Reason.NO_BEAM
    elif not np.any(weights):
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
    whole_h = reason in (Reason.NO_BEAM, Reason.NO_INVERSE, Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_H)
    whole_v = reason in (Reason.NO_BEAM, Reason.NO_INVERSE, Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_V)
    zeroed_h = layout.hcorr_names if whole_h else tuple(n for n, p in zip(layout.hcorr_names, past_h, strict=True) if p)
    zeroed_v = layout.vcorr_names if whole_v else tuple(n for n, p in zip(layout.vcorr_names, past_v, strict=True) if p)
    return reason, zeroed_h, zeroed_v


@pytest.mark.parametrize("dead", [(), ("HCM0",), ("VCM0",), ("HCM0", "VCM0"), ("HCM2", "VCM1")])
def test_correct_all_zero_implies_not_ok_and_zeroed_lists_follow_the_card_on_constructed_cases(tmp_path, dead):
    tool = _tool(tmp_path, f"correct_sweep_{len(dead)}_{'_'.join(dead)}", dead=dead)
    layout = tool.layout
    live = np.array([name not in dead for name in layout.hcorr_names + layout.vcorr_names])
    seen = set()
    for offsets in ((5, 7), (15, 17), (5, 17), (17, 5)):
        idx = _indices(h_offset=offsets[0], v_offset=offsets[1])
        for to_bpm in (None, 0, 1, 3, "none"):
            x, y = (np.full(4, np.nan), _readings()[1]) if to_bpm == "none" else _readings(to_bpm=to_bpm)
            for n_discard in (0, 2, layout.L):
                for reference in (None, (x, y), (np.zeros(4), y)):
                    step = tool.correct(x, y, reference, **idx, n_discard=n_discard)
                    has_ref = "set" if reference is not None else "None"
                    label = f"offsets={offsets} to_bpm={to_bpm} n_discard={n_discard} reference={has_ref}"
                    expected = _expected_reason_and_zeroed(
                        layout, live, x, _deviation(x, y, reference), step.pseudo_inverse.inverse_weights, idx
                    )
                    assert (step.reason, step.zeroed_h, step.zeroed_v) == expected, label
                    assert not (step.all_zero and step.reason is Reason.OK), label
                    assert step.reason is not Reason.OK or not step.all_zero, label
                    assert step.status == ("" if step.reason is Reason.OK else _STATUS_TEXT[step.reason]), label
                    assert np.all(step.step[_positions(step, step.zeroed_h + step.zeroed_v)] == 0.0), label
                    assert all(isinstance(name, str) for name in step.zeroed_h + step.zeroed_v), label
                    assert step.last_bpm == (None if to_bpm == "none" else (3 if to_bpm is None else to_bpm)), label
                    seen.add(step.reason)
    assert seen >= {Reason.NO_BEAM, Reason.NO_INVERSE, Reason.NO_UPSTREAM, Reason.ON_REFERENCE, Reason.OK}
    if not dead:
        assert seen == set(Reason), f"the live sweep must reach every reason, missing {set(Reason) - seen}"


def test_correct_the_three_overlapping_pairs_never_read_ok(tmp_path):
    tool = _tool(tmp_path, "correct_overlaps")
    x, y = _readings()
    # All-NaN x and y.
    step = tool.correct(np.full(4, np.nan), np.full(4, np.nan), **_indices())
    assert step.reason is Reason.NO_BEAM and step.all_zero
    # Beam only at BPM 1 with zero deviation there.
    x1, y1 = np.full(4, np.nan), np.full(4, np.nan)
    x1[0], y1[0] = 0.0, 0.0
    step = tool.correct(x1, y1, **_indices(h_offset=15, v_offset=17))
    assert step.reason is Reason.NO_UPSTREAM and step.all_zero
    # n_discard == L.
    step = tool.correct(x, y, **_indices(), n_discard=tool.layout.L)
    assert step.reason is Reason.NO_INVERSE and step.all_zero


def test_correct_returns_read_only_fresh_arrays(tmp_path):
    tool = _tool(tmp_path, "correct_read_only")
    x, y = _readings(to_bpm=2)
    step = tool.correct(x, y, **_indices())
    assert step.step.flags.writeable is False
    with pytest.raises(ValueError):
        step.step[0] = 1.0
    assert step.step_h.flags.writeable is False and step.step_v.flags.writeable is False
    assert step.pseudo_inverse.matrix.flags.writeable is False
    assert step.pseudo_inverse.inverse_weights.flags.writeable is False
    assert step.step.dtype == float and step.step.shape == (tool.layout.n_cor,)
    assert not np.shares_memory(step.pseudo_inverse.matrix, tool._state.matrix)
    assert isinstance(step.zeroed_h, tuple) and isinstance(step.zeroed_v, tuple) and isinstance(step.names, tuple)
    other = tool.correct(x, y, **_indices())
    assert other is not step and other.step is not step.step and other.pseudo_inverse is not step.pseudo_inverse


def _settings_of(step):
    return (step.pseudo_inverse.alpha, step.pseudo_inverse.n_discard, step.pseudo_inverse.gain)


def _input_bytes(x, y, reference, idx):
    return (x.tobytes(), y.tobytes(), reference[0].tobytes(), reference[1].tobytes(), {k: v.tobytes() for k, v in idx.items()})


def test_correct_per_call_overrides_reach_the_pseudo_inverse(tmp_path):
    tool = _tool(tmp_path, "correct_overrides", alpha=0.2, n_discard=1, gain=2.0)
    x, y = _readings()
    idx = _indices()
    configured = tool.correct(x, y, **idx)
    assert _settings_of(configured) == (0.2, 1, 2.0)
    override = tool.correct(x, y, **idx, alpha=0.3, n_discard=2, gain=0.5)
    assert _settings_of(override) == (0.3, 2, 0.5)
    assert override.step.tobytes() != configured.step.tobytes()
    assert override.pseudo_inverse.matrix.tobytes() == tool.pseudo_inverse(alpha=0.3, n_discard=2, gain=0.5).matrix.tobytes()
    unit = tool.correct(x, y, **idx, gain=1.0)
    doubled = tool.correct(x, y, **idx, gain=2.0)
    assert doubled.step.tolist() == pytest.approx((2.0 * unit.step).tolist(), rel=1e-14)
    again = tool.correct(x, y, **idx)
    assert again.step.tobytes() == configured.step.tobytes()


def test_correct_is_pure(tmp_path):
    tool = _tool(tmp_path, "correct_pure", dead=("VCM1",), alpha=0.1)
    before = _full_snapshot(tool)
    x, y = _readings(to_bpm=2)
    reference = (np.array([0.1, math.nan, 0.3, 0.4]), np.array([0.0, 0.2, math.nan, 0.4]))
    idx = {key: np.array(value) for key, value in _indices().items()}
    frozen = _input_bytes(x, y, reference, idx)
    first = tool.correct(x, y, reference, **idx)
    second = tool.correct(x, y, reference, **idx, alpha=3.0, n_discard=2, gain=0.5)
    third = tool.correct(x, y, reference, **idx)
    _assert_same_snapshot(before, _full_snapshot(tool))
    assert first is not third and first.step.tobytes() == third.step.tobytes()
    assert second.step.tobytes() != first.step.tobytes()
    assert first.pseudo_inverse.matrix.tobytes() == tool.pseudo_inverse().matrix.tobytes()
    assert _input_bytes(x, y, reference, idx) == frozen
    assert not np.shares_memory(first.step, tool._state.matrix)
    with pytest.raises(PyAMLException):
        tool.correct(x[:3], y, reference, **idx)
    _assert_same_snapshot(before, _full_snapshot(tool))
    assert tool.correct(x, y, reference, **idx).step.tobytes() == first.step.tobytes()


def test_correct_repeat_after_rejected_load_second_load_and_copy_mutation_is_byte_identical(tmp_path):
    path = _write_rm(tmp_path / "rm.json", dead=("VCM1",))
    tool = Trajectory("correct_repeat", path, alpha=0.2)
    x, y = _readings(to_bpm=2)
    idx = _indices()
    base = tool.correct(x, y, **idx)
    with pytest.raises(PyAMLException):
        tool.load(tmp_path / "missing.json")
    assert tool.correct(x, y, **idx).step.tobytes() == base.step.tobytes()
    bad = _rm_fields()
    bad["matrix"] = bad["matrix"][:-1]
    with pytest.raises(PyAMLException):
        tool.load(_write_rm(tmp_path / "bad.json", fields=bad))
    assert tool.correct(x, y, **idx).step.tobytes() == base.step.tobytes()
    tool.load(path)
    assert tool.correct(x, y, **idx).step.tobytes() == base.step.tobytes()
    rm = tool.response_matrix
    rm.matrix[0][0] += 1.0
    repeat = tool.correct(x, y, **idx)
    assert repeat.step.tobytes() == base.step.tobytes()
    assert repeat.reason is base.reason and repeat.zeroed_h == base.zeroed_h and repeat.zeroed_v == base.zeroed_v


def test_correct_from_a_file_shaped_as_the_phase_2_writer_emits_it(tmp_path):
    n_bpm, n_h, n_v = 3, 2, 2
    rows = [f"SR{i}:BPM:x" for i in range(n_bpm)] + [f"SR{i}:BPM:y" for i in range(n_bpm)]
    hcor, vcor = [f"SR{i}:HCM" for i in range(n_h)], [f"SR{i}:VCM" for i in range(n_v)]
    rng = np.random.default_rng(5)
    body = {
        "type": "pyaml.tuning_tools.orbit_response_matrix_data",
        "matrix": np.round(rng.uniform(0.5, 1.5, size=(2 * n_bpm, n_h + n_v)), 9).tolist(),
        "observable_names": rows,
        "variable_names": hcor + vcor,
        "observable_planes": ["H"] * n_bpm + ["V"] * n_bpm,
        "variable_planes": ["H"] * n_h + ["V"] * n_v,
    }
    bpm = [4, 9, 14]
    sidecar = {"bpm_index": bpm + bpm, "hcor_index": [2, 7], "vcor_index": [3, 8]}
    rm_path = tmp_path / "trajectory_rm.json"
    rm_path.write_text(json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    index_path = tmp_path / "trajectory_rm.index.json"
    index_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tool = Factory.build(_config("correct_sidecar", str(rm_path)))
    assert tool.layout.bpm_names_h == tuple(rows[:n_bpm]) and tool.layout.bpm_names_v == tuple(rows[n_bpm:])
    index = json.loads(index_path.read_text())
    x, y = _readings(n_bpm=n_bpm)
    step = tool.correct(x, y, bpm_index=index["bpm_index"], hcorr_index=index["hcor_index"], vcorr_index=index["vcor_index"])
    assert step.reason is Reason.OK and not step.all_zero
    assert step.names == tuple(hcor + vcor) and step.n_h == n_h
    assert step.last_bpm == n_bpm - 1 and step.zeroed_h == () and step.zeroed_v == ()


def test_trajectory_docstring_states_the_divergences_the_x_row_rule_the_reason_order_and_the_error_rule():
    doc = " ".join(Trajectory.__doc__.split())  # whitespace-insensitive: the docstring wraps its sentences
    for case in ("dead_last_plain", "first_bpm_only", "one_plane_upstream", "no_beam"):
        assert case in doc
    assert "NO_BEAM, NO_INVERSE, NO_UPSTREAM, NO_UPSTREAM_H / NO_UPSTREAM_V, ON_REFERENCE, OK" in doc
    assert "last non-NaN entry of ``x``" in doc
    assert "PyAMLException" in doc and "PyAMLConfigException" in doc
    assert "Configuration" in doc and "response_matrix: ${path:trajectory_rm.json}" in doc
    assert "correct(" in doc


# ---------------------------------------------------------------------------
# the case table and the goldens
#
# ``trajectory_cases.json`` is the expected cases; ``trajectory_goldens.npz`` is
# what MATLAB returned for them. These tests are the goldens' one definition: the
# freeze script reads the case table by path and re-derives none of the
# assertions below. Every test that needs the npz carries ``golden`` in its name.
# ---------------------------------------------------------------------------

_CASES_PATH = Path(__file__).with_name("trajectory_cases.json")
_CASE_TABLE = json.loads(_CASES_PATH.read_text())
_CASES = {case["name"]: case for case in _CASE_TABLE["cases"]}

_CASE_COLUMNS = frozenset(
    {
        "name",
        "kind",
        "divergence_planes",
        "reason",
        "alpha_rel",
        "n_discard",
        "gain",
        "n_bpm",
        "n_h",
        "n_v",
        "dead",
        "dead_per_plane",
        "structure",
        "beam",
        "beam_to_bpm",
        "reference",
        "cond_live",
        "inputs",
    }
)
_KINDS = ("parity", "inverse", "verdict-only", "step")
_DEAD_PATTERNS = ("none", "last", "mid", "real")
_BEAM_PATTERNS = ("all", "to_bpm", "x_none")
_REFERENCE_PATTERNS = ("none", "nan_entries", "equal_to_readings", "v_only")
_STRUCTURES = ("collinear_live_pair",)
_SUITE = {"n_bpm": 16, "n_h": 12, "n_v": 12}
_EXAMPLE_SR = "example_sr_tikhonov"

PARITY_RTOL = 1e-10  # design rule 3: max|tool - MATLAB| <= PARITY_RTOL * max|MATLAB| per array
_STEP_FLOOR = 1e-6  # a real MATLAB step is ~1e-3 of max|Mplus| * max|dR|, roundoff ~1e-13 of it
_CALIBRATION = 1e9
_CALIBRATION_CASES = ("full_plain", _EXAMPLE_SR)  # alpha = 0, and the stored alpha on the real matrix
_THREADED_REASONS = frozenset({Reason.OK, Reason.ON_REFERENCE, Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_H, Reason.NO_UPSTREAM_V})
_SOLVED_REASONS = frozenset({Reason.OK, Reason.ON_REFERENCE})

_TOP_LEVEL_FIELDS = (
    "schema_version",
    "case_names",
    "n_cases",
    "sha256_cases",
    "matlab_version",
    "blas",
    "lapack",
    "host",
    "frozen_on",
    "freeze_script_sha",
    "freeze_script_commit",
    "sha256_SCgetPinv",
    "sha256_calcCMstep",
    "sha256_driver",
    "index_base",
)
_CASE_FIELDS = (
    "matrix",
    "variable_names",
    "observable_names",
    "variable_planes",
    "observable_planes",
    "n_bpm",
    "n_h",
    "n_v",
    "seed",
    "cond_live",
    "alpha_rel",
    "alpha",
    "n_discard",
    "gain",
    "x",
    "y",
    "reference",
    "bpm_index",
    "hcorr_index",
    "vcorr_index",
    "kind",
    "reason",
    "S",
    "Mplus",
    "dPhiHCM",
    "dPhiVCM",
    "n_kept_h",
    "n_kept_v",
)
_EXAMPLE_SR_FIELDS = ("names_source", "names_source_sha256", "osprey_commit")


def _case_names(*kinds):
    return [name for name, case in _CASES.items() if not kinds or case["kind"] in kinds]


def _suite_l(case):
    return min(2 * case["n_bpm"], case["n_h"] + case["n_v"])


# -- the case table itself (runs without the npz) ----------------------------


def test_case_table_has_nineteen_distinct_cases():
    names = [case["name"] for case in _CASE_TABLE["cases"]]
    assert len(names) == 19
    assert len(set(names)) == 19
    assert _CASE_TABLE["schema_version"] == 1


@pytest.mark.parametrize("name", _case_names())
def test_case_table_row_has_every_column_with_a_value_in_range(name):
    case = _CASES[name]
    assert set(case) == _CASE_COLUMNS
    assert case["kind"] in _KINDS
    assert case["reason"] in Reason.__members__
    assert isinstance(case["alpha_rel"], float) and math.isfinite(case["alpha_rel"]) and case["alpha_rel"] >= 0.0
    assert isinstance(case["n_discard"], int) and not isinstance(case["n_discard"], bool)
    assert 0 <= case["n_discard"] <= _suite_l(case)
    assert isinstance(case["gain"], float) and math.isfinite(case["gain"]) and case["gain"] > 0.0
    assert case["dead"] in _DEAD_PATTERNS
    assert case["beam"] in _BEAM_PATTERNS
    assert case["reference"] in _REFERENCE_PATTERNS
    assert set(case["structure"]) <= set(_STRUCTURES)
    assert set(case["divergence_planes"]) <= {"H", "V"}
    assert bool(case["divergence_planes"]) == (case["kind"] in ("verdict-only", "step"))
    assert isinstance(case["inputs"], str) and case["inputs"]
    if case["beam"] == "to_bpm":
        assert isinstance(case["beam_to_bpm"], int) and 1 <= case["beam_to_bpm"] < case["n_bpm"]
    else:
        assert case["beam_to_bpm"] is None
    if case["dead"] == "real":
        assert case["dead_per_plane"] is None
    else:
        assert case["dead_per_plane"] == (0 if case["dead"] == "none" else 1)
    if name == _EXAMPLE_SR:
        assert case["cond_live"] is None
        assert (case["n_bpm"], case["n_h"], case["n_v"]) == (72, 72, 72)
        assert case["dead"] == "real"
    else:
        assert isinstance(case["cond_live"], float) and math.isfinite(case["cond_live"]) and case["cond_live"] > 0.0
        assert {k: case[k] for k in _SUITE} == _SUITE


def test_case_table_states_the_divergence_cases_and_the_beam_cut():
    assert _CASES["dead_last_plain"]["kind"] == "inverse"
    assert _CASES["dead_last_plain"]["structure"] == ["collinear_live_pair"]
    assert _CASES["first_bpm_only"]["beam_to_bpm"] == 1
    assert _CASES["first_bpm_only"]["divergence_planes"] == ["H", "V"]
    assert _CASES["one_plane_upstream"]["divergence_planes"] == ["V"]
    assert _CASES["no_beam"]["divergence_planes"] == ["V"] and _CASES["no_beam"]["beam"] == "x_none"
    # the beam cut keeps a causal V block with every VCM after BPM j at full column rank
    one_plane = _CASES["one_plane_upstream"]
    assert one_plane["beam_to_bpm"] <= one_plane["n_bpm"] - one_plane["n_v"]
    assert _CASES["all_discarded"]["n_discard"] == _suite_l(_CASES["all_discarded"])
    # the only dead cases at alpha = 0 are the inverse divergence and a cut covering the structural zeros
    for name, case in _CASES.items():
        if case["dead"] in ("last", "mid") and case["alpha_rel"] == 0.0:
            assert name in ("dead_last_plain", "dead_last_discard_2")
    assert _CASES["dead_last_discard_2"]["n_discard"] == 2 * _CASES["dead_last_discard_2"]["dead_per_plane"]
    for name in _CALIBRATION_CASES:
        assert name in _CASES


# -- reading the goldens ------------------------------------------------------


@pytest.fixture(scope="module")
def goldens():
    """The frozen npz; a missing file is a ``FileNotFoundError``, never a skip."""
    path = os.environ.get("PYAML_TRAJECTORY_GOLDENS", str(Path(__file__).with_name("trajectory_goldens.npz")))
    with np.load(path, allow_pickle=False) as npz:
        yield {key: npz[key] for key in npz.files}


def _field(goldens, name, field):
    return goldens[f"{name}/{field}"]  # a missing case or field is a KeyError on its own id


def _scalar(value):
    return np.asarray(value).item()  # stored npz values pass to the tool as Python scalars


def _text(value):
    return str(np.asarray(value).item())


def _texts(value):
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _golden_tool(goldens, name, scale=1.0, alpha=None):
    """The tool built from the case's stored matrix and names (a mapping, so nothing is written to disk)."""
    fields = {
        "matrix": (scale * np.asarray(_field(goldens, name, "matrix"), dtype=float)).tolist(),
        "variable_names": _texts(_field(goldens, name, "variable_names")),
        "observable_names": _texts(_field(goldens, name, "observable_names")),
        "variable_planes": _texts(_field(goldens, name, "variable_planes")),
        "observable_planes": _texts(_field(goldens, name, "observable_planes")),
    }
    return Trajectory(
        f"golden_{name}",
        fields,
        alpha=_scalar(_field(goldens, name, "alpha")) if alpha is None else alpha,
        n_discard=_scalar(_field(goldens, name, "n_discard")),
        gain=_scalar(_field(goldens, name, "gain")),
    )


def _golden_bpm_index(goldens, name):
    """The tool's ``bpm_index``: the npz stores MATLAB's one-plane ``BPMindex``, both halves equal."""
    stored = np.asarray(_field(goldens, name, "bpm_index"))
    n_bpm = _scalar(_field(goldens, name, "n_bpm"))
    if stored.reshape(-1).size == n_bpm:
        return np.concatenate([stored.reshape(-1), stored.reshape(-1)])
    return stored


def _golden_inputs(goldens, name, scale=1.0):
    readings = {
        "x": scale * np.asarray(_field(goldens, name, "x"), dtype=float),
        "y": scale * np.asarray(_field(goldens, name, "y"), dtype=float),
        "reference": scale * np.asarray(_field(goldens, name, "reference"), dtype=float),  # verbatim
    }
    indices = {
        "bpm_index": _golden_bpm_index(goldens, name),
        "hcorr_index": np.asarray(_field(goldens, name, "hcorr_index")),
        "vcorr_index": np.asarray(_field(goldens, name, "vcorr_index")),
    }
    return readings, indices


def _golden_step(goldens, name):
    tool = _golden_tool(goldens, name)
    readings, indices = _golden_inputs(goldens, name)
    return tool, tool.correct(readings["x"], readings["y"], readings["reference"], **indices)


def _golden_d_r(goldens, name):
    """MATLAB's ``dR = R - R0`` with NaN set to 0, in RM row order (all x rows, then all y rows)."""
    readings, _ = _golden_inputs(goldens, name)
    d_r = np.concatenate([readings["x"], readings["y"]]) - readings["reference"].reshape(-1)
    d_r[np.isnan(d_r)] = 0.0
    return d_r


def _assert_parity(label, tool_array, matlab_array):
    tool_array = np.asarray(tool_array, dtype=float)
    matlab_array = np.asarray(matlab_array, dtype=float)
    assert tool_array.shape == matlab_array.shape, label
    assert np.all(np.isfinite(matlab_array)), label
    error = np.max(np.abs(tool_array - matlab_array))
    assert error <= PARITY_RTOL * np.max(np.abs(matlab_array)), f"{label}: max|tool - MATLAB| = {error:.3e}"


def _plane(step, plane):
    return step.step_h if plane == "H" else step.step_v


_DPHI = {"H": "dPhiHCM", "V": "dPhiVCM"}


def _parity_arrays(case):
    """The arrays a case compares under the parity rule (design rule 3), by kind."""
    if case["kind"] == "inverse":
        return ()  # raw Mplus and dPhi document the divergence; the inverse is compared to MATLAB pinv
    if case["kind"] == "step":
        return ("Mplus",)  # the tool's whole step is zero under NO_BEAM; MATLAB's is the divergence
    return ("Mplus",) + tuple(_DPHI[p] for p in ("H", "V") if p not in case["divergence_planes"])


# -- the goldens --------------------------------------------------------------


def test_golden_file_provenance(goldens):
    for field in _TOP_LEVEL_FIELDS:
        assert field in goldens, field
    assert _scalar(goldens["schema_version"]) == 1
    assert _text(goldens["sha256_cases"]) == hashlib.sha256(_CASES_PATH.read_bytes()).hexdigest()
    assert _text(goldens["sha256_driver"])
    assert _scalar(goldens["index_base"]) == 0
    assert sorted(_texts(goldens["case_names"])) == sorted(_CASES)
    assert _scalar(goldens["n_cases"]) == len(_CASES) == 19
    for key, value in goldens.items():
        if value.dtype.kind in "US":
            for text in _texts(value):
                assert not text.startswith(("/", "~")), key
                assert "/Users/" not in text and "/home/" not in text and ".claude/" not in text, key


@pytest.mark.parametrize("name", _case_names())
def test_golden_case_fields_and_settings_match_the_table(goldens, name):
    case = _CASES[name]
    for field in _CASE_FIELDS:
        assert f"{name}/{field}" in goldens, field
    if case["kind"] == "inverse":
        assert f"{name}/pinv" in goldens
    if name == _EXAMPLE_SR:
        for field in _EXAMPLE_SR_FIELDS:
            assert f"{name}/{field}" in goldens, field
        source = _text(_field(goldens, name, "names_source"))
        assert source and not source.startswith(("/", "~")) and ".claude/" not in source
        assert len(_text(_field(goldens, name, "names_source_sha256"))) == 64
        assert _text(_field(goldens, name, "osprey_commit"))
    assert _text(_field(goldens, name, "kind")) == case["kind"]
    assert _text(_field(goldens, name, "reason")) == case["reason"]
    assert _scalar(_field(goldens, name, "alpha_rel")) == case["alpha_rel"]
    assert _scalar(_field(goldens, name, "n_discard")) == case["n_discard"]
    assert _scalar(_field(goldens, name, "gain")) == case["gain"]
    for count in ("n_bpm", "n_h", "n_v"):
        assert _scalar(_field(goldens, name, count)) == case[count], count
    for kept in ("n_kept_h", "n_kept_v"):
        assert np.asarray(_field(goldens, name, kept)).dtype == np.int64, kept
    cond_live = _scalar(_field(goldens, name, "cond_live"))
    assert math.isfinite(cond_live) and cond_live > 0.0
    if case["cond_live"] is not None:
        assert abs(cond_live / case["cond_live"] - 1.0) <= 1e-2


@pytest.mark.parametrize("name", _case_names())
def test_golden_alpha_is_alpha_rel_times_the_matrix_norm(goldens, name):
    case = _CASES[name]
    alpha = _scalar(_field(goldens, name, "alpha"))
    if case["alpha_rel"] == 0.0:
        assert alpha == 0.0
    else:
        norm = np.linalg.norm(np.asarray(_field(goldens, name, "matrix"), dtype=float), 2)
        assert alpha == pytest.approx(case["alpha_rel"] * norm, rel=1e-12, abs=0.0)
    tool, step = _golden_step(goldens, name)
    assert step.pseudo_inverse.alpha == alpha
    assert step.pseudo_inverse.n_discard == case["n_discard"]
    assert step.pseudo_inverse.gain == case["gain"]


@pytest.mark.parametrize("name", _case_names())
def test_golden_inputs_follow_the_case_table(goldens, name):
    case = _CASES[name]
    readings, indices = _golden_inputs(goldens, name)
    x, y, reference = readings["x"], readings["y"], readings["reference"]
    n_bpm = case["n_bpm"]
    assert x.shape == (n_bpm,) and y.shape == (n_bpm,) and reference.shape == (2, n_bpm)
    if case["beam"] == "all":
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    elif case["beam"] == "to_bpm":
        j = case["beam_to_bpm"]
        assert np.all(np.isfinite(x[:j])) and np.all(np.isfinite(y[:j]))
        assert np.all(np.isnan(x[j:])) and np.all(np.isnan(y[j:]))
    else:
        assert np.all(np.isnan(x)) and np.all(np.isfinite(y))
    if case["reference"] == "none":
        assert np.array_equal(reference, np.zeros((2, n_bpm)))
    elif case["reference"] == "nan_entries":
        assert np.any(np.isnan(reference))
    elif case["reference"] == "equal_to_readings":
        assert np.array_equal(reference, np.vstack([x, y]), equal_nan=True)
    else:
        assert np.array_equal(reference[0], np.zeros(n_bpm))
        assert np.all(np.isfinite(reference[1])) and np.all(reference[1] != y)
    assert indices["bpm_index"].shape == (2 * n_bpm,)
    assert indices["hcorr_index"].shape == (case["n_h"],) and indices["vcorr_index"].shape == (case["n_v"],)


@pytest.mark.parametrize("name", _case_names())
def test_golden_layout_identities(goldens, name):
    case = _CASES[name]
    tool, step = _golden_step(goldens, name)
    layout = tool.layout
    assert (layout.n_bpm, layout.n_h, layout.n_v) == (case["n_bpm"], case["n_h"], case["n_v"])
    assert layout.n_cor == layout.n_h + layout.n_v
    assert layout.L == min(2 * layout.n_bpm, layout.n_cor)
    assert layout.k_live == min(2 * layout.n_bpm, layout.n_live)
    assert layout.n_struct == layout.L - layout.k_live
    assert step.names == tuple(_texts(_field(goldens, name, "variable_names")))
    assert len(step.pseudo_inverse.singular_values) == layout.L
    assert np.asarray(_field(goldens, name, "S")).reshape(-1).shape == (layout.L,)
    assert step.pseudo_inverse.matrix.shape == np.asarray(_field(goldens, name, "Mplus")).shape
    matrix = np.asarray(_field(goldens, name, "matrix"), dtype=float)
    dead_names = tuple(n for n, column in zip(step.names, matrix.T, strict=True) if not np.any(column))
    assert layout.no_response_names == dead_names
    if case["dead"] == "none":
        assert dead_names == ()
    elif case["dead"] == "last":
        assert dead_names == (layout.hcorr_names[-1], layout.vcorr_names[-1])
    elif case["dead"] == "mid":
        assert len(dead_names) == 2
        assert dead_names[0] in layout.hcorr_names[1:-1] and dead_names[1] in layout.vcorr_names[1:-1]


@pytest.mark.parametrize("name", _case_names())
def test_golden_reason_and_the_all_zero_invariant(goldens, name):
    tool, step = _golden_step(goldens, name)
    assert step.reason is Reason[_CASES[name]["reason"]]
    if step.all_zero:
        assert step.reason is not Reason.OK


@pytest.mark.parametrize("name", _case_names())
def test_golden_zeroing_matches_matlab_kept_counts(goldens, name):
    tool, step = _golden_step(goldens, name)
    layout = tool.layout
    if step.reason in _THREADED_REASONS:
        for plane, names, zeroed in (
            ("h", layout.hcorr_names, step.zeroed_h),
            ("v", layout.vcorr_names, step.zeroed_v),
        ):
            n_kept = _scalar(_field(goldens, name, f"n_kept_{plane}"))
            if n_kept > 0:
                assert zeroed == names[n_kept:], plane
            else:
                # the stated divergence: MATLAB zeroes nothing in a plane with no upstream corrector
                assert zeroed == names, plane
                assert step.reason in (Reason.NO_UPSTREAM, Reason.NO_UPSTREAM_H if plane == "h" else Reason.NO_UPSTREAM_V)
    else:
        # NO_BEAM and NO_INVERSE zero both planes whole; MATLAB's fallback keeps correctors, so the
        # recorded counts are not compared
        assert step.reason in (Reason.NO_BEAM, Reason.NO_INVERSE)
        assert step.zeroed_h == layout.hcorr_names
        assert step.zeroed_v == layout.vcorr_names


@pytest.mark.parametrize("name", [n for n in _case_names() if _parity_arrays(_CASES[n])])
def test_golden_parity(goldens, name):
    case = _CASES[name]
    tool, step = _golden_step(goldens, name)
    tool_arrays = {"Mplus": step.pseudo_inverse.matrix, "dPhiHCM": step.step_h, "dPhiVCM": step.step_v}
    for array in _parity_arrays(case):
        _assert_parity(f"{name}/{array}", tool_arrays[array], np.asarray(_field(goldens, name, array)))


@pytest.mark.parametrize("name", _case_names("verdict-only"))
def test_golden_verdict_only_divergence(goldens, name):
    tool, step = _golden_step(goldens, name)
    mplus = np.asarray(_field(goldens, name, "Mplus"), dtype=float)
    bound = PARITY_RTOL * np.max(np.abs(mplus)) * np.max(np.abs(_golden_d_r(goldens, name)))
    for plane in _CASES[name]["divergence_planes"]:
        assert np.all(_plane(step, plane) == 0.0), plane
        matlab = np.asarray(_field(goldens, name, _DPHI[plane]), dtype=float)
        assert np.all(np.isfinite(matlab)) and np.max(np.abs(matlab)) <= bound, plane


@pytest.mark.parametrize("name", _case_names("step"))
def test_golden_step_divergence(goldens, name):
    tool, step = _golden_step(goldens, name)
    assert np.all(step.step == 0.0)
    mplus = np.asarray(_field(goldens, name, "Mplus"), dtype=float)
    n_bpm = _CASES[name]["n_bpm"]
    for plane in _CASES[name]["divergence_planes"]:
        rows = slice(0, n_bpm) if plane == "H" else slice(n_bpm, 2 * n_bpm)
        d_r_plane = _golden_d_r(goldens, name)[rows]
        assert np.all(_plane(step, plane) == 0.0), plane
        matlab = np.asarray(_field(goldens, name, _DPHI[plane]), dtype=float)
        assert np.max(np.abs(matlab)) > _STEP_FLOOR * np.max(np.abs(mplus)) * np.max(np.abs(d_r_plane)), plane


@pytest.mark.parametrize("name", _case_names("inverse"))
def test_golden_inverse_divergence(goldens, name):
    tool, step = _golden_step(goldens, name)
    layout = tool.layout
    golden_pinv = np.asarray(_field(goldens, name, "pinv"), dtype=float)
    _assert_parity(f"{name}/pinv", tool.pseudo_inverse().matrix, golden_pinv)
    singular = np.asarray(_field(goldens, name, "S"), dtype=float).reshape(-1)
    guard = max(2 * layout.n_bpm, layout.n_cor) * np.finfo(float).eps * singular[0]
    assert np.count_nonzero(singular <= guard) == layout.n_struct + 1
    raw_mplus = np.asarray(_field(goldens, name, "Mplus"), dtype=float)
    finite = np.all(np.isfinite(raw_mplus))
    assert not finite or np.max(np.abs(raw_mplus)) > 1e6 * np.max(np.abs(golden_pinv))
    non_finite_rows = ~np.all(np.isfinite(raw_mplus), axis=1)
    raw_dphi = np.concatenate(
        [np.asarray(_field(goldens, name, "dPhiHCM"), dtype=float), np.asarray(_field(goldens, name, "dPhiVCM"), dtype=float)]
    )
    assert not np.any(np.isfinite(raw_dphi[non_finite_rows]))
    assert step.pseudo_inverse.rank_discarded == 1


@pytest.mark.parametrize("name", [n for n in _case_names() if Reason[_CASES[n]["reason"]] in _SOLVED_REASONS])
def test_golden_observable_purity(goldens, name):
    tool, step = _golden_step(goldens, name)
    zeroed = set(step.zeroed_h) | set(step.zeroed_v)
    mask = np.array([n in zeroed for n in step.names])
    solved = step.pseudo_inverse.matrix @ _golden_d_r(goldens, name)
    assert np.all(step.step[mask] == 0.0)
    assert np.array_equal(step.step[~mask], solved[~mask])


def test_golden_full_plain_zeroes_nothing(goldens):
    tool, step = _golden_step(goldens, "full_plain")
    assert step.zeroed_h == () and step.zeroed_v == ()


@pytest.mark.parametrize("name", _CALIBRATION_CASES)
def test_golden_calibration_invariance(goldens, name):
    """A change of units scales the matrix, the readings and alpha by one factor; the step is unchanged."""
    tool, step = _golden_step(goldens, name)
    scaled_tool = _golden_tool(goldens, name, scale=_CALIBRATION, alpha=_CALIBRATION * _scalar(_field(goldens, name, "alpha")))
    readings, indices = _golden_inputs(goldens, name, scale=_CALIBRATION)
    scaled = scaled_tool.correct(readings["x"], readings["y"], readings["reference"], **indices)
    assert scaled.reason is step.reason
    assert np.max(np.abs(scaled.step - step.step)) <= PARITY_RTOL * np.max(np.abs(step.step))


def test_golden_example_sr_cond_live_is_read_from_the_goldens(goldens):
    cond_live = _scalar(_field(goldens, _EXAMPLE_SR, "cond_live"))
    assert _CASES[_EXAMPLE_SR]["cond_live"] is None
    assert math.isfinite(cond_live) and cond_live > 1.0
