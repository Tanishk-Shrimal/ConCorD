"""Independent end-to-end numerical oracle for ordinary LapH closed loops."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from concord_lqcd.job import JobSpec, run_job, write_job_plan


T = 4
NV = 3
SOURCE_TIME = 1
CONFIGURATION = 17


def _write_operator_source(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, d, dbar, gammaT, gamma_identity, u, ubar",
                "",
                "def build():",
                "    u_sink = Operator(ubar('y','a','ua') * gamma_identity('ua','ub') * u('y','a','ub'), momentum=(0,0,0))",
                "    u_gamma_t = Operator(ubar('g','b','ga') * gammaT('ga','gb') * u('g','b','gb'), momentum=(0,0,0))",
                "    d_source = Operator(dbar('x','c','da') * gamma_identity('da','db') * d('x','c','db'), momentum=(0,0,0))",
                "    u_source = Operator(ubar('x','d','sa') * gamma_identity('sa','sb') * u('x','d','sb'), momentum=(0,0,0))",
                "    return {",
                "        'one_scalar_loop': Correlator(u_sink),",
                "        'one_gamma_t_loop': Correlator(u_gamma_t),",
                "        'distinct_flavor_loops': Correlator(u_sink, d_source),",
                "        'same_flavor_scalar': Correlator(u_sink, u_source),",
                "    }",
            ]
        ),
        encoding="utf-8",
    )


def _write_eigenvectors(path: Path) -> None:
    values = np.zeros((T, 1, 1, 1, 3, NV), dtype=np.complex128)
    values[:, 0, 0, 0] = np.eye(3, NV, dtype=np.complex128)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("eigenvectors", data=values)


def _write_perambulator(
    path: Path,
    generator: np.random.Generator,
) -> dict[int, np.ndarray]:
    values: dict[int, np.ndarray] = {}
    with h5py.File(path, "w") as handle:
        group = handle.create_group("perambulators")
        for source_time in range(T):
            data = (
                generator.standard_normal((T, 4, 4, NV, NV))
                + 1j * generator.standard_normal((T, 4, 4, NV, NV))
            ).astype(np.complex128)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset(
                "data",
                data=data,
            )
            dataset.attrs["dimension_order"] = (
                "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
            )
            values[source_time] = data
    return values


def _job_spec(tmp_path: Path) -> JobSpec:
    operator_file = tmp_path / "operators.py"
    _write_operator_source(operator_file)
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "disconnected_loop_oracle"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "disconnected.concord")},
            "lattice": {
                "spatial_extent": [1, 1, 1],
                "temporal_extent": T,
            },
            "compile": {
                "mode_size": NV,
                "spin_basis": "DP",
                "scalar_type": "complex128",
                "contraction": {"strategy": "auto-hq"},
            },
            "run": {
                "device": "cpu",
                "time_batch_size": 2,
                "configurations": {"values": [CONFIGURATION]},
                "times": {
                    "contexts": [
                        {
                            "id": f"sink_{sink_time}",
                            "operator_times": {
                                1: [sink_time],
                                2: [sink_time, SOURCE_TIME],
                            },
                        }
                        for sink_time in range(T)
                    ]
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(tmp_path / "eigenvectors.h5"),
                    "dataset": "/eigenvectors",
                    "axes": ["time", "z", "y", "x", "color", "mode"],
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "files": {
                            "u": str(tmp_path / "up.h5"),
                            "d": str(tmp_path / "down.h5"),
                        },
                        "dataset": "/perambulators/src_t_{source_time}/data",
                        "axes": [
                            "sink_time",
                            "source_spin",
                            "sink_spin",
                            "source_mode",
                            "sink_mode",
                        ],
                    },
                },
            },
            "output": {
                "file": str(tmp_path / "result.h5"),
                "on_existing": "error",
            },
        }
    )


def _scalar_trace(block: np.ndarray) -> np.complex128:
    return np.einsum("aaii->", block, optimize=False)


def _independent_expected(
    up: dict[int, np.ndarray],
    down: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    gamma_t = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)
    up_source_loop = _scalar_trace(up[SOURCE_TIME][SOURCE_TIME])
    down_source_loop = _scalar_trace(down[SOURCE_TIME][SOURCE_TIME])
    one_scalar = np.empty(T, dtype=np.complex128)
    one_gamma_t = np.empty(T, dtype=np.complex128)
    distinct = np.empty(T, dtype=np.complex128)
    same = np.empty(T, dtype=np.complex128)

    for sink_time in range(T):
        up_equal = up[sink_time][sink_time]
        up_sink_loop = _scalar_trace(up_equal)
        one_scalar[sink_time] = -up_sink_loop
        one_gamma_t[sink_time] = -np.einsum(
            "ab,abii->", gamma_t, up_equal, optimize=False
        )
        distinct[sink_time] = up_sink_loop * down_source_loop
        connected = np.einsum(
            "caij,acji->",
            up[SOURCE_TIME][sink_time],
            up[sink_time][SOURCE_TIME],
            optimize=False,
        )
        same[sink_time] = up_sink_loop * up_source_loop - connected

    return {
        "one_scalar_loop": one_scalar,
        "one_gamma_t_loop": one_gamma_t,
        "distinct_flavor_loops": distinct,
        "same_flavor_scalar": same,
    }


def _stored_outputs(path: Path) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for group in handle["correlators"].values():
            outputs[str(group.attrs["correlator_id"])] = np.asarray(group["values"])[0]
    return outputs


def test_disconnected_loops_match_independent_full_job_oracle(tmp_path):
    generator = np.random.default_rng(20260918)
    up = _write_perambulator(tmp_path / "up.h5", generator)
    down = _write_perambulator(tmp_path / "down.h5", generator)
    _write_eigenvectors(tmp_path / "eigenvectors.h5")
    expected = _independent_expected(up, down)

    spec = _job_spec(tmp_path)
    plan = write_job_plan(spec)
    summary = run_job(spec, plan)
    actual = _stored_outputs(spec.output.file)

    assert summary.all_complete
    assert summary.executed_work_units == T
    assert set(actual) == set(expected)
    for correlator_id in expected:
        np.testing.assert_allclose(
            actual[correlator_id],
            expected[correlator_id],
            rtol=1e-12,
            atol=1e-12,
        )

    wrong_one_loop_sign = -expected["one_scalar_loop"]
    assert np.max(np.abs(actual["one_scalar_loop"] - wrong_one_loop_sign)) > 1.0

    up_source_loop = _scalar_trace(up[SOURCE_TIME][SOURCE_TIME])
    disconnected = np.asarray(
        [
            _scalar_trace(up[sink_time][sink_time]) * up_source_loop
            for sink_time in range(T)
        ]
    )
    wrong_same_flavor_sign = 2.0 * disconnected - expected["same_flavor_scalar"]
    assert np.max(
        np.abs(actual["same_flavor_scalar"] - wrong_same_flavor_sign)
    ) > 1.0
