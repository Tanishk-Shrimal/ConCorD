"""End-to-end checks for independent spatial LapH correlator validation."""

from __future__ import annotations

from pathlib import Path
import struct

import h5py
import numpy as np
import pytest

from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.job import JobSpec, run_job, write_job_plan
from tools.validate_spatial_laph_baryon_correlators import compare as compare_baryon
from tools.validate_spatial_laph_correlators import compare


T = 2
LX = 3
LY = 3
LZ = 3
NV = 2
CONFIGURATION = 7
SOURCE_TIME = 0
SINK_TIME = 1
PERAMBULATOR_AXES = (
    "sink_time",
    "source_spin",
    "sink_spin",
    "source_mode",
    "sink_mode",
)
PION_CHANNELS = {
    "pion_derivative_x": "derivative/x",
    "pion_derivative_y": "derivative/y",
    "pion_derivative_z": "derivative/z",
    "pion_transport_plus_x": "transport/plus_x",
    "pion_transport_plus_y": "transport/plus_y",
    "pion_transport_plus_z": "transport/plus_z",
    "pion_transport_plus_x_plus_y": "transport/plus_x_plus_y",
}
PROTON_CHANNEL = "proton_spectator_transport_plus_x"


def _write_identity_laph_gauge(path: Path) -> None:
    values = np.zeros((3, T, LZ, LY, LX, 3, 3), dtype=np.dtype("<c16"))
    values[...] = np.eye(3, dtype=np.complex128)
    record = bytearray(struct.pack("<i", 3))
    for direction in range(3):
        record.extend(
            struct.pack(
                "<6I",
                4,
                LX,
                LY,
                LZ,
                T,
                values.dtype.itemsize * 9,
            )
        )
        record.extend(values[direction].tobytes(order="C"))

    header = b"<SmearedGaugeField/>"
    body = bytearray(struct.pack("<Q", 0))
    body.extend(b"N")
    body.extend(struct.pack("<i", len(header)))
    body.extend(header)
    record_position = len(body)
    body.extend(struct.pack("<Q", len(record)))
    body.extend(record)
    map_start = len(body)
    body.extend(struct.pack("<QQ", 1, 4))
    body.extend(struct.pack("<I", 0))
    body.extend(struct.pack("<Q", record_position))
    body.extend(b"E")
    struct.pack_into("<Q", body, 0, map_start)
    file_id = b"Laph--SmearedGaugeField4D".ljust(32, b" ")
    path.write_bytes(b"L" + file_id + body)


def _reference_values(values: np.ndarray) -> np.ndarray:
    level_first = np.moveaxis(values, -1, 0)
    return np.stack((level_first.real, level_first.imag), axis=-1)


def _elemental_values(source: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    source_modes = source.reshape(-1, source.shape[-1])
    transformed_modes = transformed.reshape(-1, transformed.shape[-1])
    elemental = source_modes.conj().T @ transformed_modes
    return np.stack((elemental.real, elemental.imag), axis=-1)


def _direct_dense_pion_references(eigenvectors, canonical_qbar, *, adjoint_source=True):
    """Build full spin-mode matrix traces before planning or executing ConCorD."""
    matrices = get_spin_matrices("DP")
    gamma = np.asarray(matrices["P"])
    gamma4 = np.asarray(matrices["gammaT"])
    gamma5 = np.asarray(matrices["gamma5"])
    source_gamma = gamma4 @ gamma.conj().T @ gamma4
    perambulator = canonical_qbar[SINK_TIME].reshape(4 * NV, 4 * NV)
    gamma5_modes = np.kron(np.eye(NV), gamma5)
    reverse = gamma5_modes @ perambulator.conj().T @ gamma5_modes
    references = {}
    for channel, path in PION_CHANNELS.items():
        source = eigenvectors[SOURCE_TIME]
        sink = eigenvectors[SINK_TIME]
        axes = {"x": 2, "y": 1, "z": 0}
        if path.startswith("derivative/"):
            axis = axes[path.rsplit("/", 1)[1]]
            transformed_source = 0.5 * (np.roll(source, -1, axis) - np.roll(source, 1, axis))
            transformed_sink = 0.5 * (np.roll(sink, -1, axis) - np.roll(sink, 1, axis))
        else:
            path_axes = tuple(axes[label] for label in path.rsplit("/", 1)[1].split("_") if label in axes)
            transformed_source = np.roll(source, (-1,) * len(path_axes), path_axes)
            transformed_sink = np.roll(sink, (-1,) * len(path_axes), path_axes)
        source_modes = source.reshape(-1, NV)
        sink_modes = sink.reshape(-1, NV)
        transformed_source_modes = transformed_source.reshape(-1, NV)
        transformed_sink_modes = transformed_sink.reshape(-1, NV)
        source_overlap = (
            transformed_source_modes.conj().T @ source_modes
            if adjoint_source else source_modes.conj().T @ transformed_source_modes
        )
        sink_overlap = sink_modes.conj().T @ transformed_sink_modes
        references[channel] = -np.trace(
            np.kron(sink_overlap, gamma) @ perambulator
            @ np.kron(source_overlap, source_gamma) @ reverse
        )
    return references


def _write_chroma_reference(
    path: Path,
    source: np.ndarray,
    *,
    time_slice: int,
    output_mode: str,
) -> None:
    spatial_axes = {"x": 2, "y": 1, "z": 0}
    operations: dict[str, tuple[np.ndarray, dict[str, str]]] = {}
    for axis_name, axis in spatial_axes.items():
        positive = np.roll(source, shift=-1, axis=axis)
        negative = np.roll(source, shift=1, axis=axis)
        operations[f"transport/plus_{axis_name}"] = (
            positive,
            {"path": f"+{axis_name}"},
        )
        operations[f"transport/minus_{axis_name}"] = (
            negative,
            {"path": f"-{axis_name}"},
        )
        operations[f"derivative/{axis_name}"] = (
            0.5 * (positive - negative),
            {"direction": f"+{axis_name}"},
        )
    operations["transport/plus_x_plus_y"] = (
        np.roll(source, shift=(-1, -1), axis=(2, 1)),
        {"path": "+x,+y"},
    )
    with h5py.File(path, "w") as handle:
        compact = output_mode == "elementals"
        handle.attrs["format"] = (
            "chroma_laph_spatial_reference_v2"
            if compact
            else "chroma_laph_spatial_reference_v1"
        )
        if compact:
            handle.attrs["representation"] = "elementals"
        handle.attrs["time_slice"] = time_slice
        handle.create_dataset("levels", data=np.arange(NV, dtype=np.int32))
        source_dataset = handle.create_dataset(
            "source",
            data=(
                _elemental_values(source, source)
                if compact
                else _reference_values(source)
            ),
        )
        source_dataset.attrs["dimension_order"] = (
            "bra_level,ket_level,complex_pair"
            if compact
            else "level,z,y,x,color,complex_pair"
        )
        for operation_path, (transformed, attributes) in operations.items():
            dataset = handle.create_dataset(
                operation_path,
                data=(
                    _elemental_values(source, transformed)
                    if compact
                    else _reference_values(transformed)
                ),
            )
            dataset.attrs["dimension_order"] = (
                "bra_level,ket_level,complex_pair"
                if compact
                else "level,z,y,x,color,complex_pair"
            )
            for name, value in attributes.items():
                dataset.attrs[name] = value


def _write_qdagger_perambulator(
    path: Path,
    canonical_qbar: np.ndarray,
) -> None:
    gamma4 = np.asarray(get_spin_matrices("DP")["gammaT"], dtype=np.complex128)
    qdagger = np.einsum("tiajc,cb->tiajb", canonical_qbar, gamma4, optimize=True)
    stored = np.transpose(qdagger, (0, 4, 2, 3, 1))
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "/perambulators/src_t_0/data",
            data=stored,
        )
        dataset.attrs["dimension_order"] = (
            "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
        )


def _job_spec(tmp_path: Path, perambulator_path: Path) -> JobSpec:
    operator_file = Path(__file__).resolve().parents[1] / "examples" / "displaced_operators" / "pions.py"
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "spatial_nonlocal_pion_validation"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "spatial_nonlocal.concord")},
            "lattice": {
                "spatial_extent": [LX, LY, LZ],
                "temporal_extent": T,
            },
            "compile": {
                "mode_size": NV,
                "scalar_type": "complex128",
                "spin_basis": "DP",
            },
            "run": {
                "device": "cpu",
                "time_batch_size": 1,
                "configurations": {"values": [CONFIGURATION]},
                "times": {
                    "generator": {
                        "type": "two_point",
                        "source": SOURCE_TIME,
                        "sinks": {"values": [SINK_TIME]},
                    }
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(tmp_path / "eigenvectors.h5"),
                    "dataset": "/eigenvectors",
                    "axes": ["time", "z", "y", "x", "color", "mode"],
                },
                "gauge_links": {
                    "provider": "laph_iomap",
                    "path": str(tmp_path / "smeared_gauge"),
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "files": {
                            "u": str(perambulator_path),
                            "d": str(perambulator_path),
                        },
                        "dataset": "/perambulators/src_t_{fixed_source_time}/data",
                        "fixed_source_time": SOURCE_TIME,
                        "source_field": "qdagger",
                        "axes": list(PERAMBULATOR_AXES),
                    },
                },
            },
            "output": {
                "file": str(tmp_path / "concord_lqcd.h5"),
                "on_existing": "error",
            },
        }
    )


def _proton_job_spec(tmp_path: Path, perambulator_path: Path) -> JobSpec:
    operator_file = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "displaced_operators"
        / "proton.py"
    )
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "spatial_nonlocal_proton_validation"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "spatial_nonlocal_proton.concord")},
            "lattice": {
                "spatial_extent": [LX, LY, LZ],
                "temporal_extent": T,
            },
            "compile": {
                "mode_size": NV,
                "scalar_type": "complex128",
                "spin_basis": "DP",
            },
            "run": {
                "device": "cpu",
                "time_batch_size": 1,
                "configurations": {"values": [CONFIGURATION]},
                "times": {
                    "generator": {
                        "type": "two_point",
                        "source": SOURCE_TIME,
                        "sinks": {"values": [SINK_TIME]},
                    }
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(tmp_path / "eigenvectors.h5"),
                    "dataset": "/eigenvectors",
                    "axes": ["time", "z", "y", "x", "color", "mode"],
                },
                "gauge_links": {
                    "provider": "laph_iomap",
                    "path": str(tmp_path / "smeared_gauge"),
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "files": {
                            "u": str(perambulator_path),
                            "d": str(perambulator_path),
                        },
                        "dataset": "/perambulators/src_t_{fixed_source_time}/data",
                        "fixed_source_time": SOURCE_TIME,
                        "source_field": "qdagger",
                        "axes": list(PERAMBULATOR_AXES),
                    },
                },
            },
            "output": {
                "file": str(tmp_path / "concord_proton.h5"),
                "on_existing": "error",
            },
        }
    )
def _run_validation(
    tmp_path: Path,
    *,
    source_mode: str = "fields",
    sink_mode: str = "fields",
) -> tuple[dict, JobSpec]:
    generator = np.random.default_rng(20260914)
    eigenvectors = generator.normal(size=(T, LZ, LY, LX, 3, NV))
    eigenvectors = eigenvectors + 1j * generator.normal(size=eigenvectors.shape)
    eigenvectors = eigenvectors.astype(np.complex128)
    with h5py.File(tmp_path / "eigenvectors.h5", "w") as handle:
        handle.create_dataset("eigenvectors", data=eigenvectors)

    source_reference_path = tmp_path / "chroma_reference_source.h5"
    sink_reference_path = tmp_path / "chroma_reference_sink.h5"
    _write_chroma_reference(
        source_reference_path,
        eigenvectors[SOURCE_TIME],
        time_slice=SOURCE_TIME,
        output_mode=source_mode,
    )
    _write_chroma_reference(
        sink_reference_path,
        eigenvectors[SINK_TIME],
        time_slice=SINK_TIME,
        output_mode=sink_mode,
    )
    _write_identity_laph_gauge(tmp_path / "smeared_gauge")

    canonical_qbar = generator.normal(size=(T, NV, 4, NV, 4))
    canonical_qbar = canonical_qbar + 1j * generator.normal(
        size=canonical_qbar.shape
    )
    perambulator_path = tmp_path / "perambulator.h5"
    _write_qdagger_perambulator(
        perambulator_path,
        canonical_qbar.astype(np.complex128),
    )

    spec = _job_spec(tmp_path, perambulator_path)
    expected = _direct_dense_pion_references(eigenvectors, canonical_qbar)
    wrong_adjoint = _direct_dense_pion_references(
        eigenvectors, canonical_qbar, adjoint_source=False,
    )
    for axis in ("x", "y", "z"):
        channel = f"pion_derivative_{axis}"
        assert abs(expected[channel]) > 1e-6
        np.testing.assert_allclose(wrong_adjoint[channel], -expected[channel], rtol=5e-13, atol=0)
    assert abs(expected["pion_transport_plus_x"] - wrong_adjoint["pion_transport_plus_x"]) > 1e-6
    write_job_plan(spec)
    summary = run_job(spec)
    assert summary.all_complete
    with h5py.File(spec.output.file, "r") as handle:
        for group in handle["correlators"].values():
            channel = group.attrs["correlator_id"]
            np.testing.assert_allclose(group["values"][0, 0], expected[channel], rtol=5e-13, atol=0)

    report = compare(
        source_reference_path=source_reference_path,
        sink_reference_path=sink_reference_path,
        quark_perambulator_path=perambulator_path,
        antiquark_perambulator_path=perambulator_path,
        perambulator_dataset="/perambulators/src_t_{fixed_source_time}/data",
        perambulator_axes=PERAMBULATOR_AXES,
        source_field="qdagger",
        concord_output_path=spec.output.file,
        channels=PION_CHANNELS,
        configuration=CONFIGURATION,
        source_time=SOURCE_TIME,
        sink_time=SINK_TIME,
        spin_basis="DP",
        spin_matrix="P",
        rtol=5.0e-13,
        atol=0.0,
    )
    return report, spec


@pytest.mark.parametrize(
    ("source_mode", "sink_mode"),
    (
        ("fields", "fields"),
        ("elementals", "elementals"),
        ("fields", "elementals"),
    ),
)
def test_independent_spatial_pion_reference_matches_complete_concord_job(
    tmp_path,
    source_mode,
    sink_mode,
):
    report, _ = _run_validation(
        tmp_path,
        source_mode=source_mode,
        sink_mode=sink_mode,
    )

    assert report["passed"] is True
    assert set(report["comparisons"]) == set(PION_CHANNELS)
    assert all(
        comparison["passed"] for comparison in report["comparisons"].values()
    )


def test_independent_spatial_pion_reference_detects_correlator_change(tmp_path):
    report, spec = _run_validation(tmp_path)
    assert report["passed"] is True

    with h5py.File(spec.output.file, "r+") as handle:
        derivative_group = next(
            group
            for group in handle["correlators"].values()
            if group.attrs["correlator_id"] == "pion_derivative_x"
        )
        derivative_group["values"][0, 0] += 1.0

    changed = compare(
        source_reference_path=tmp_path / "chroma_reference_source.h5",
        sink_reference_path=tmp_path / "chroma_reference_sink.h5",
        quark_perambulator_path=tmp_path / "perambulator.h5",
        antiquark_perambulator_path=tmp_path / "perambulator.h5",
        perambulator_dataset="/perambulators/src_t_{fixed_source_time}/data",
        perambulator_axes=PERAMBULATOR_AXES,
        source_field="qdagger",
        concord_output_path=spec.output.file,
        channels=PION_CHANNELS,
        configuration=CONFIGURATION,
        source_time=SOURCE_TIME,
        sink_time=SINK_TIME,
        spin_basis="DP",
        spin_matrix="P",
        rtol=5.0e-13,
        atol=0.0,
    )

    assert changed["passed"] is False
    assert changed["comparisons"]["pion_derivative_x"]["passed"] is False
    assert changed["comparisons"]["pion_transport_plus_x"]["passed"] is True


@pytest.mark.filterwarnings("error:The given NumPy array is not writable:UserWarning")
def test_independent_spectator_transport_proton_matches_complete_concord_job(
    tmp_path,
):
    generator = np.random.default_rng(20260915)
    eigenvectors = generator.normal(size=(T, LZ, LY, LX, 3, NV))
    eigenvectors = eigenvectors + 1j * generator.normal(size=eigenvectors.shape)
    eigenvectors = eigenvectors.astype(np.complex128)
    with h5py.File(tmp_path / "eigenvectors.h5", "w") as handle:
        handle.create_dataset("eigenvectors", data=eigenvectors)

    source_reference = tmp_path / "chroma_proton_source.h5"
    sink_reference = tmp_path / "chroma_proton_sink.h5"
    _write_chroma_reference(
        source_reference,
        eigenvectors[SOURCE_TIME],
        time_slice=SOURCE_TIME,
        output_mode="fields",
    )
    _write_chroma_reference(
        sink_reference,
        eigenvectors[SINK_TIME],
        time_slice=SINK_TIME,
        output_mode="fields",
    )
    _write_identity_laph_gauge(tmp_path / "smeared_gauge")

    canonical_qbar = generator.normal(size=(T, NV, 4, NV, 4))
    canonical_qbar = canonical_qbar + 1j * generator.normal(
        size=canonical_qbar.shape
    )
    perambulator_path = tmp_path / "perambulator.h5"
    _write_qdagger_perambulator(
        perambulator_path,
        canonical_qbar.astype(np.complex128),
    )

    spec = _proton_job_spec(tmp_path, perambulator_path)
    write_job_plan(spec)
    summary = run_job(spec)
    assert summary.all_complete

    report = compare_baryon(
        source_reference_path=source_reference,
        sink_reference_path=sink_reference,
        up_perambulator_path=perambulator_path,
        down_perambulator_path=perambulator_path,
        perambulator_dataset="/perambulators/src_t_{fixed_source_time}/data",
        perambulator_axes=PERAMBULATOR_AXES,
        source_field="qdagger",
        concord_output_path=spec.output.file,
        correlator_id=PROTON_CHANNEL,
        configuration=CONFIGURATION,
        source_time=SOURCE_TIME,
        sink_time=SINK_TIME,
        source_operation="transport/plus_x",
        sink_operation="transport/plus_x",
        transformed_leg=2,
        spin_basis="DP",
        device="cpu",
        rtol=5.0e-13,
        atol=0.0,
    )

    assert report["passed"] is True
    assert report["comparison"]["actual_shape"] == [4, 4]
    assert report["comparison"]["reference_frobenius_norm"] > 0.0
    assert report["comparison"]["frobenius_relative"] < 5.0e-13
    assert report["comparison"]["max_tolerance_ratio"] <= 1.0
    assert report["transformed_field"] == "spectator_u"
