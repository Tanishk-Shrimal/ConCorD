#!/usr/bin/env python3
"""Compare spatially transformed pion correlators with a direct reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from concord_lqcd.core.dirac_basis import get_spin_matrices


FIELD_REFERENCE_FORMAT = "chroma_laph_spatial_reference_v1"
ELEMENTAL_REFERENCE_FORMAT = "chroma_laph_spatial_reference_v2"
FIELD_REFERENCE_ORDER = "level,z,y,x,color,complex_pair"
ELEMENTAL_REFERENCE_ORDER = "bra_level,ket_level,complex_pair"
CANONICAL_PROPAGATION_AXES = (
    "sink_mode",
    "sink_spin",
    "source_mode",
    "source_spin",
)
DEFAULT_PROPAGATION_AXES = (
    "sink_time",
    "source_spin",
    "sink_spin",
    "source_mode",
    "sink_mode",
)
_AXIS_ALIASES = {
    "t_sink": "sink_time",
    "spin_src": "source_spin",
    "spin_source": "source_spin",
    "src_spin": "source_spin",
    "spin_snk": "sink_spin",
    "spin_sink": "sink_spin",
    "snk_spin": "sink_spin",
    "src_eigs": "source_mode",
    "source_eigs": "source_mode",
    "eig_src": "source_mode",
    "snk_eigs": "sink_mode",
    "sink_eigs": "sink_mode",
    "eig_snk": "sink_mode",
}


@dataclass(frozen=True)
class SpatialReference:
    path: Path
    time_slice: int
    levels: tuple[int, ...]
    representation: str
    source: np.ndarray
    operations: Mapping[str, np.ndarray]


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _complex_field(dataset: h5py.Dataset) -> np.ndarray:
    order = _text(dataset.attrs.get("dimension_order", ""))
    if order != FIELD_REFERENCE_ORDER:
        raise ValueError(
            f"{dataset.name} dimension_order is {order!r}, "
            f"expected {FIELD_REFERENCE_ORDER!r}"
        )
    values = np.asarray(dataset)
    if values.ndim != 6 or values.shape[-1] != 2:
        raise ValueError(
            f"{dataset.name} must have shape [level,z,y,x,color,2], "
            f"got {values.shape}"
        )
    return np.asarray(values[..., 0] + 1j * values[..., 1], dtype=np.complex128)


def _complex_elemental(dataset: h5py.Dataset) -> np.ndarray:
    order = _text(dataset.attrs.get("dimension_order", ""))
    if order != ELEMENTAL_REFERENCE_ORDER:
        raise ValueError(
            f"{dataset.name} dimension_order is {order!r}, "
            f"expected {ELEMENTAL_REFERENCE_ORDER!r}"
        )
    values = np.asarray(dataset)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(
            f"{dataset.name} must have shape [bra_level,ket_level,2], "
            f"got {values.shape}"
        )
    return np.asarray(values[..., 0] + 1j * values[..., 1], dtype=np.complex128)


def load_spatial_reference(
    path: Path,
    operation_paths: Sequence[str],
) -> SpatialReference:
    operations: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        file_format = _text(handle.attrs.get("format", ""))
        if file_format == FIELD_REFERENCE_FORMAT:
            representation = "fields"
            decoder = _complex_field
        elif file_format == ELEMENTAL_REFERENCE_FORMAT:
            representation = _text(handle.attrs.get("representation", ""))
            if representation != "elementals":
                raise ValueError(
                    f"{path}: v2 reference representation must be 'elementals'"
                )
            decoder = _complex_elemental
        else:
            raise ValueError(
                f"unsupported spatial reference format {file_format!r} in {path}"
            )
        time_slice = int(handle.attrs["time_slice"])
        levels_array = np.asarray(handle["levels"], dtype=np.int64)
        if levels_array.ndim != 1 or levels_array.size == 0:
            raise ValueError(f"{path}: levels must be a non-empty vector")
        levels = tuple(int(level) for level in levels_array)
        if tuple(sorted(set(levels))) != levels or levels[0] < 0:
            raise ValueError(f"{path}: levels must be strictly increasing")
        source = decoder(handle["source"])
        if source.shape[0] != len(levels):
            raise ValueError(f"{path}: source level axis disagrees with /levels")
        if representation == "elementals" and source.shape != (
            len(levels),
            len(levels),
        ):
            raise ValueError(
                f"{path}: source elemental shape {source.shape} disagrees with /levels"
            )
        for operation_path in operation_paths:
            normalized = operation_path.strip("/")
            if normalized not in handle:
                raise ValueError(f"{path}: operation /{normalized} is absent")
            values = decoder(handle[normalized])
            if values.shape != source.shape:
                raise ValueError(
                    f"{path}: operation /{normalized} shape {values.shape} "
                    f"does not match source shape {source.shape}"
                )
            operations[normalized] = values
    return SpatialReference(
        path=path,
        time_slice=time_slice,
        levels=levels,
        representation=representation,
        source=source,
        operations=operations,
    )


def _elemental(reference: SpatialReference, operation_path: str) -> np.ndarray:
    transformed = reference.operations[operation_path.strip("/")]
    if reference.representation == "elementals":
        return transformed
    source = reference.source.reshape(len(reference.levels), -1)
    transformed = transformed.reshape(len(reference.levels), -1)
    return source.conj() @ transformed.T


def _take_levels(
    values: np.ndarray,
    levels: Sequence[int],
    *,
    axis: int,
    role: str,
) -> np.ndarray:
    if not levels:
        raise ValueError(f"{role} levels must be non-empty")
    largest = max(int(level) for level in levels)
    if min(int(level) for level in levels) < 0 or largest >= values.shape[axis]:
        raise ValueError(
            f"{role} LapH level {largest} lies outside axis extent "
            f"{values.shape[axis]}"
        )
    return np.take(values, tuple(levels), axis=axis)


def _normalized_dimension_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        values: Sequence[Any] = tuple(
            part.strip() for part in value.split(",") if part.strip()
        )
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError("dimension_order must be a string or sequence")
    return tuple(_AXIS_ALIASES.get(_text(axis).strip(), _text(axis).strip()) for axis in values)


def _decode_hdf5_complex(
    array: np.ndarray,
    axes: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(array)
    names = values.dtype.names
    if names is not None:
        real_name = "re" if "re" in names else ("real" if "real" in names else None)
        imag_name = "im" if "im" in names else ("imag" if "imag" in names else None)
        if real_name is None or imag_name is None:
            if len(names) != 2:
                raise ValueError("compound perambulator data must contain two fields")
            real_name, imag_name = names
        values = np.asarray(values[real_name]) + 1j * np.asarray(values[imag_name])
    if "complex_pair" in axes:
        pair_axis = axes.index("complex_pair")
        if values.shape[pair_axis] != 2:
            raise ValueError("perambulator complex_pair axis must have extent two")
        values = np.take(values, 0, axis=pair_axis) + 1j * np.take(
            values, 1, axis=pair_axis
        )
        axes.pop(pair_axis)
    return np.asarray(values, dtype=np.complex128), axes


def load_fixed_source_perambulator(
    *,
    path: Path,
    dataset_template: str,
    source_time: int,
    sink_time: int,
    axes: Sequence[str],
    sink_levels: Sequence[int],
    source_levels: Sequence[int],
    source_field: str,
    spin_basis: str,
) -> np.ndarray:
    dataset_name = dataset_template.format(
        fixed_source_time=source_time,
        source_time=source_time,
        sink_time=sink_time,
    )
    configured_axes = tuple(str(axis) for axis in axes)
    if len(configured_axes) != len(set(configured_axes)):
        raise ValueError("perambulator axes contain duplicates")
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"{path}: perambulator dataset {dataset_name!r} is absent")
        dataset = handle[dataset_name]
        if dataset.ndim != len(configured_axes):
            raise ValueError(
                f"{path}:{dataset_name} rank {dataset.ndim} disagrees with "
                f"the {len(configured_axes)} configured axes"
            )
        if "dimension_order" in dataset.attrs:
            actual_axes = _normalized_dimension_order(dataset.attrs["dimension_order"])
            if actual_axes != configured_axes:
                raise ValueError(
                    f"{path}:{dataset_name} dimension_order normalizes to "
                    f"{actual_axes}, expected {configured_axes}"
                )
        if "sink_time" not in configured_axes:
            raise ValueError("fixed-source perambulator axes require sink_time")
        indexers: list[int | slice] = [slice(None)] * len(configured_axes)
        indexers[configured_axes.index("sink_time")] = sink_time
        try:
            stored = np.asarray(dataset[tuple(indexers)])
        except IndexError as exc:
            raise ValueError(f"sink time {sink_time} lies outside {dataset_name}") from exc

    remaining_axes = [axis for axis in configured_axes if axis != "sink_time"]
    stored, remaining_axes = _decode_hdf5_complex(stored, remaining_axes)
    if set(remaining_axes) != set(CANONICAL_PROPAGATION_AXES):
        raise ValueError(
            "selected perambulator axes must be exactly "
            f"{CANONICAL_PROPAGATION_AXES}, got {tuple(remaining_axes)}"
        )
    permutation = tuple(remaining_axes.index(axis) for axis in CANONICAL_PROPAGATION_AXES)
    canonical = np.transpose(stored, permutation)
    canonical = _take_levels(
        canonical,
        sink_levels,
        axis=0,
        role="sink",
    )
    canonical = _take_levels(
        canonical,
        source_levels,
        axis=2,
        role="source",
    )
    if canonical.shape[1] != 4 or canonical.shape[3] != 4:
        raise ValueError(
            "spatial hadron validation requires four sink and source spin components; "
            f"got shape {canonical.shape}"
        )

    if source_field == "qdagger":
        gamma4 = np.asarray(
            get_spin_matrices(spin_basis)["gammaT"], dtype=np.complex128
        )
        canonical = np.einsum("iajc,cb->iajb", canonical, gamma4, optimize=True)
    elif source_field != "qbar":
        raise ValueError("source_field must be 'qbar' or 'qdagger'")
    return np.ascontiguousarray(canonical)


def _reverse_fixed_source(
    direct: np.ndarray,
    *,
    spin_basis: str,
) -> np.ndarray:
    sink_modes, sink_spins, source_modes, source_spins = direct.shape
    gamma5 = np.asarray(
        get_spin_matrices(spin_basis)["gamma5"], dtype=np.complex128
    )
    left = np.kron(np.eye(source_modes, dtype=np.complex128), gamma5)
    right = np.kron(np.eye(sink_modes, dtype=np.complex128), gamma5)
    matrix = direct.reshape(sink_modes * sink_spins, source_modes * source_spins)
    return (left @ matrix.conj().T @ right).reshape(
        source_modes,
        source_spins,
        sink_modes,
        sink_spins,
    )


def direct_pion_correlator(
    *,
    sink_elemental: np.ndarray,
    source_elemental: np.ndarray,
    quark_perambulator: np.ndarray,
    antiquark_perambulator: np.ndarray,
    spin_basis: str,
    spin_matrix: str,
) -> complex:
    """Contract forward sink/source elementals with the source Hermitian adjoint."""
    matrices = get_spin_matrices(spin_basis)
    try:
        gamma = np.asarray(matrices[spin_matrix], dtype=np.complex128)
    except KeyError as exc:
        raise ValueError(f"unknown spin matrix {spin_matrix!r}") from exc
    gamma4 = np.asarray(matrices["gammaT"], dtype=np.complex128)
    source_gamma = gamma4 @ gamma.conj().T @ gamma4
    reverse = _reverse_fixed_source(antiquark_perambulator, spin_basis=spin_basis)
    return complex(
        -np.einsum(
            "ij,ab,jbkc,cd,kl,ldia->",
            sink_elemental,
            gamma,
            quark_perambulator,
            source_gamma,
            source_elemental.conj().T,
            reverse,
            optimize=True,
        )
    )


def _find_work_indices(
    handle: h5py.File,
    *,
    configuration: int,
    source_time: int,
    sink_time: int,
) -> tuple[int, int]:
    configurations = np.asarray(handle["work/configurations"], dtype=np.int64)
    matches = np.flatnonzero(configurations == configuration)
    if matches.size != 1:
        raise ValueError(
            f"ConCorD output contains {matches.size} rows for configuration {configuration}"
        )
    expected = (sink_time, source_time)
    time_matches: list[int] = []
    for index, raw in enumerate(handle["work/operator_times_json"]):
        payload = json.loads(_text(raw))
        times = payload.get("2")
        if times is not None and tuple(int(value) for value in times) == expected:
            time_matches.append(index)
    if len(time_matches) != 1:
        raise ValueError(
            f"ConCorD output contains {len(time_matches)} work columns for "
            f"sink/source times {expected}"
        )
    configuration_index = int(matches[0])
    time_index = time_matches[0]
    if not bool(handle["completed"][configuration_index, time_index]):
        raise ValueError("requested ConCorD work unit is incomplete")
    return configuration_index, time_index


def load_concord_tensors(
    *,
    path: Path,
    correlator_ids: Sequence[str],
    configuration: int,
    source_time: int,
    sink_time: int,
) -> Mapping[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        configuration_index, time_index = _find_work_indices(
            handle,
            configuration=configuration,
            source_time=source_time,
            sink_time=sink_time,
        )
        groups = {
            _text(group.attrs.get("correlator_id", "")): group
            for group in handle["correlators"].values()
        }
        result: dict[str, np.ndarray] = {}
        for correlator_id in correlator_ids:
            try:
                values = groups[correlator_id]["values"]
            except KeyError as exc:
                raise ValueError(
                    f"ConCorD output has no correlator {correlator_id!r}"
                ) from exc
            result[correlator_id] = np.asarray(
                values[configuration_index, time_index]
            )
    return result


def load_concord_values(
    *,
    path: Path,
    correlator_ids: Sequence[str],
    configuration: int,
    source_time: int,
    sink_time: int,
) -> Mapping[str, complex]:
    tensors = load_concord_tensors(
        path=path,
        correlator_ids=correlator_ids,
        configuration=configuration,
        source_time=source_time,
        sink_time=sink_time,
    )
    result: dict[str, complex] = {}
    for correlator_id, value in tensors.items():
        if value.shape != ():
            raise ValueError(
                f"correlator {correlator_id!r} is not scalar; got shape {value.shape}"
            )
        result[correlator_id] = complex(value)
    return result


def _metrics(actual: complex, expected: complex, *, rtol: float, atol: float) -> dict[str, Any]:
    difference = abs(actual - expected)
    reference_magnitude = abs(expected)
    scale = max(reference_magnitude, np.finfo(np.float64).tiny)
    return {
        "passed": bool(np.isclose(actual, expected, rtol=rtol, atol=atol)),
        "concord": {"real": actual.real, "imag": actual.imag},
        "reference": {"real": expected.real, "imag": expected.imag},
        "reference_magnitude": float(reference_magnitude),
        "absolute_difference": float(difference),
        "scaled_difference": float(difference / scale),
    }


def compare(
    *,
    source_reference_path: Path,
    sink_reference_path: Path,
    quark_perambulator_path: Path,
    antiquark_perambulator_path: Path,
    perambulator_dataset: str,
    perambulator_axes: Sequence[str],
    source_field: str,
    concord_output_path: Path,
    channels: Mapping[str, str],
    configuration: int,
    source_time: int,
    sink_time: int,
    spin_basis: str,
    spin_matrix: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    operation_paths = tuple(dict.fromkeys(path.strip("/") for path in channels.values()))
    source_reference = load_spatial_reference(source_reference_path, operation_paths)
    sink_reference = load_spatial_reference(sink_reference_path, operation_paths)
    if source_reference.time_slice != source_time:
        raise ValueError(
            f"source reference time is {source_reference.time_slice}, expected {source_time}"
        )
    if sink_reference.time_slice != sink_time:
        raise ValueError(
            f"sink reference time is {sink_reference.time_slice}, expected {sink_time}"
        )
    if source_reference.levels != sink_reference.levels:
        raise ValueError("source and sink Chroma references use different LapH levels")

    levels = source_reference.levels
    quark = load_fixed_source_perambulator(
        path=quark_perambulator_path,
        dataset_template=perambulator_dataset,
        source_time=source_time,
        sink_time=sink_time,
        axes=perambulator_axes,
        sink_levels=levels,
        source_levels=levels,
        source_field=source_field,
        spin_basis=spin_basis,
    )
    antiquark = load_fixed_source_perambulator(
        path=antiquark_perambulator_path,
        dataset_template=perambulator_dataset,
        source_time=source_time,
        sink_time=sink_time,
        axes=perambulator_axes,
        sink_levels=levels,
        source_levels=levels,
        source_field=source_field,
        spin_basis=spin_basis,
    )
    actual_values = load_concord_values(
        path=concord_output_path,
        correlator_ids=tuple(channels),
        configuration=configuration,
        source_time=source_time,
        sink_time=sink_time,
    )

    comparisons: dict[str, dict[str, Any]] = {}
    for correlator_id, operation_path in channels.items():
        normalized = operation_path.strip("/")
        expected = direct_pion_correlator(
            sink_elemental=_elemental(sink_reference, normalized),
            source_elemental=_elemental(source_reference, normalized),
            quark_perambulator=quark,
            antiquark_perambulator=antiquark,
            spin_basis=spin_basis,
            spin_matrix=spin_matrix,
        )
        comparisons[correlator_id] = {
            "operation": normalized,
            **_metrics(actual_values[correlator_id], expected, rtol=rtol, atol=atol),
        }

    return {
        "passed": all(result["passed"] for result in comparisons.values()),
        "configuration": configuration,
        "source_time": source_time,
        "sink_time": sink_time,
        "levels": list(levels),
        "spin_basis": spin_basis,
        "spin_matrix": spin_matrix,
        "source_reference": str(source_reference_path),
        "source_reference_representation": source_reference.representation,
        "sink_reference": str(sink_reference_path),
        "sink_reference_representation": sink_reference.representation,
        "quark_perambulator": str(quark_perambulator_path),
        "antiquark_perambulator": str(antiquark_perambulator_path),
        "concord_output": str(concord_output_path),
        "rtol": rtol,
        "atol": atol,
        "comparisons": comparisons,
    }


def _channel(value: str) -> tuple[str, str]:
    correlator_id, separator, operation = value.partition("=")
    if not separator or not correlator_id.strip() or not operation.strip("/").strip():
        raise argparse.ArgumentTypeError("channel must be CORRELATOR_ID=REFERENCE_DATASET")
    return correlator_id.strip(), operation.strip("/")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--sink-reference", type=Path, required=True)
    parser.add_argument("--quark-perambulator", type=Path, required=True)
    parser.add_argument("--antiquark-perambulator", type=Path)
    parser.add_argument(
        "--perambulator-dataset",
        default="/perambulators/src_t_{fixed_source_time}/data",
    )
    parser.add_argument(
        "--perambulator-axes",
        default=",".join(DEFAULT_PROPAGATION_AXES),
    )
    parser.add_argument("--source-field", choices=("qbar", "qdagger"), default="qdagger")
    parser.add_argument("--concord-output", type=Path, required=True)
    parser.add_argument("--channel", action="append", type=_channel, required=True)
    parser.add_argument("--configuration", type=int, required=True)
    parser.add_argument("--source-time", type=int, required=True)
    parser.add_argument("--sink-time", type=int, required=True)
    parser.add_argument("--spin-basis", choices=("DR", "DP"), default="DP")
    parser.add_argument("--spin-matrix", default="P")
    parser.add_argument("--rtol", type=float, default=5.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--json-report", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    channels = dict(args.channel)
    if len(channels) != len(args.channel):
        raise SystemExit("--channel correlator ids must be unique")
    axes = tuple(axis.strip() for axis in args.perambulator_axes.split(",") if axis.strip())
    report = compare(
        source_reference_path=args.source_reference,
        sink_reference_path=args.sink_reference,
        quark_perambulator_path=args.quark_perambulator,
        antiquark_perambulator_path=(
            args.antiquark_perambulator or args.quark_perambulator
        ),
        perambulator_dataset=args.perambulator_dataset,
        perambulator_axes=axes,
        source_field=args.source_field,
        concord_output_path=args.concord_output,
        channels=channels,
        configuration=args.configuration,
        source_time=args.source_time,
        sink_time=args.sink_time,
        spin_basis=args.spin_basis,
        spin_matrix=args.spin_matrix,
        rtol=args.rtol,
        atol=args.atol,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
