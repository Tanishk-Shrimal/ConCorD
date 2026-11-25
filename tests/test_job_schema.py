"""Focused tests for the V1 single-file ConCorD job specification."""

from __future__ import annotations

from pathlib import Path

import pytest

from concord_lqcd.core.correlator import Correlator
from concord_lqcd.job import (
    JobLoadError,
    JobSpec,
    OperatorLoadError,
    compute_job_fingerprint,
    load_job_spec,
    load_operator_collection,
)


def _base_payload():
    return {
        "schema_version": 1,
        "job": {"name": "test_job"},
        "operators": {
            "file": "operators.py",
            "factory": "build",
            "parameters": {"momentum": [0, 0, 0]},
        },
        "plan": {"file": "plans/test.concord"},
        "lattice": {
            "spatial_extent": [24, 24, 24],
            "temporal_extent": 128,
        },
        "compile": {"mode_size": 45},
        "run": {
            "device": "cpu",
            "configurations": {
                "range": {"start": 100, "stop": 112, "step": 4}
            },
            "times": {
                "generator": {
                    "type": "two_point",
                    "source": 20,
                    "sinks": {
                        "range": {"start": 21, "stop": 24}
                    },
                }
            },
        },
        "data": {
            "eigenvectors": {
                "provider": "hdf5_distillation",
                "path": "data/evec/cfg_{configuration}.h5",
                "dataset": "/eigenvectors",
                "axes": ["time", "z", "y", "x", "color", "mode"],
            },
            "propagation": {
                "provider": "hdf5_perambulator",
                "ordinary": {
                    "path": "data/peram/{flavor}/cfg_{configuration}.h5",
                    "dataset": "/perambulators/src_t_{source_time}",
                    "axes": [
                        "sink_time",
                        "sink_spin",
                        "source_spin",
                        "sink_mode",
                        "source_mode",
                    ],
                },
            },
        },
        "output": {
            "file": "results/test.h5",
            "on_existing": "resume",
        },
    }


def test_job_schema_defaults_and_stop_exclusive_expansion():
    spec = JobSpec.model_validate(_base_payload())

    assert spec.schema_version == 1
    assert spec.compile.mode_size == 45
    assert spec.compile.spin_size == 4
    assert spec.compile.scalar_type == "complex128"
    assert spec.compile.element_bytes == 16
    assert spec.compile.contraction.strategy == "greedy"
    assert spec.compile.scheduling.strategy == "memory_aware"
    assert spec.compile.physical.reuse_strategy == "best_fit"
    assert spec.run.time_batch_size == "auto"
    assert spec.configurations == (100, 104, 108)

    contexts = spec.time_contexts
    assert tuple(context.id for context in contexts) == (
        "2pt_src_20_snk_21",
        "2pt_src_20_snk_22",
        "2pt_src_20_snk_23",
    )
    assert tuple(context.operator_times for context in contexts) == (
        {2: (21, 20)},
        {2: (22, 20)},
        {2: (23, 20)},
    )


def test_visualization_labels_are_presentation_only(tmp_path: Path):
    payload = _base_payload()
    payload["operators"]["file"] = str(tmp_path / "operators.py")
    payload["visualization"] = {
        "operator_labels": {
            "pion": ["pion sink", "pion source"],
        }
    }
    spec_with_labels = JobSpec.model_validate(payload)
    payload.pop("visualization")
    spec_without_labels = JobSpec.model_validate(payload)

    assert spec_with_labels.visualization.operator_labels == {
        "pion": ("pion sink", "pion source")
    }
    assert compute_job_fingerprint(
        spec_with_labels, operator_source_sha256="same-source"
    ) == compute_job_fingerprint(
        spec_without_labels, operator_source_sha256="same-source"
    )


def test_visualization_labels_reject_empty_names():
    payload = _base_payload()
    payload["visualization"] = {
        "operator_labels": {"pion": ["pion sink", "  "]}
    }

    with pytest.raises(Exception, match="String should have at least 1 character"):
        JobSpec.model_validate(payload)


def test_two_point_generator_expands_multiple_sources_source_major():
    payload = _base_payload()
    payload["run"]["times"] = {
        "generator": {
            "type": "two_point",
            "sources": {"values": [3, 7]},
            "sinks": {"values": [1, 2]},
        }
    }

    spec = JobSpec.model_validate(payload)

    assert tuple(context.id for context in spec.time_contexts) == (
        "2pt_src_3_snk_1",
        "2pt_src_3_snk_2",
        "2pt_src_7_snk_1",
        "2pt_src_7_snk_2",
    )
    assert tuple(context.operator_times for context in spec.time_contexts) == (
        {2: (1, 3)},
        {2: (2, 3)},
        {2: (1, 7)},
        {2: (2, 7)},
    )


def test_two_point_generator_requires_exactly_one_source_form():
    payload = _base_payload()
    payload["run"]["times"]["generator"]["sources"] = {"values": [3, 7]}
    with pytest.raises(Exception, match="exactly one"):
        JobSpec.model_validate(payload)

    payload = _base_payload()
    del payload["run"]["times"]["generator"]["source"]
    with pytest.raises(Exception, match="exactly one"):
        JobSpec.model_validate(payload)


def test_job_schema_accepts_automatic_runtime_device_selection():
    payload = _base_payload()
    payload["run"]["device"] = "auto"

    spec = JobSpec.model_validate(payload)

    assert spec.run.device == "auto"


@pytest.mark.parametrize("provider", ("laph_iomap", "openqcd"))
def test_job_schema_accepts_native_laph_and_openqcd_gauge_links(provider):
    payload = _base_payload()
    payload["data"]["gauge_links"] = {
        "provider": provider,
        "path": "data/gauge/smeared_gauge_cfg{configuration}",
    }

    spec = JobSpec.model_validate(payload)

    assert spec.data.gauge_links.provider == provider
    assert spec.data.gauge_links.path == (
        "data/gauge/smeared_gauge_cfg{configuration}"
    )

    payload["data"]["gauge_links"]["path"] = "gauge_{time}"
    with pytest.raises(Exception, match="unsupported placeholder"):
        JobSpec.model_validate(payload)


def test_job_schema_accepts_auto_or_positive_time_batch_size():
    payload = _base_payload()
    payload["run"]["time_batch_size"] = 3
    assert JobSpec.model_validate(payload).run.time_batch_size == 3

    payload["run"]["time_batch_size"] = 0
    with pytest.raises(Exception, match="time_batch_size"):
        JobSpec.model_validate(payload)

    payload["run"]["time_batch_size"] = "largest"
    with pytest.raises(Exception, match="time_batch_size"):
        JobSpec.model_validate(payload)


def test_job_schema_exposes_reuse_aware_but_not_internal_path_candidates():
    payload = _base_payload()
    payload["compile"]["contraction"] = {"strategy": "reuse-aware"}

    spec = JobSpec.model_validate(payload)

    assert spec.compile.contraction.strategy == "reuse-aware"

    payload["compile"]["contraction"] = {"strategy": "branch-1"}
    with pytest.raises(Exception, match="contraction.strategy"):
        JobSpec.model_validate(payload)


def test_job_schema_exposes_hardware_adaptive_meta_strategy():
    payload = _base_payload()
    payload["compile"]["contraction"] = {"strategy": "adaptive"}

    spec = JobSpec.model_validate(payload)

    assert spec.compile.contraction.strategy == "adaptive"


def test_ordinary_propagation_accepts_fixed_source_dataset_metadata():
    payload = _base_payload()
    ordinary = payload["data"]["propagation"]["ordinary"]
    ordinary["dataset"] = "/perambulators/src_t_{fixed_source_time}/data"
    ordinary["fixed_source_time"] = 20

    spec = JobSpec.model_validate(payload)

    assert spec.data.propagation.ordinary.fixed_source_time == 20


def test_complex64_derives_element_bytes_instead_of_accepting_user_byte_model():
    payload = _base_payload()
    payload["compile"]["scalar_type"] = "complex64"
    spec = JobSpec.model_validate(payload)
    assert spec.compile.element_bytes == 8

    payload = _base_payload()
    payload["compile"]["element_bytes"] = 8
    with pytest.raises(Exception, match="element_bytes"):
        JobSpec.model_validate(payload)


def test_three_point_generator_expands_sink_insertion_source_order():
    payload = _base_payload()
    payload["run"]["times"] = {
        "generator": {
            "type": "three_point",
            "source": 20,
            "sink": 40,
            "insertions": {"values": [22, 24, 30]},
        }
    }
    spec = JobSpec.model_validate(payload)

    assert tuple(context.operator_times for context in spec.time_contexts) == (
        {3: (40, 22, 20)},
        {3: (40, 24, 20)},
        {3: (40, 30, 20)},
    )


def test_explicit_contexts_support_mixed_arities():
    payload = _base_payload()
    payload["run"]["times"] = {
        "contexts": [
            {
                "id": "geometry_0",
                "operator_times": {
                    2: [40, 20],
                    3: [40, 30, 20],
                },
            }
        ]
    }
    spec = JobSpec.model_validate(payload)

    assert spec.time_contexts[0].operator_times == {
        2: (40, 20),
        3: (40, 30, 20),
    }


def test_time_bounds_are_checked_against_temporal_extent():
    payload = _base_payload()
    payload["run"]["times"] = {
        "generator": {
            "type": "two_point",
            "source": 20,
            "sinks": {"values": [127, 128]},
        }
    }
    with pytest.raises(Exception, match=r"outside \[0, 128\)"):
        JobSpec.model_validate(payload)


def test_integer_sequence_requires_exactly_one_nonempty_form():
    payload = _base_payload()
    payload["run"]["configurations"] = {
        "values": [1, 2],
        "range": {"start": 1, "stop": 3},
    }
    with pytest.raises(Exception, match="exactly one"):
        JobSpec.model_validate(payload)

    payload = _base_payload()
    payload["run"]["configurations"] = {
        "range": {"start": 4, "stop": 4}
    }
    with pytest.raises(Exception, match="range is empty"):
        JobSpec.model_validate(payload)

    payload = _base_payload()
    payload["run"]["configurations"] = {
        "range": {"start": 1, "stop": 4, "step": 0}
    }
    with pytest.raises(Exception, match="step cannot be zero"):
        JobSpec.model_validate(payload)


def test_extra_keys_and_unsupported_placeholders_are_rejected():
    payload = _base_payload()
    payload["compile"]["unknown_setting"] = True
    with pytest.raises(Exception, match="unknown_setting"):
        JobSpec.model_validate(payload)

    payload = _base_payload()
    payload["data"]["propagation"]["ordinary"]["path"] = (
        "data/{resource_id}/{flavor}.h5"
    )
    with pytest.raises(Exception, match="resource_id"):
        JobSpec.model_validate(payload)


def test_ordinary_propagation_requires_path_xor_files_and_all_data_axes():
    payload = _base_payload()
    ordinary = payload["data"]["propagation"]["ordinary"]
    ordinary["files"] = {"u": "u.h5"}
    with pytest.raises(Exception, match="exactly one"):
        JobSpec.model_validate(payload)

    payload = _base_payload()
    payload["data"]["propagation"]["ordinary"]["axes"] = [
        "sink_spin",
        "source_spin",
        "sink_mode",
    ]
    with pytest.raises(Exception, match="source_mode"):
        JobSpec.model_validate(payload)


def test_load_job_spec_resolves_relative_filesystem_paths_from_yaml_directory(tmp_path):
    job_dir = tmp_path / "calculation"
    job_dir.mkdir()
    yaml_path = job_dir / "job.yaml"

    import yaml

    yaml_path.write_text(yaml.safe_dump(_base_payload()), encoding="utf-8")
    spec = load_job_spec(yaml_path)

    assert spec.operators.file == (job_dir / "operators.py").resolve()
    assert spec.plan.file == (job_dir / "plans/test.concord").resolve()
    assert spec.output.file == (job_dir / "results/test.h5").resolve()
    assert spec.data.eigenvectors.path == str(
        (job_dir / "data/evec/cfg_{configuration}.h5").resolve()
    )
    assert spec.data.propagation.ordinary.path == str(
        (job_dir / "data/peram/{flavor}/cfg_{configuration}.h5").resolve()
    )
    assert spec.data.eigenvectors.dataset == "/eigenvectors"

    payload = _base_payload()
    payload["data"]["gauge_links"] = {
        "provider": "laph_iomap",
        "path": "data/gauge/smeared_gauge_cfg{configuration}",
    }
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    spec = load_job_spec(yaml_path)
    assert spec.data.gauge_links.path == str(
        (job_dir / "data/gauge/smeared_gauge_cfg{configuration}").resolve()
    )
    payload["data"]["gauge_links"] = {"provider": "openqcd", "path": "cfg/A_n{configuration}"}
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    spec = load_job_spec(yaml_path)
    assert spec.data.gauge_links.provider == "openqcd"
    assert spec.data.gauge_links.path == str((job_dir / "cfg/A_n{configuration}").resolve())

    payload["data"]["kernel_cache"] = {"directory": "cache/kernels"}
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    spec = load_job_spec(yaml_path)
    assert spec.data.kernel_cache.directory == (job_dir / "cache/kernels").resolve()


def test_load_job_spec_reports_invalid_yaml_root(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(JobLoadError, match="root must be a mapping"):
        load_job_spec(path)


def test_operator_loader_executes_factory_parameters_and_accepts_mapping(tmp_path):
    operator_file = tmp_path / "operators.py"
    operator_file.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "",
                "def build(momentum):",
                "    p = tuple(momentum)",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=p)",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=tuple(-x for x in p))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )

    payload = _base_payload()
    payload["operators"] = {
        "file": str(operator_file),
        "factory": "build",
        "parameters": {"momentum": [0, 0, 1]},
    }
    spec = JobSpec.model_validate(payload)

    result = load_operator_collection(spec)
    assert set(result) == {"D"}
    assert isinstance(result["D"], Correlator)


def test_operator_loader_rejects_missing_factory_and_wrong_result(tmp_path):
    operator_file = tmp_path / "operators.py"
    operator_file.write_text("value = 3\n", encoding="utf-8")

    payload = _base_payload()
    payload["operators"] = {
        "file": str(operator_file),
        "factory": "build",
    }
    spec = JobSpec.model_validate(payload)

    with pytest.raises(OperatorLoadError, match="has no factory"):
        load_operator_collection(spec)

    operator_file.write_text("def build():\n    return 17\n", encoding="utf-8")
    with pytest.raises(OperatorLoadError, match="must return"):
        load_operator_collection(spec)


def test_custom_references_are_validated_and_relative_file_references_are_resolved(tmp_path):
    payload = _base_payload()
    payload["data"]["eigenvectors"] = {
        "provider": "custom",
        "factory": "./providers.py:build_kernel_provider",
    }
    payload["data"]["propagation"]["generalized"] = {
        "resolver": "./resolvers.py:resolve_generalized",
    }

    import yaml

    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    spec = load_job_spec(path)

    assert spec.data.eigenvectors.factory == (
        f"{(tmp_path / 'providers.py').resolve()}:build_kernel_provider"
    )
    assert spec.data.propagation.generalized.resolver == (
        f"{(tmp_path / 'resolvers.py').resolve()}:resolve_generalized"
    )


def test_persistent_kernel_cache_requires_builtin_hdf5_eigenvectors():
    payload = _base_payload()
    payload["data"]["eigenvectors"] = {
        "provider": "custom",
        "factory": "providers:build_kernel_provider",
    }
    payload["data"]["kernel_cache"] = {"directory": "cache/kernels"}

    with pytest.raises(Exception, match="requires HDF5 LapH eigenvectors"):
        JobSpec.model_validate(payload)


def test_kernel_cache_packing_is_explicit_and_validated():
    payload = _base_payload()
    payload["data"]["kernel_cache"] = {"directory": "cache/kernels"}
    assert JobSpec.model_validate(payload).data.kernel_cache.packing == "none"
    payload["data"]["kernel_cache"]["packing"] = "antisym3"
    assert JobSpec.model_validate(payload).data.kernel_cache.packing == "antisym3"
    for invalid in ("automatic", "gzip", True, None):
        payload["data"]["kernel_cache"]["packing"] = invalid
        with pytest.raises(Exception, match="packing"):
            JobSpec.model_validate(payload)


def test_ordinary_propagation_source_field_is_explicit_and_validated():
    payload = _base_payload()
    ordinary = payload["data"]["propagation"]["ordinary"]
    assert JobSpec.model_validate(payload).data.propagation.ordinary.source_field == "qbar"

    ordinary["source_field"] = "qdagger"
    assert JobSpec.model_validate(payload).data.propagation.ordinary.source_field == "qdagger"

    ordinary["source_field"] = "unknown"
    with pytest.raises(Exception, match="source_field"):
        JobSpec.model_validate(payload)
