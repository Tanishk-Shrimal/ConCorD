"""Focused tests for the V1 ConCorD command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
import torch
import yaml
from typer.testing import CliRunner

import concord_lqcd.backend.runtime.resource_planning as resource_planning
import concord_lqcd.cli as cli_module
from concord_lqcd.backend.runtime import HardwareDeviceProfile, HardwareProfile
from concord_lqcd.cli import app
from concord_lqcd.job import COMPILER_SEMANTICS_VERSION, load_concord_plan


runner = CliRunner()

T = 6
LX = 2
LY = 1
LZ = 1
NV = 2
SOURCE = 1
SINKS = (3, 4)
CONFIGURATIONS = (0, 4)


def _synthetic_cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=12,
        ),
        system="TestOS",
        machine="test-machine",
        torch_num_threads=6,
        torch_num_interop_threads=2,
    )


def _write_operator(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "def build():",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'))",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )


def _write_data(tmp_path: Path) -> None:
    shape = (T, LZ, LY, LX, 3, NV)
    grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    for configuration in CONFIGURATIONS:
        evec = (grid + 10 * configuration) / 31.0
        evec = evec + 1j * ((grid + configuration) % 7) / 17.0
        with h5py.File(tmp_path / f"evec_{configuration}.h5", "w") as handle:
            handle.create_dataset("evec", data=evec.astype(np.complex128))

        for flavor_index, flavor in enumerate(("u", "c")):
            with h5py.File(tmp_path / f"peram_{flavor}_{configuration}.h5", "w") as handle:
                for source_time in (SOURCE,) + SINKS:
                    raw = np.arange(4 * 4 * NV * NV, dtype=np.float64).reshape(4, 4, NV, NV)
                    data = np.empty((T, 4, 4, NV, NV), dtype=np.complex128)
                    for sink_time in range(T):
                        offset = 1000 * configuration + 100 * (flavor_index + 1) + 10 * source_time + sink_time
                        data[sink_time] = offset + raw / 23.0 + 1j * (raw % 5) / 29.0
                    handle.create_dataset(f"tau_{source_time}", data=data)


def _write_job(
    tmp_path: Path,
    *,
    contraction_strategy: str | None = None,
    operator_labels: list[str] | None = None,
) -> Path:
    _write_operator(tmp_path / "operators.py")
    payload = {
        "schema_version": 1,
        "job": {"name": "cli_test"},
        "operators": {"file": "./operators.py", "factory": "build"},
        "plan": {"file": "./cli.concord"},
        "lattice": {"spatial_extent": [LX, LY, LZ], "temporal_extent": T},
        "compile": {"mode_size": NV, "scalar_type": "complex128"},
        "run": {
            "device": "cpu",
            # Keep unrelated CLI tests deterministic; automatic calibration is
            # covered by the focused job-runner tests.
            "time_batch_size": 1,
            "configurations": {"values": list(CONFIGURATIONS)},
            "times": {
                "generator": {
                    "type": "two_point",
                    "source": SOURCE,
                    "sinks": {"values": list(SINKS)},
                }
            },
        },
        "data": {
            "eigenvectors": {
                "provider": "hdf5_distillation",
                "path": "./evec_{configuration}.h5",
                "dataset": "/evec",
                "axes": ["time", "z", "y", "x", "color", "mode"],
            },
            "propagation": {
                "provider": "hdf5_perambulator",
                "ordinary": {
                    "path": "./peram_{flavor}_{configuration}.h5",
                    "dataset": "/tau_{source_time}",
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
        "output": {"file": "./cli_result.h5", "on_existing": "resume"},
    }
    if contraction_strategy is not None:
        payload["compile"]["contraction"] = {
            "strategy": contraction_strategy,
        }
    if operator_labels is not None:
        payload["visualization"] = {"operator_labels": {"D": operator_labels}}
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_help_lists_plan_inspect_visualize_and_run():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "plan" in result.stdout
    assert "inspect" in result.stdout
    assert "visualize" in result.stdout
    assert "run" in result.stdout
    assert "hardware" in result.stdout


def test_cli_hardware_reports_detected_memory_and_threads(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "detect_hardware_profile",
        _synthetic_cpu_profile,
    )

    result = runner.invoke(app, ["hardware"])

    assert result.exit_code == 0, result.output
    assert "ConCorD hardware profile" in result.stdout
    assert "cpu (Test CPU)" in result.stdout
    assert "usable CPUs      12" in result.stdout
    assert "safe working set" in result.stdout


def test_cli_plan_and_inspect_detail_views(tmp_path):
    job = _write_job(
        tmp_path,
        operator_labels=["pion sink", "pion source"],
    )
    planned = runner.invoke(app, ["plan", str(job)])
    assert planned.exit_code == 0, planned.output
    assert (tmp_path / "cli.concord").exists()
    assert "Wrote plan:" in planned.stdout
    assert "ConCorD execution plan" in planned.stdout
    assert "Outputs:" in planned.stdout

    duplicate = runner.invoke(app, ["plan", str(job)])
    assert duplicate.exit_code == 1
    assert "already exists" in duplicate.output

    forced = runner.invoke(app, ["plan", str(job), "--force"])
    assert forced.exit_code == 0, forced.output

    inspected = runner.invoke(
        app,
        [
            "inspect",
            str(tmp_path / "cli.concord"),
            "--resources",
            "--schedule",
            "--buffers",
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    assert "Resources" in inspected.stdout
    assert "Physical schedule" in inspected.stdout
    assert "Physical buffers" in inspected.stdout
    assert "ordinary" in inspected.stdout
    assert "Analytical costs" in inspected.stdout
    assert "Wick terms" in inspected.stdout
    assert "est. FLOPs" in inspected.stdout

    visualized = runner.invoke(
        app,
        ["visualize", str(job)],
    )
    assert visualized.exit_code == 0, visualized.output
    report = tmp_path / "job.html"
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "ConCorD calculation report" in content
    assert "Resulting correlator" in content
    assert r"\gamma_{5}" in content
    assert r"C_{\mathrm{D}}" in content
    assert "LaTeX source" in content
    assert "compiled tensor network" in content
    assert "Quark-flow diagram" in content
    assert "Quark flow" in content
    assert "Quark flow (source → sink)" in content
    assert "marker-end" in content
    assert "operator-label" in content
    assert 'id="tensor-network-canvas"' in content
    assert 'id="network-channel"' in content
    assert 'id="tensor-network-data"' in content
    assert "Source operator" in content
    assert "Sink operator" in content
    assert "Technical contraction details" in content
    assert "connected contractions" in content
    assert "Runtime" in content
    assert "Numerical record" in content
    assert "compiler buffer names" in content
    payload_text = content.split(
        '<script id="tensor-network-data" type="application/json">', 1
    )[1].split("</script>", 1)[0]
    network = json.loads(payload_text)
    correlator = network["correlators"][0]
    assert correlator["operator_labels"] == ["pion sink", "pion source"]
    component = network["components"][correlator["terms"][0]["components"][0]]
    assert [operator["role"] for operator in component["operators"]] == [
        "Source operator",
        "Sink operator",
    ]
    assert {
        operator["id"]: correlator["operator_labels"][operator["id"]]
        for operator in component["operators"]
    } == {0: "pion sink", 1: "pion source"}
    assert sorted(line["flavor"] for line in component["propagators"]) == [
        "c",
        "u",
    ]

    duplicate_visualization = runner.invoke(
        app,
        ["visualize", str(job)],
    )
    assert duplicate_visualization.exit_code == 1
    assert "already exists" in duplicate_visualization.output

    replaced_visualization = runner.invoke(
        app,
        ["visualize", str(job), "--force"],
    )
    assert replaced_visualization.exit_code == 0, replaced_visualization.output


def test_cli_visualization_requires_one_label_per_operator(tmp_path):
    job = _write_job(tmp_path, operator_labels=["pion"])

    planned = runner.invoke(app, ["plan", str(job)])
    assert planned.exit_code == 0, planned.output

    visualized = runner.invoke(app, ["visualize", str(job)])
    assert visualized.exit_code == 1
    assert "one label for each of its 2 operator insertions" in visualized.output


def test_cli_dry_run_auto_creates_missing_plan_without_output_tensor_io(tmp_path):
    job = _write_job(tmp_path)
    # No lattice HDF5 files are created.  Dry-run must resolve the work/resource
    # grid without materializing tensors or writing a result file.
    result = runner.invoke(app, ["run", str(job), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Plan created:" in result.stdout
    assert "Dry run successful" in result.stdout
    assert "Work units:  4" in result.stdout
    assert (tmp_path / "cli.concord").exists()
    assert not (tmp_path / "cli_result.h5").exists()


def test_cli_inspect_reports_concrete_reuse_aware_path_strategy(tmp_path):
    job = _write_job(tmp_path, contraction_strategy="reuse-aware")

    planned = runner.invoke(app, ["plan", str(job)])

    assert planned.exit_code == 0, planned.output
    assert "Contraction:      reuse-aware" in planned.stdout
    assert "Selected paths:" in planned.stdout


def test_cli_run_executes_and_second_invocation_resumes(tmp_path):
    job = _write_job(tmp_path)
    _write_data(tmp_path)

    first = runner.invoke(
        app,
        [
            "run",
            str(job),
            "--report",
            str(tmp_path / "calculation.html"),
        ],
    )
    assert first.exit_code == 0, first.output
    assert "Plan created:" in first.stdout
    assert "Job complete" in first.stdout
    assert "Executed:    4" in first.stdout
    assert "Time batch:  1" in first.stdout
    assert "Batches:     4" in first.stdout
    assert "Pool allocs: 2" in first.stdout
    assert "Runtime policies:" in first.stdout
    assert "cache reason:" in first.stdout
    assert "prefetch: enabled" in first.stdout
    assert "Cache hits:" in first.stdout
    assert "HDF5 files:" in first.stdout
    assert "Prefetch:" in first.stdout
    assert "Prefetched:" in first.stdout
    assert "Wall time:" in first.stdout
    assert "Run record:" in first.stdout
    assert "Report:" in first.stdout
    assert (tmp_path / "cli_result.h5").exists()
    assert (tmp_path / "calculation.html").exists()
    assert (tmp_path / "calculation.run.json").exists()
    report = (tmp_path / "calculation.html").read_text(encoding="utf-8")
    assert "Where the wall time went" in report
    assert "Hardware resources" in report
    assert "Resources used for this run" in report
    assert "Compute hardware" in report
    assert "Logical CPUs visible" in report
    assert "Most observed wall time was spent on" in report
    assert "Executing tensor operations" in report
    assert 'class="stage-key stage-0"' in report
    assert "use a separate device profiler for kernel-level timing" in report
    assert "Advanced execution details" in report
    runtime_section = report.split('id="runtime"', 1)[1].split(
        'id="numerical"', 1
    )[0]
    assert "Correlator channels</span>" in runtime_section
    assert "Numerical batches</span>" not in runtime_section
    numerical_section = report.split('id="numerical"', 1)[1]
    numerical_summary, numerical_advanced = numerical_section.split(
        "Advanced execution details", 1
    )
    assert "Result file" not in numerical_summary
    assert "Plan fingerprint" not in numerical_summary
    assert "Result file" in numerical_advanced
    assert "Plan fingerprint" in numerical_advanced
    assert "Numerical batches" in numerical_advanced

    second = runner.invoke(app, ["run", str(job)])
    assert second.exit_code == 0, second.output
    assert "Job complete" in second.stdout
    assert "Executed:    0" in second.stdout
    assert "Skipped:     4" in second.stdout

    with h5py.File(tmp_path / "cli_result.h5", "r") as handle:
        assert bool(np.all(handle["completed"][...]))
        assert handle["correlators/0"].attrs["correlator_id"] == "D"


def test_cli_stale_plan_requires_explicit_replan(tmp_path):
    job = _write_job(tmp_path)
    _write_data(tmp_path)

    planned = runner.invoke(app, ["plan", str(job)])
    assert planned.exit_code == 0, planned.output

    operator_path = tmp_path / "operators.py"
    operator_path.write_text(
        operator_path.read_text(encoding="utf-8") + "\n# compile fingerprint change\n",
        encoding="utf-8",
    )

    stale = runner.invoke(app, ["run", str(job), "--dry-run"])
    assert stale.exit_code == 1
    assert "stale" in stale.output.lower()
    assert "--replan" in stale.output

    replanned = runner.invoke(app, ["run", str(job), "--dry-run", "--replan"])
    assert replanned.exit_code == 0, replanned.output
    assert "Plan replanned:" in replanned.stdout
    assert "Dry run successful" in replanned.stdout


def test_cli_old_source_adjoint_plan_requires_explicit_replan(tmp_path):
    job = _write_job(tmp_path)
    _write_data(tmp_path)
    planned = runner.invoke(app, ["plan", str(job)])
    assert planned.exit_code == 0, planned.output
    assert f"Compiler semantics: {COMPILER_SEMANTICS_VERSION}" in planned.output
    plan_path = tmp_path / "cli.concord"
    with ZipFile(plan_path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    del manifest["compiler_semantics_version"]
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(plan_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, values in members.items():
            archive.writestr(name, values)

    rejected = runner.invoke(app, ["run", str(job), "--dry-run"])
    assert rejected.exit_code == 1
    assert "source-adjoint correction" in rejected.output
    assert "--replan" in rejected.output
    assert not (tmp_path / "cli_result.h5").exists()
    replanned = runner.invoke(app, ["run", str(job), "--dry-run", "--replan"])
    assert replanned.exit_code == 0, replanned.output
    assert "Plan replanned:" in replanned.output
    assert load_concord_plan(plan_path).manifest.compiler_semantics_version == COMPILER_SEMANTICS_VERSION


def test_cli_device_override_is_runtime_only(tmp_path):
    job = _write_job(tmp_path)
    # Dry-run is enough to exercise the CLI override without requiring CUDA.
    result = runner.invoke(app, ["run", str(job), "--dry-run", "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert "Compute:     cpu" in result.stdout
    assert "Storage:     cpu" in result.stdout
    assert "Mode:        resident" in result.stdout


def test_cli_auto_device_uses_hardware_planner_for_dry_run(tmp_path, monkeypatch):
    job = _write_job(tmp_path)
    monkeypatch.setattr(
        resource_planning,
        "detect_hardware_profile",
        _synthetic_cpu_profile,
    )

    result = runner.invoke(
        app,
        ["run", str(job), "--dry-run", "--device", "auto"],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run successful" in result.stdout
    assert "Compute:     cpu" in result.stdout
    assert "Storage:     cpu" in result.stdout
    assert "Mode:        resident" in result.stdout


def test_cli_inspect_hardware_reports_plan_working_set(tmp_path, monkeypatch):
    job = _write_job(tmp_path)
    planned = runner.invoke(app, ["plan", str(job)])
    assert planned.exit_code == 0, planned.output
    monkeypatch.setattr(
        cli_module,
        "detect_hardware_profile",
        _synthetic_cpu_profile,
    )

    result = runner.invoke(
        app,
        ["inspect", str(tmp_path / "cli.concord"), "--hardware"],
    )

    assert result.exit_code == 0, result.output
    assert "Hardware-aware execution feasibility" in result.stdout
    assert "Compute device:    cpu" in result.stdout
    assert "Storage device:    cpu" in result.stdout
    assert "Execution mode:    resident" in result.stdout
    assert "Known working set:" in result.stdout
    assert "provider/library workspace" in result.stdout


def test_cli_run_with_workers_reports_each_worker(tmp_path):
    job = _write_job(tmp_path)
    _write_data(tmp_path)
    threads_before = torch.get_num_threads()
    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(job),
                "--workers",
                "2",
                "--threads-per-worker",
                "1",
                "--run-record",
                str(tmp_path / "workers.run.json"),
            ],
        )
    finally:
        torch.set_num_threads(threads_before)
    assert result.exit_code == 0, result.output
    assert "Job complete" in result.stdout
    assert "Executed:    4" in result.stdout
    assert "Workers:     2 x 1 threads" in result.stdout
    assert "Worker time:" in result.stdout
    record = json.loads((tmp_path / "workers.run.json").read_text(encoding="utf-8"))
    workers = record["execution"]["workers"]
    assert [worker["work_units"] for worker in workers] == [2, 2]
    assert all(worker["threads"] == 1 for worker in workers)
