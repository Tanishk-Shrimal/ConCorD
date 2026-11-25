from __future__ import annotations

import pytest

from concord_lqcd.backend.runtime.resource_planning import MemorySafetyMargin
from concord_lqcd.job._worker_split import auto_worker_split, workers_for_peak, workers_in_memory


GIB = 1024**3
MARGIN = MemorySafetyMargin(0.20, 2 * GIB)  # the default CPU margin
# Steps, cost-report FLOPs and buffer bytes per time slice of the saved N_v = 32
# plans, as measured on a 64-core node.
PIPI_18 = {"steps": 160_675, "flops": 105_710_291_456, "peak_bytes": 1_329_555_024}
NPI_14 = {"steps": 23_716, "flops": 1_393_263_478_688, "peak_bytes": 1_797_571_184}
NN_10 = {"steps": 27_400, "flops": 2_277_019_255_040, "peak_bytes": 3_911_500_240}


def _split(plan, *, pending=128, available_bytes=240 * 10**9, threads="auto", **node):
    shape = {"cores": 64, "domain_cores": 16, "can_fork": True, **node}
    return auto_worker_split(
        threads=threads,
        pending=pending,
        available_bytes=available_bytes,
        margin=MARGIN,
        **plan,
        **shape,
    )


@pytest.mark.parametrize(
    ("plan", "expected"),
    [(PIPI_18, (64, 1)), (NPI_14, (16, 4)), (NN_10, (16, 4))],
)
def test_auto_split_matches_the_best_measured_split(plan, expected):
    # 64 cores in four memory domains: one thread per worker for the plan with
    # tiny steps, four for the plans dominated by arithmetic.
    split = _split(plan)
    assert (split.workers, split.threads) == expected
    assert split.automatic and split.reason.startswith("auto: ")


def test_auto_split_fits_workers_in_memory_and_gives_them_the_idle_cores():
    split = _split(NN_10, available_bytes=40 * 10**9)
    assert (split.workers, split.threads) == (6, 10)
    assert "6 by memory" in split.reason
    share = 40 * 10**9 // 6
    assert share - MARGIN.reserve(share) >= NN_10["peak_bytes"]
    share = 40 * 10**9 // 7
    assert share - MARGIN.reserve(share) < NN_10["peak_bytes"]


def test_auto_split_uses_one_worker_per_pending_unit_within_one_memory_domain():
    split = _split(NPI_14, pending=2)
    assert (split.workers, split.threads) == (2, 16)


def test_auto_split_keeps_small_jobs_in_one_process_on_one_memory_domain():
    tiny = {"steps": 50, "flops": 10**6, "peak_bytes": 10**6}
    split = _split(tiny, pending=4)
    assert (split.workers, split.threads) == (1, 16)
    assert "one process" in split.reason
    assert (_split(NN_10, pending=1).workers, _split(NN_10, pending=1).threads) == (1, 16)


def test_auto_split_without_fork_or_memory_information():
    split = _split(PIPI_18, can_fork=False)
    assert (split.workers, split.threads) == (1, None)
    assert _split(PIPI_18, available_bytes=None).workers == 64
    assert _split(NN_10, available_bytes=5 * GIB).workers == 1


def test_auto_split_with_fixed_threads():
    split = _split(PIPI_18, threads=2)
    assert (split.workers, split.threads) == (32, 2)
    assert _split(NPI_14, threads=8, pending=3).threads == 8


def test_workers_in_memory_counts_down_from_the_limit():
    assert workers_in_memory(GIB, 64 * GIB, MARGIN, 8) == 8
    assert workers_in_memory(GIB, 64 * GIB, MARGIN, 100) == 21
    assert workers_in_memory(10 * GIB, 8 * GIB, MARGIN, 4) == 0


def test_workers_for_peak_keeps_growth_room_and_a_node_reserve():
    # 5 % of 100 GiB is kept free; each worker needs 1.1 x its measured peak.
    assert workers_for_peak(10 * GIB, 100 * GIB, 64) == 8  # 95 // 11
    assert workers_for_peak(GIB, 100 * GIB, 16) == 16
    # The reserve is at least 2 GiB.
    assert workers_for_peak(GIB, 10 * GIB, 64) == 7  # 8 // 1.1
    assert workers_for_peak(20 * GIB, 10 * GIB, 64) == 0


# The N pi plan at N_v = 48 with auto-hq: 11.16 GiB of buffers per
# time slice, 16 x 4 by the margin rule; its workers peaked at 13.5 GiB of own
# memory each, and the node at 99 %.
NPI_48 = {"steps": 23_716, "flops": 1_393_263_478_688, "peak_bytes": int(11.16 * GIB)}


def test_auto_split_measures_first_when_the_rough_allowance_does_not_fit_all_workers():
    split = _split(NPI_48, pending=32, available_bytes=232 * GIB)
    assert (split.workers, split.threads) == (16, 4)
    # 220 GiB usable / (1.1 x (11.16 + 3) GiB) = 14.1
    assert split.measure_with == 14
    assert "memory measured first by 14 workers (14 fit with 3.0 GiB each" in split.reason

    measured = _split(
        NPI_48, pending=18, available_bytes=232 * GIB, measured_peak_bytes=int(13.46 * GIB)
    )
    assert (measured.workers, measured.threads, measured.measure_with) == (14, 4, 0)
    assert "14 by measured memory, 13.5 GiB each" in measured.reason


def test_auto_split_starts_all_workers_when_the_rough_allowance_fits_or_the_measurement_allows():
    # bbūd̄ at N_v = 16: 0.05 GiB of buffers; 63 workers fit with 3 GiB each beyond them.
    small = {"steps": 160_675, "flops": 105_710_291_456, "peak_bytes": int(0.05 * GIB)}
    split = _split(small, pending=63, available_bytes=238 * GIB)
    assert (split.workers, split.threads, split.measure_with) == (63, 1, 0)

    # ππ at N_v = 32: 1.24 GiB of buffers; the rough allowance fits only 48 of
    # 63 one-thread workers, the measured 2.7 GiB per worker fits all of them.
    pipi_32 = {**small, "peak_bytes": int(1.24 * GIB)}
    split = _split(pipi_32, pending=63, available_bytes=238 * GIB)
    assert (split.workers, split.measure_with) == (63, 48)
    measured = _split(
        pipi_32, pending=200, available_bytes=238 * GIB, measured_peak_bytes=int(2.7 * GIB)
    )
    assert (measured.workers, measured.threads) == (64, 1)
    assert "64 by measured memory" in measured.reason
    # Fewer pending units than cores: the idle cores become threads, as without measuring.
    last = _split(pipi_32, pending=15, available_bytes=238 * GIB, measured_peak_bytes=int(2.7 * GIB))
    assert (last.workers, last.threads) == (15, 4)
