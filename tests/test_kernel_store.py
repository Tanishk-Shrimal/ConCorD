"""Focused tests for persistent numerical kernel entries."""

from __future__ import annotations


import h5py
import numpy as np
import pytest

from concord_lqcd.job.kernel_store import HDF5KernelStore


def _entry(store: HDF5KernelStore, shape: tuple[int, ...], axes: tuple[str, ...]):
    manifest = {
        "configuration": store.configuration,
        "shape": list(shape),
        "axes": list(axes),
        "scalar_type": "complex128",
        "recipe_signature": '{"momentum":[0,0,0]}',
    }
    manifest_json = store.canonical_manifest(manifest)
    return store.entry_key(manifest_json), manifest_json


def test_kernel_store_round_trip_and_validates_metadata_and_checksum(tmp_path):
    store = HDF5KernelStore(tmp_path / "kernels", configuration=17)
    values = np.arange(6, dtype=np.float64).reshape(2, 3).astype(np.complex128)
    axes = ("mode_0", "mode_1")
    key, manifest = _entry(store, values.shape, axes)

    assert store.store_array(key, manifest, values, axes=axes)
    actual = store.load_array(
        key,
        manifest,
        expected_shape=values.shape,
        expected_dtype=np.dtype(np.complex128),
        expected_axes=axes,
        expected_scalar_type="complex128",
    )
    np.testing.assert_array_equal(actual, values)
    assert store.statistics.hits == 1
    assert store.statistics.writes == 1

    with pytest.warns(RuntimeWarning, match="shape mismatch"):
        mismatched = store.load_array(
            key,
            manifest,
            expected_shape=(3, 2),
            expected_dtype=np.dtype(np.complex128),
            expected_axes=axes,
            expected_scalar_type="complex128",
        )
    assert mismatched is None
    assert store.statistics.misses == 1
    assert store.statistics.invalid_entries == 1

    with h5py.File(store.entry_path(key), "r+") as handle:
        handle["values"][0, 0] = 99.0 + 1.0j
    with pytest.warns(RuntimeWarning, match="checksum mismatch"):
        corrupt = store.load_array(
            key,
            manifest,
            expected_shape=values.shape,
            expected_dtype=np.dtype(np.complex128),
            expected_axes=axes,
            expected_scalar_type="complex128",
        )
    assert corrupt is None
    assert store.statistics.misses == 2
    assert store.statistics.invalid_entries == 2


def test_kernel_store_preserves_scalar_tensor_shape(tmp_path):
    store = HDF5KernelStore(tmp_path / "kernels", configuration=3)
    values = np.asarray(2.5 - 1.25j, dtype=np.complex128)
    axes: tuple[str, ...] = ()
    key, manifest = _entry(store, values.shape, axes)

    assert store.store_array(key, manifest, values, axes=axes)
    actual = store.load_array(
        key,
        manifest,
        expected_shape=(),
        expected_dtype=np.dtype(np.complex128),
        expected_axes=(),
        expected_scalar_type="complex128",
    )
    assert actual is not None
    assert actual.shape == ()
    assert actual.item() == values.item()


@pytest.mark.parametrize("scalar_type", ["complex64", "complex128"])
@pytest.mark.parametrize("packing", ["none", "antisym3"])
def test_antisym3_storage_requires_explicit_policy_and_recipe_eligibility(tmp_path, scalar_type, packing):
    store = HDF5KernelStore(tmp_path / "kernels", configuration=5, packing=packing)
    values = np.zeros((4, 4, 4), dtype=scalar_type)
    values[0, 1, 2] = values[1, 2, 0] = values[2, 0, 1] = 1 + 2j
    values[0, 2, 1] = values[2, 1, 0] = values[1, 0, 2] = -1 - 2j
    axes = ("mode_0", "mode_1", "mode_2")
    key, manifest = _entry(store, values.shape, axes)
    for eligible in (False, True):
        assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=eligible)
        with h5py.File(store.entry_path(key), "r") as handle:
            expected = "antisym3" if eligible and packing == "antisym3" else "none"
            assert handle["values"].attrs["storage_packing"] == expected
        actual = store.load_array(key, manifest, expected_shape=values.shape,
                                  expected_dtype=values.dtype, expected_axes=axes,
                                  expected_scalar_type=scalar_type)
        np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("value", [np.inf + 0j, np.nan + 0j, 1e308 + 1e308j])
def test_packing_does_not_mask_nonfinite_or_overflowing_values(tmp_path, value):
    store = HDF5KernelStore(tmp_path, configuration=0, packing="antisym3")
    values = np.zeros((3, 3, 3), dtype=np.complex128)
    values[0, 0, 0] = value
    axes = ("mode_0", "mode_1", "mode_2")
    key, manifest = _entry(store, values.shape, axes)
    with np.errstate(over="ignore", invalid="ignore"):
        assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=True)
    with h5py.File(store.entry_path(key), "r") as handle:
        assert handle["values"].attrs["storage_packing"] == "none"
    actual = store.load_array(key, manifest, expected_shape=values.shape,
                              expected_dtype=values.dtype, expected_axes=axes,
                              expected_scalar_type="complex128")
    np.testing.assert_array_equal(actual, values)


def test_packed_checksum_metadata_and_policy_are_validated(tmp_path):
    store = HDF5KernelStore(tmp_path, configuration=0, packing="antisym3")
    values = np.zeros((3, 3, 3), dtype=np.complex128)
    axes = ("mode_0", "mode_1", "mode_2")
    key, manifest = _entry(store, values.shape, axes)
    assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=True)
    kwargs = dict(expected_shape=values.shape, expected_dtype=values.dtype,
                  expected_axes=axes, expected_scalar_type="complex128")
    dense_store = HDF5KernelStore(tmp_path, configuration=0)
    with pytest.warns(RuntimeWarning, match="not enabled"):
        assert dense_store.load_array(key, manifest, **kwargs) is None
    with h5py.File(store.entry_path(key), "r+") as handle:
        handle["values"][0] = 1
    with pytest.warns(RuntimeWarning, match="checksum mismatch"):
        assert store.load_array(key, manifest, **kwargs) is None
    assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=True)
    with h5py.File(store.entry_path(key), "r+") as handle:
        handle["values"].attrs["shape_json"] = "[4,4,4]"
    with pytest.warns(RuntimeWarning, match="recorded tensor shape"):
        assert store.load_array(key, manifest, **kwargs) is None


@pytest.mark.parametrize("packing", ["none", "antisym3"])
def test_only_opt_in_packing_projects_roundoff_residues(tmp_path, packing):
    store = HDF5KernelStore(tmp_path, configuration=0, packing=packing)
    values = np.zeros((3, 3, 3), dtype=np.complex128)
    values[0, 1, 2] = values[1, 2, 0] = values[2, 0, 1] = 1
    values[0, 2, 1] = values[2, 1, 0] = values[1, 0, 2] = -1
    values[0, 0, 0] = 5e-12
    axes = ("mode_0", "mode_1", "mode_2")
    key, manifest = _entry(store, values.shape, axes)
    assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=True)
    actual = store.load_array(key, manifest, expected_shape=values.shape,
                              expected_dtype=values.dtype, expected_axes=axes,
                              expected_scalar_type="complex128")
    if packing == "none":
        np.testing.assert_array_equal(actual, values)
    else:
        assert actual[0, 0, 0] == 0


def test_packing_scratch_is_blocked_and_does_not_reconstruct_a_dense_tensor(tmp_path, monkeypatch):
    from concord_lqcd.job import kernel_store

    values = np.zeros((33, 33, 33), dtype=np.complex128)
    store = HDF5KernelStore(tmp_path, configuration=0, packing="antisym3")
    axes = ("mode_0", "mode_1", "mode_2")
    key, manifest = _entry(store, values.shape, axes)
    monkeypatch.setattr(kernel_store, "_unpack_antisym3", lambda *args: pytest.fail("packing reconstructed a dense tensor"))
    assert store.store_array(key, manifest, values, axes=axes, allow_antisym3=True)
    sizes = [first.size for first, second, third in kernel_store._antisym3_blocks(33)]
    assert sum(sizes) == 5456
    assert max(sizes) <= kernel_store._ANTISYM3_BLOCK_ENTRIES


@pytest.mark.parametrize("packing", ["gzip", "automatic", True, None])
def test_unknown_packing_policy_is_rejected(tmp_path, packing):
    with pytest.raises(ValueError, match="packing"):
        HDF5KernelStore(tmp_path, configuration=0, packing=packing)
