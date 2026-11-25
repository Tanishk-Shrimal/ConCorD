"""Preflight explicit flavor equality against the built-in source catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from concord_lqcd.compiler.numerical_flavors import normalize_numerical_flavor_groups

if TYPE_CHECKING:
    from .schema import JobSpec


def validate_numerical_flavor_sources(
    spec: JobSpec, *, configuration: int | None = None,
) -> None:
    """Require identical configured ordinary tensors for every declared flavor.

    Call before compilation and before execution, not just for resources left
    in an optimized plan. With no declaration this is a no-op. By default all
    configured configurations are checked; a runtime caller may select one.

    The current JobSpec shares axes, fixed-source time, source-field convention,
    spin basis/size, scalar type, mode truncation, time contexts, lattice, and
    the declared orthonormal LapH basis across flavors. Only file selection and
    dataset formatting depend on flavor. Their complete configured identities
    must agree, including unresolved runtime time placeholders. No tensor is
    opened; equal masses or merely similar filenames do not establish source
    identity, and distinct files are never presumed equal.

    Custom bases and opaque generalized resolvers cannot certify this contract.
    Reject a configured generalized resolver conservatively, even if a later
    compiler pass would discard its currents or find them unrelated to a group.
    """
    from .schema import HDF5EigenvectorSpec, JobSpec

    if not isinstance(spec, JobSpec):
        raise ValueError("spec must be JobSpec")
    groups = normalize_numerical_flavor_groups(
        getattr(spec.compile, "numerical_flavor_groups", ()),
    )
    if not groups:
        return
    if not isinstance(spec.data.eigenvectors, HDF5EigenvectorSpec):
        raise ValueError(
            "numerical flavor equivalence requires the built-in shared LapH basis adapter"
        )
    if spec.data.propagation.generalized is not None:
        raise ValueError(
            "numerical flavor equivalence does not certify generalized current sources; "
            "a generalized resolver requires an independent equality contract"
        )

    from .data import _format_template, _partial_format_template

    ordinary = spec.data.propagation.ordinary
    if "configuration" in ordinary.axes:
        raise ValueError(
            "the built-in ordinary HDF5 catalog does not infer a configuration-axis "
            "row from a configuration id"
        )
    shared_configuration = {
        "provider": spec.data.propagation.provider,
        "ordinary": ordinary.model_dump(exclude={"path", "files", "dataset"}),
        "basis": spec.data.eigenvectors.model_dump(),
        "lattice": spec.lattice.model_dump(),
        "mode_size": spec.compile.mode_size,
        "spin_size": spec.compile.spin_size,
        "spin_basis": spec.compile.spin_basis,
        "scalar_type": spec.compile.scalar_type,
        "times": spec.run.times.model_dump(),
    }
    configurations = spec.configurations if configuration is None else (configuration,)
    for current_configuration in configurations:
        for group in groups:
            identities = []
            for flavor in group:
                values = {"configuration": current_configuration, "flavor": flavor}
                if ordinary.path is not None:
                    template = ordinary.path
                else:
                    if ordinary.files is None or flavor not in ordinary.files:
                        raise ValueError(
                            "numerical flavor group lacks an ordinary source for "
                            f"exact flavor {flavor!r}"
                        )
                    template = ordinary.files[flavor]
                location = _format_template(
                    template, values, role="data.propagation.ordinary.path",
                )
                dataset = _partial_format_template(
                    ordinary.dataset, values, role="data.propagation.ordinary.dataset",
                )
                identities.append((location, dataset, shared_configuration))
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError(
                    f"declared numerical flavor group {group!r} resolves to different "
                    f"ordinary HDF5 tensors for configuration {current_configuration!r}; "
                    "file, dataset, axes, metadata, basis, time, spin, and conventions "
                    "must agree; equal masses alone do not establish tensor equality"
                )


__all__ = ["validate_numerical_flavor_sources"]
