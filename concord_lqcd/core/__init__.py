"""
ConCorD core package.
"""

# ── Index Registry & Types ──────────────────────────────────────────────
from .indices import (
    Index,
    IndexRegistry,
    IndexConflictError,
    get_registry,
    reset_registry,
    SPIN,
    COLOR,
    POSITION,
)

# ── Expression Base & Products ──────────────────────────────────────────
from .expr import (
    Expr,
    Product,
)

# ── Field Leaves ────────────────────────────────────────────────────────
from .fields import (
    QuarkField,
    u, ubar,
    d, dbar,
    l, lbar,
    s, sbar,
    c, cbar,
    b, bbar,
    t, tbar,
)

# ── Gamma Matrices ──────────────────────────────────────────────────────
from .gamma import (
    GammaMatrix,
    gamma5,
    gammaX,
    gammaY,
    gammaZ,
    gammaT,
    gamma_identity,
    spin_matrix,
    charge_conjugation,
    cgamma5,
)
from .dirac_basis import SpinBasis, get_spin_matrices

# ── Color Tensors ───────────────────────────────────────────────────────
from .color import (
    ColorDelta,
    ColorEpsilon,
    delta,
    epsilon,
)

# ── Linear Operators & Applications ───────────────────────────────────
from .operations import (
    Space,
    Locality,
    Axis,
    FieldType,
    Path,
    LinearOperator,
    Shift,
    ParallelTransport,
    CovariantDerivative,
    Projection,
    Apply,
)

# ── Operator Wrapper & Derivative / Transport Helpers ──────────────────
from .operator import (
    Operator,
    derivative,
    transport,
    shift,
    projected,
    laph,
    point,
)

# ── Operator Linear Combinations ───────────────────────────────────────
from .operator_expression import (
    OperatorLike,
    OperatorCombination,
)

# ── Baryon helpers ─────────────────────────────────────────────────────
from .baryon import build_baryon_operator

# ── Validation ─────────────────────────────────────────────────────────
from .validate import validate

# ── Correlators & Correlation Matrices ────────────────────────────────
from .correlator import (
    Correlator,
    CorrMatrix,
)

# ── Gauge Links ────────────────────────────────────────────────────────
from .gauge import (
    GaugeLink,
    link,
)

__all__ = [
    # Indices
    "Index", "IndexRegistry", "IndexConflictError", "get_registry", "reset_registry",
    "SPIN", "COLOR", "POSITION",
    # Expressions
    "Expr", "Product",
    # Fields
    "QuarkField", "u", "ubar", "d", "dbar", "l", "lbar", "s", "sbar",
    "c", "cbar", "b", "bbar", "t", "tbar",
    # Gammas
    "GammaMatrix", "gamma5", "gammaX", "gammaY", "gammaZ", "gammaT",
    "gamma_identity", "spin_matrix", "charge_conjugation", "cgamma5",
    "SpinBasis", "get_spin_matrices",
    # Color
    "ColorDelta", "ColorEpsilon", "delta", "epsilon",
    # Operations
    "Space", "Locality", "Axis", "FieldType", "Path", "LinearOperator",
    "Shift", "ParallelTransport", "CovariantDerivative", "Projection", "Apply",
    # Operators
    "Operator", "derivative", "transport", "shift", "projected", "laph", "point",
    # Operator Linear Algebra
    "OperatorLike", "OperatorCombination",
    # Baryon
    "build_baryon_operator",
    # Validation
    "validate",
    # Correlators
    "Correlator", "CorrMatrix",
    # Gauge
    "GaugeLink", "link",
]
