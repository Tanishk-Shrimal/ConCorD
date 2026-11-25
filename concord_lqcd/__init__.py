"""
ConCorD — Symbolic compiler for lattice QCD correlation functions.

Usage::

    from concord_lqcd import *

    pion = Operator(
        dbar('a', 'α') * gamma5('α', 'β') * u('a', 'β'),
        momentum=(0, 0, 1),
    )
"""

from __future__ import annotations

__version__ = "1.0.0"

# The package root is the symbolic frontend boundary. Compilation and runtime
# entry points live in their own staged packages and are imported explicitly.
from . import core as _core

for _name in _core.__all__:
    globals()[_name] = getattr(_core, _name)


__all__ = list(_core.__all__)
