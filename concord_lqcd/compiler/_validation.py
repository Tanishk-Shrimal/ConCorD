"""Object-local evidence for static metadata owned by plan preparation.

Only preparation records this evidence, after copying and fully validating the
read-only graph. There is no global cache or persisted trust flag. A copied
receipt does not certify another object, even if its fields happen to match.
Runtime data, binding contexts and numerical results are never certified here.
"""

from dataclasses import dataclass
from weakref import ReferenceType, ref


_RECEIPT_ATTRIBUTE = "_concord_static_validation"


@dataclass(frozen=True, slots=True)
class _StaticValidationReceipt:
    owner: ReferenceType


def _has_static_validation(owner: object) -> bool:
    receipt = getattr(owner, _RECEIPT_ATTRIBUTE, None)
    return type(receipt) is _StaticValidationReceipt and receipt.owner() is owner


def _record_static_validation(owner: object) -> None:
    """Record a completed check on an owned, immutable preparation node."""

    object.__setattr__(owner, _RECEIPT_ATTRIBUTE, _StaticValidationReceipt(ref(owner)))
