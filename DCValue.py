"""Three-valued outputs for observations from an incomplete teacher."""

from __future__ import annotations

from enum import Enum
from typing import Any


class DCValue(Enum):
    """Three-valued value used for incomplete observations.

    ``TRUE`` and ``FALSE`` represent known Boolean values, while ``DC``
    represents a don't-care value. When comparing two ``DCValue`` instances,
    ``DC`` is considered equal to either known value.

    Note:
        Equality involving ``DC`` is intentionally non-standard: both
        ``DCValue.DC == DCValue.TRUE`` and
        ``DCValue.DC == DCValue.FALSE`` evaluate to ``True``. Because this
        relation is not transitive, the type is deliberately unhashable and
        cannot be used as a dictionary key or set member. Compare members
        with ``is`` when the exact value matters.
    """

    TRUE = True
    FALSE = False
    DC = None

    # Defining ``__eq__`` already removes the inherited hash; this makes it explicit.
    __hash__ = None  # type: ignore[assignment]

    def is_known(self) -> bool:
        """Return whether this value represents a known Boolean value.

        Returns:
            ``True`` for :attr:`TRUE` and :attr:`FALSE`, and ``False`` for
            :attr:`DC`.
        """
        return self is not DCValue.DC

    def __eq__(self, other: Any) -> bool:
        """Compare this value with another :class:`DCValue`.

        A don't-care value is considered equal to every ``DCValue``. Two
        known values are equal exactly when their underlying Boolean values
        are equal.

        Args:
            other: The value to compare with.

        Returns:
            ``True`` if either value is ``DC``, or if both values are known
            and equal.

        Raises:
            TypeError: If ``other`` is not a ``DCValue``.
        """
        if not isinstance(other, DCValue):
            raise TypeError(f"Cannot compare DCValue with {type(other)}")

        if not self.is_known() or not other.is_known():
            return True

        return self.value == other.value
