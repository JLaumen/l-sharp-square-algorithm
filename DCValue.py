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
        ``DCValue.DC == DCValue.FALSE`` evaluate to ``True``. Consequently,
        this type should not be used as a normal key in dictionaries or as a
        member of sets.
    """

    TRUE = True
    FALSE = False
    DC = None

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
