from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator


_STRICT_EQUALITY = False


class DCValue(Enum):
    """Three-valued value used for incomplete observations.

    ``TRUE`` and ``FALSE`` represent known Boolean values, while ``DC``
    represents a don't-care value. When comparing two ``DCValue`` instances,
    ``DC`` is considered equal to either known value.

    Note:
        Equality involving ``DC`` is intentionally non-standard: both
        ``DCValue.DC == DCValue.TRUE`` and
        ``DCValue.DC == DCValue.FALSE`` evaluate to ``True``.
        Consequently, this type should not be used as a normal key in
        dictionaries or as a member of sets.
    """

    TRUE = True
    FALSE = False
    DC = None

    def is_known(self) -> bool:
        """Return whether this value represents a known Boolean value."""
        return self is not DCValue.DC

    def __eq__(self, other: Any) -> bool:
        """Compare this value with another :class:`DCValue`."""
        if not isinstance(other, DCValue):
            raise TypeError(
                f"Cannot compare DCValue with {type(other)}"
            )

        if _STRICT_EQUALITY:
            return self.value == other.value

        if not self.is_known() or not other.is_known():
            return True

        return self.value == other.value

    @classmethod
    @contextmanager
    def strict_equality(cls) -> Iterator[None]:
        """Temporarily use ordinary equality instead of DC equality."""
        global _STRICT_EQUALITY

        old_value = _STRICT_EQUALITY
        _STRICT_EQUALITY = True
        try:
            yield
        finally:
            _STRICT_EQUALITY = old_value

    def __hash__(self) -> int:
        """Return a hash value for this instance."""
        if self.is_known():
            return hash(self.value)

        return 0
