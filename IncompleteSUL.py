"""System under learning backed by a finite, partial set of labelled words."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from DCValue import DCValue


class IncompleteSUL:
    """Finite partial specification used as an incomplete system under learning.

    Words explicitly contained in the dataset are assigned ``TRUE`` or
    ``FALSE``. All other words return ``DCValue.DC``.

    Attributes:
        words: Known value of each word; looking up any other word yields
            ``DCValue.DC``.
        num_queries: Number of calls to :meth:`query`.
        num_steps: Total length of all queried words.
    """

    def __init__(self, words: list[tuple[tuple[Any, ...], bool]]) -> None:
        """Initialize the incomplete system.

        Args:
            words: Input words paired with their expected acceptance result.
                ``True`` denotes acceptance and ``False`` denotes rejection.
                Words not present in the dataset return ``DCValue.DC``.
        """
        self.words: defaultdict[tuple[Any, ...], DCValue] = defaultdict(lambda: DCValue.DC)

        for word, value in words:
            self.words[word] = DCValue.TRUE if value else DCValue.FALSE

        self.num_queries = 0
        self.num_steps = 0

    def query(self, word: tuple[Any, ...]) -> DCValue:
        """Return the known value of an input word.

        Args:
            word: Input word represented as a tuple of symbols.

        Returns:
            ``DCValue.TRUE`` or ``DCValue.FALSE`` for known words and
            ``DCValue.DC`` otherwise.
        """
        self.num_queries += 1
        self.num_steps += len(word)
        return self.words[word]
