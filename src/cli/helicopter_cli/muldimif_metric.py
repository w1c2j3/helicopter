"""Official MulDimIF constraint evaluation adapter.

The checker classes are vendored unchanged from Junjie-Ye/MulDimIF.  This
module only maps the constraint labels present in the official test split to
their corresponding evaluator and exposes a small, typed API for LightEval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .vendor.muldimif.scripts import (
    Content_Keywords,
    Content_Others,
    Content_Punctuation,
    Format_Json,
    Format_Markdown,
    Format_Others,
    Format_Table,
    Language_Chinese,
    Language_English,
    Length_Paragraphs,
    Length_Sentences,
    Length_Words,
)


_CONTENT_KEYWORDS = Content_Keywords()
_CONTENT_OTHERS = Content_Others()
_CONTENT_PUNCTUATION = Content_Punctuation()
_FORMAT_JSON = Format_Json()
_FORMAT_MARKDOWN = Format_Markdown()
_FORMAT_OTHERS = Format_Others()
_FORMAT_TABLE = Format_Table()
_LANGUAGE_CHINESE = Language_Chinese()
_LANGUAGE_ENGLISH = Language_English()
_LENGTH_PARAGRAPHS = Length_Paragraphs()
_LENGTH_SENTENCES = Length_Sentences()
_LENGTH_WORDS = Length_Words()


# Exact labels observed in the official 1,200-row Data/test.json split.
_CHECKERS = {
    "Content_Identifiers": _CONTENT_OTHERS,
    "Content_Keywords": _CONTENT_KEYWORDS,
    "Content_Keywords: Must include": _CONTENT_KEYWORDS,
    "Content_Punctuation": _CONTENT_PUNCTUATION,
    "Content_Punctuation: Ending punctuation": _CONTENT_PUNCTUATION,
    "Format_Blurb": _FORMAT_OTHERS,
    "Format_Json": _FORMAT_JSON,
    "Format_Markdown": _FORMAT_MARKDOWN,
    "Format_Markdown: Heading levels": _FORMAT_MARKDOWN,
    "Format_Table": _FORMAT_TABLE,
    "Format_Table: Column limit": _FORMAT_TABLE,
    "Format_Table: Row limit": _FORMAT_TABLE,
    "Format_Text": _FORMAT_OTHERS,
    "Format_XML": _FORMAT_OTHERS,
    "Json_Object nesting levels": _FORMAT_JSON,
    "Language_Chinese": _LANGUAGE_CHINESE,
    "Language_Chinese: Traditional": _LANGUAGE_CHINESE,
    "Language_English": _LANGUAGE_ENGLISH,
    "Language_English: All Uppercase": _LANGUAGE_ENGLISH,
    "Language_English: Capitalized": _LANGUAGE_ENGLISH,
    "Length_Paragraphs": _LENGTH_PARAGRAPHS,
    "Length_Paragraphs: At most": _LENGTH_PARAGRAPHS,
    "Length_Sentences": _LENGTH_SENTENCES,
    "Length_Sentences: At least": _LENGTH_SENTENCES,
    "Length_Sentences: At most": _LENGTH_SENTENCES,
    "Length_Words": _LENGTH_WORDS,
    "Length_Words: At least": _LENGTH_WORDS,
    "Length_Words: At most": _LENGTH_WORDS,
    "Length_Words: Range": _LENGTH_WORDS,
    "Markdown_Block quotes": _FORMAT_MARKDOWN,
    "Markdown_Heading levels": _FORMAT_MARKDOWN,
    "Punctuation_Ending punctuation": _CONTENT_PUNCTUATION,
    "Table_Column limit": _FORMAT_TABLE,
    "Table_Row limit": _FORMAT_TABLE,
    "Table_Table": _FORMAT_OTHERS,
    "XML_Number of attributes": _FORMAT_OTHERS,
}


def constraint_key(constraint: Sequence[Any]) -> str:
    if len(constraint) < 3:
        raise ValueError(f"MulDimIF constraint must have at least 3 fields: {constraint!r}")
    return f"{constraint[0]}_{constraint[1]}"


def evaluate_constraints(
    constraints: Sequence[Sequence[Any]],
    response: str,
) -> list[bool]:
    """Return the official per-constraint verdicts for one model response."""

    verdicts: list[bool] = []
    for constraint in constraints:
        key = constraint_key(constraint)
        checker = _CHECKERS.get(key)
        if checker is None:
            raise ValueError(f"unsupported official MulDimIF constraint: {key!r}")
        verdicts.append(bool(checker.check(str(constraint[-1]), str(response))))
    return verdicts


def official_constraint_keys() -> frozenset[str]:
    return frozenset(_CHECKERS)


__all__ = [
    "constraint_key",
    "evaluate_constraints",
    "official_constraint_keys",
]
