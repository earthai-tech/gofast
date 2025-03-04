# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
This module provides various text-related
utilities such as cleaning, normalization,
tokenization, anonymization, and more.
Each function serves a specific role in
manipulating or analyzing textual data.
"""

import re
import unicodedata
import string
from collections import Counter
from typing import List, Dict, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..core.checks import assert_ratio, check_non_emptiness 
from ..core.io import to_text
from ..decorators import isdf 

__all__ = [
    "clean_text",
    "normalize_text",
    "to_title_case",
    "to_camel_case",
    "to_snake_case",
    "capitalize_first",
    "tokenize",
    "text_analysis",
    "substring_pattern_match",
    "slugify",
    "reverse_text",
    "truncate",
    "extract_sentences",
    "encode_text",
    "decode_text",
    "levenshtein_distance",
    "is_palindrome",
    "anonymize_text", 
    "select_text_columns", 
    "select_text_columns_in", 
    "vectorize_text_columns", 
    "tokenize_text_columns"
    
]
   
@to_text
def clean_text(
    text: str,
    *,
    remove_punctuation: bool = True,
    punctuation: str = string.punctuation,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
    remove_digits: bool = False,
    extra_chars_to_remove: str = ""
) -> str:
    """
    Clean the input `text` by removing extraneous
    characters and formatting it using the
    ``clean_text`` transformation.

    This function applies a sequence of text
    processing steps such as removing punctuation,
    optional removal of digits, case conversion,
    and whitespace normalization [1]_. It is
    particularly useful for preparing textual data
    for machine learning pipelines, tokenization,
    or general text analysis.

    .. math::
       \\
       \\text{cleaned_text} = f(\\text{input_text})
       =
       \\underbrace{\\mathrm{Collapse}\\bigl(\\mathrm{Strip}(
       \\mathrm{CaseConvert}(\\mathrm{RemovePunct}(
       \\mathrm{RemoveDigits}(\\text{input_text})
       )))\\bigr)}_{\\mathrm{clean\\_text}\\,\\mathrm{pipeline}}

    Parameters
    ----------
    text : str
        The input text to be cleaned. This is the
        primary `parameter inline` consumed by
        ``clean_text``.
    remove_punctuation : bool
        If `remove_punctuation` is True, all
        punctuation characters defined by
        `punctuation` are removed.
    punctuation : str
        A string of punctuation characters to be
        removed from `text`. By default, this uses
        the standard set in Python's
        `string.punctuation`.
    lowercase : bool
        If `lowercase` is True, the resulting text
        is converted entirely to lowercase.
    strip_whitespace : bool
        If `strip_whitespace` is True, leading
        and trailing whitespace are stripped
        from the text.
    collapse_whitespace : bool
        If `collapse_whitespace` is True, multiple
        spaces are collapsed into a single space.
    remove_digits : bool
        If `remove_digits` is True, all digits
        (0-9) are removed from the text.
    extra_chars_to_remove : str
        Additional characters to remove from
        `text`, beyond those specified in
        `punctuation`.

    Returns
    -------
    str
        A cleaned and formatted version of the
        input `text`.

    Notes
    -----
    1. The function `clean_text` does not
       normalize Unicode characters (e.g., it does
       not remove diacritics). For advanced
       normalization steps, a separate approach
       may be required.
    2. Internally, `clean_text` chains multiple
       transformations to produce its final
       output.

    Examples
    --------
    >>> from gofast.utils.text import clean_text
    >>> sample_text = ``"Hello, World! 123"``  # double backticks
    >>> cleaned = clean_text(sample_text,
    ...                      remove_punctuation=True,
    ...                      remove_digits=True)
    >>> print(cleaned)
    hello world

    See Also
    --------
    :func:`clean_text` : The core function for
        text cleaning. (Listed here to illustrate
        reST linking of inline methods that do
        not start with `_`.)

    References
    ----------
    .. [1] Smith, J. and Doe, A., *Text Processing
           Techniques*, Journal of Text Analytics,
           2025.

    """
    # Optionally strip leading/trailing whitespace.
    if strip_whitespace:
        text = text.strip()

    # Optionally convert to lowercase.
    if lowercase:
        text = text.lower()

    # Optionally remove punctuation.
    if remove_punctuation:
        # Build a translation table for punctuation
        # plus any extra characters to remove.
        all_chars = punctuation + extra_chars_to_remove
        translator = str.maketrans('', '', all_chars)
        text = text.translate(translator)

    # Optionally remove digits.
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Optionally collapse multiple whitespace
    # characters into a single space.
    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text)

    return text

@to_text
def normalize_text(
    text: str,
    *,
    normalization_form: str = "NFKC",
    remove_accents: bool = True,
    additional_normalizations: Optional[dict] = None,
    to_lower: bool = False,
    strip_extra_spaces: bool = False
) -> str:
    """
    Normalize the input ``text`` using Unicode
    normalization and optionally remove diacritical
    marks (accents) and perform extra text
    transformations using ``normalize_text``.

    .. math::
       \\text{normalized\\_text} = g(
       \\mathrm{RemoveAccents}(
       \\mathrm{UnicodeNormalize}(\\text{text})))
       \\quad + \\dots

    Parameters
    ----------
    text : str
        The input `parameter inline` string to be
        processed by ``normalize_text``.
    normalization_form : str
        The Unicode normalization form to apply.
        Common values include "NFC", "NFD", "NFKC",
        and "NFKD". Defaults to "NFKC".
    remove_accents : bool
        If `remove_accents` is True, diacritical
        marks are removed from the text by
        decomposing characters and discarding those
        of category "Mn".
    additional_normalizations : dict, optional
        A dictionary of additional custom
        replacements to apply after normalization.
        For instance, {"–": "-", "“": '"',
        "”": '"'}.
    to_lower : bool
        If `to_lower` is True, convert the text to
        lowercase after normalization.
    strip_extra_spaces : bool
        If `strip_extra_spaces` is True, collapse
        consecutive whitespace into a single space
        and strip leading/trailing spaces.

    Returns
    -------
    str
        The fully normalized and optionally
        accent-stripped, lowercased, and
        whitespace-collapsed string.

    Notes
    -----
    1. The default normalization form "NFKC" often
       applies compatibility decomposition followed
       by canonical composition [1]_.
    2. Removing accents uses an NFD pass internally
       and discards characters of category "Mn".

    Examples
    --------
    >>> from gofast.utils.text import normalize_text
    >>> raw_text = ``"Café – with Accent"``   # double backticks
    >>> result = normalize_text(raw_text,
    ...                         normalization_form="NFKC",
    ...                         remove_accents=True,
    ...                         additional_normalizations={"–": "-"},
    ...                         to_lower=True,
    ...                         strip_extra_spaces=True)
    >>> print(result)
    cafe - with accent

    See Also
    --------
    :func:`normalize_text` : Demonstrates how to
        do advanced text normalizations and
        diacritic removal.

    References
    ----------
    .. [1] The Unicode Consortium, "Unicode
           Standard Annex #15: Unicode
           Normalization Forms", 2025.

    """
    # Apply Unicode normalization.
    normalized_text = unicodedata.normalize(
        normalization_form, text
    )

    # Optionally remove accents (diacritical marks).
    if remove_accents:
        # Decompose into base + diacritical marks,
        # then remove marks (category "Mn").
        normalized_text = "".join(
            c for c in unicodedata.normalize(
                "NFD", normalized_text
            )
            if unicodedata.category(c) != "Mn"
        )

    # Apply any additional custom normalizations.
    if additional_normalizations:
        for target, replacement in additional_normalizations.items():
            normalized_text = normalized_text.replace(
                target,
                replacement
            )

    # Optionally lowercase the result.
    if to_lower:
        normalized_text = normalized_text.lower()

    # Optionally strip and collapse spaces.
    if strip_extra_spaces:
        normalized_text = re.sub(
            r"\s+",
            " ",
            normalized_text
        ).strip()

    return normalized_text

@to_text(allow_none= True)
def to_title_case(
    text: str,
    *,
    ex: Optional[List[str]] = None,
    all_cap: bool = False,
    delim: Optional[str] = None
) -> str:
    """
    Convert a string ``text`` to title case with
    flexible options using ``to_title_case``.

    .. math::
       \\text{title\\_case}(\\text{word})
       = \\mathrm{CapitalizeIfNeeded}(
       \\mathrm{word})

    Parameters
    ----------
    text : str
        The input `parameter inline` string to be
        processed by ``to_title_case``.
    ex : list of str, optional
        A list of words in lowercase that should
        remain in lowercase if they are not the
        first or last word. By default, common
        articles and prepositions (e.g., "and",
        "or", "the", etc.) are used.
    all_cap : bool
        If `all_cap` is True, ignore exclusions
        and capitalize every word in the string.
    delim : str, optional
        A custom delimiter for splitting the text.
        If not provided, the text is split on
        whitespace.

    Returns
    -------
    str
        A title-cased string, respecting the
        optional exclusions and custom delimiter.

    Notes
    -----
    1. The function `to_title_case` ensures that
       each word is capitalized unless it is
       specified in `ex` (exclusions) and not at
       the beginning or end of the string.
    2. When `all_cap` is set to True, the exclusion
       list is ignored entirely.

    Examples
    --------
    >>> from gofast.utils.text import to_title_case
    >>> sample_text = ``"the lord of the rings"``  # double backticks
    >>> titled = to_title_case(sample_text,
    ...                        ex=["the", "of"])
    >>> print(titled)
    The Lord of the Rings

    >>> # Using custom delimiters
    >>> joined_words = ``"hello-world-foo"``      # double backticks
    >>> result = to_title_case(joined_words,
    ...                        delim="-")
    >>> print(result)
    Hello-World-Foo

    See Also
    --------
    :func:`to_title_case` : Illustrates how to
        handle special-case lower words, full
        capitalization overrides, or custom
        delimiters.

    """
    if not text:
        return text

    # Default exclusions if not provided.
    if ex is None:
        ex = [
            "and", "or", "the", "a", "an", "in", "on",
            "at", "to", "for", "by", "of"
        ]

    # Split using the provided delimiter or whitespace.
    words = (
        text.split(delim)
        if delim is not None else text.split()
    )
    new_words = []

    for i, w in enumerate(words):
        lw = w.lower()
        # If not forcing all words to capitalize
        # and word is in the exclusion list,
        # keep it lowercase if it's not the first
        # or last word.
        if (
            not all_cap
            and i not in (0, len(words) - 1)
            and lw in ex
        ):
            new_words.append(lw)
        else:
            new_words.append(lw.capitalize())

    return " ".join(new_words)


@to_text
def to_camel_case(
    text: str,
    *,
    up: bool = False,
    sep: Optional[str] = None,
    keep_alphanumeric: bool = True,
    strip_extra_spaces: bool = True,
    to_lower: bool = True
) -> str:
    """
    Convert the input ``text`` into camelCase or
    PascalCase using ``to_camel_case``.

    .. math::
       \\text{camel\\_case}(\\text{string})
       = \\mathrm{Join}\\bigl(
       \\mathrm{Capitalize}(
       \\mathrm{Split}(\\text{string}))\\bigr)

    Parameters
    ----------
    text : str
        The `parameter inline` string to convert.
    up : bool
        If `up` is True, convert to PascalCase
        (with the first letter uppercase).
        Otherwise, produce camelCase. Default is
        False.
    sep : str, optional
        A custom separator for splitting words.
        If None, words are split on non-alphanumeric
        characters.
    keep_alphanumeric : bool
        If `keep_alphanumeric` is True, discard
        empty entries after splitting on
        non-alphanumeric boundaries. Default True.
    strip_extra_spaces : bool
        If `strip_extra_spaces` is True, leading
        and trailing whitespace are removed from
        `text` before further processing. Default
        True.
    to_lower : bool
        If `to_lower` is True, transform initial
        parts of each word to lowercase before
        capitalizing (except for PascalCase
        first word if `up` is True). This helps
        ensure consistent casing. Default True.

    Returns
    -------
    str
        A string in camelCase or PascalCase.

    Notes
    -----
    1. When `up` is False, the function returns
       camelCase with the first word in lowercase
       and subsequent words capitalized.
    2. Non-alphanumeric characters (if
       `sep` is None) are used as split points,
       and empty tokens are discarded unless
       `keep_alphanumeric` is False.

    Examples
    --------
    >>> from gofast.utils.text import to_camel_case
    >>> raw_text = ``"Hello world!"``  # double backticks
    >>> camel = to_camel_case(raw_text, up=False)
    >>> print(camel)
    helloWorld

    >>> pascal = to_camel_case(raw_text, up=True)
    >>> print(pascal)
    HelloWorld

    See Also
    --------
    to_camel_case : Converts a textual
        input into either camelCase or PascalCase,
        handling splitting and capitalization.

    """
    # Pre-strip whitespace if requested.
    if strip_extra_spaces:
        text = text.strip()

    # If there's no text left, return it.
    if not text:
        return text

    # Split using the given separator, or split on
    # non-alphanumeric chars if `sep` is None.
    if sep is not None:
        parts = text.split(sep)
    else:
        parts = re.split(r'[^a-zA-Z0-9]+', text)

    # Filter out empty strings if keep_alphanumeric is True.
    parts = [p for p in parts if p] if keep_alphanumeric else parts

    # If no valid parts remain, return an empty string.
    if not parts:
        return ""

    # Decide how to handle the first part.
    # For PascalCase, we capitalize the first part.
    # For camelCase, we lowercase if `to_lower`
    # is True, else keep as is.
    first_part = parts[0]
    if up:
        first_part = first_part.capitalize()
    else:
        first_part = first_part.lower() if to_lower else first_part

    # Capitalize subsequent parts.
    rest_parts = []
    for p in parts[1:]:
        # Optionally lowercase before capitalizing
        # for consistency.
        piece = p.lower() if to_lower else p
        rest_parts.append(piece.capitalize())

    return first_part + "".join(rest_parts)


@to_text
def to_snake_case(
    text: str,
    *,
    low: bool = True,
    sep: str = "_",
    sp: bool = True,
    strip_extra: bool = True
) -> str:
    """
    Convert the input ``text`` into snake_case
    using ``to_snake_case`` with customizable
    options.

    .. math::
       \\text{snake\\_case}(\\text{input})
       = \\mathrm{Lower}(
       \\mathrm{ReplaceWhitespace}(
       \\mathrm{Split}(\\text{input}))) \\dots

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        processed.
    low : bool
        If `low` is True, the final output is
        converted to lowercase. Default True.
    sep : str
        The separator used to join words; defaults
        to underscore ``_``.
    sp : bool
        If `sp` is True, whitespace in `text`
        is replaced by `sep`. Default True.
    strip_extra : bool
        If `strip_extra` is True, leading and
        trailing instances of `sep` and duplicate
        separators are removed. Default True.

    Returns
    -------
    str
        A string in snake_case format.

    Notes
    -----
    1. By default, non-word characters (excluding
       the chosen separator) are replaced by `sep`.
    2. Duplicate `sep` are condensed into a
       single instance, and leading/trailing
       separators are stripped if `strip_extra`
       is set.

    Examples
    --------
    >>> from gofast.utils.text import to_snake_case
    >>> raw_text = ``"Hello World"`` # double backticks
    >>> result = to_snake_case(raw_text, low=True)
    >>> print(result)
    hello_world

    >>> # Using dashes instead of underscores
    >>> dashed = to_snake_case("Data  Science",
    ...                        sep="-")
    >>> print(dashed)
    data-science

    See Also
    --------
    :func:`to_snake_case` : Creates a fully
        snake_cased representation of a string,
        optionally removing extraneous whitespace
        and forcing lowercase.

    """
    # If text is empty, just return it.
    if not text:
        return text

    # Replace whitespace with the separator if sp
    # is True.
    if sp:
        text = re.sub(r'\s+', sep, text)

    # Replace non-word characters (except for `sep`)
    # with the separator.
    pattern = r'[^\w{}]+'.format(re.escape(sep))
    text = re.sub(pattern, sep, text)

    if strip_extra:
        # Remove duplicate separators.
        text = re.sub(r'{}+'.format(re.escape(sep)),
                      sep, text)
        # Trim leading/trailing separators.
        text = text.strip(sep)

    return text.lower() if low else text


@to_text
def capitalize_first(
    text: str,
    *,
    rest_low: bool = False,
    st: bool = True,
    suffix: Optional[str] = None
) -> str:
    """
    Capitalize only the first letter of ``text``
    using ``capitalize_first``.

    .. math::
       \\text{cap\\_first}(\\text{input})
       = \\mathrm{UpperFirst}(
       \\mathrm{OptionallyLowerRest}(\\text{input})
       )

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        processed.
    rest_low : bool
        If `rest_low` is True, all characters
        after the first are converted to
        lowercase. Default False.
    st : bool
        If `st` is True, strip leading and trailing
        whitespace. Default True.
    suffix : str, optional
        A string to append to the result after
        capitalization. For example, "!" or ".".

    Returns
    -------
    str
        The resulting string with the first
        character capitalized, optional lowercase
        remainder, and optional appended suffix.

    Notes
    -----
    1. If the input `text` is empty, it is
       returned as-is without modification.
    2. The parameter `suffix` adds a custom
       trailing string to the result, which can be
       useful for punctuation or emphasis.

    Examples
    --------
    >>> from gofast.utils.text import capitalize_first
    >>> text_input = ``" hello WORLD "``  # double backticks
    >>> capped = capitalize_first(text_input,
    ...                           rest_low=True,
    ...                           suffix=".")
    >>> print(capped)
    Hello world.

    See Also
    --------
    :func:`capitalize_first` : Demonstrates how
        to transform only the first character in
        a string and optionally alter the rest.

    """
    # Return early if there's no text.
    if not text:
        return text

    # Optionally strip leading and trailing spaces.
    if st:
        text = text.strip()

    if not text:
        return text

    # Capitalize the first char.
    if rest_low:
        result = text[0].upper() + text[1:].lower()
    else:
        result = text[0].upper() + text[1:]

    # Append suffix if provided.
    if suffix:
        result += suffix

    return result

@to_text
def tokenize(
    text: str,
    *,
    delim: Optional[str] = None,
    to_lower: bool = True,
    remove_empty: bool = True,
    keep_punct: bool = False
) -> List[str]:
    """
    Splits the input ``text`` into a list of tokens
    based on delimiters and punctuation, using
    ``tokenize``.

    .. math::
       \\text{tokens} = \\mathrm{Tokenize}(
       \\mathrm{ApplyDelim}(
       \\mathrm{Case}(
       \\mathrm{RemovePunct}(
       \\text{text})))) \\dots

    Parameters
    ----------
    text : str
        The `parameter inline` string to tokenize.
    delim : str, optional
        A custom delimiter for splitting text.
        If None, the function uses a regular
        expression to split on non-word characters
        (\\W+).
    to_lower : bool
        If `to_lower` is True, convert all tokens
        to lowercase.
    remove_empty : bool
        If `remove_empty` is True, remove empty tokens
        from the result.
    keep_punct : bool
        If `keep_punct` is False, remove punctuation
        from each token. When True, punctuation
        characters in tokens remain.

    Returns
    -------
    List[str]
        A list of tokenized words or strings,
        depending on the chosen splitting rules.

    Notes
    -----
    1. When no `delim` is provided, the regular
       expression ``r'\\W+'`` is used to split
       the text on any sequence of non-word
       characters.
    2. If `keep_punct` is False, standard
       punctuation from Python's
       :obj:`string.punctuation` is removed from
       each token.

    Examples
    --------
    >>> from gofast.utils.text import tokenize
    >>> sample_text = ``"Hello, World!"``  # double backticks
    >>> tokens = tokenize(sample_text,
    ...                   to_lower=True,
    ...                   rm_empty=True,
    ...                   keep_punct=False)
    >>> print(tokens)
    ['hello', 'world']

    See Also
    --------
    :func:`tokenize` : Splits a string into tokens
        with optional punctuation removal and
        custom delimiters.

    """
    # If a custom delimiter is provided, use it
    # to split. Otherwise, split on non-word chars.
    if delim is not None:
        tokens = text.split(delim)
    else:
        tokens = re.split(r'\W+', text)

    # Optionally convert to lowercase.
    if to_lower:
        tokens = [t.lower() for t in tokens]

    # Optionally remove punctuation.
    if not keep_punct:
        translator = str.maketrans('', '', string.punctuation)
        tokens = [t.translate(translator) for t in tokens]

    # Optionally remove empty tokens.
    if remove_empty:
        tokens = [t for t in tokens if t]

    return tokens

@to_text
def text_analysis(
    text: str,
    *,
    excl_space: bool = False,
    mode: Optional[str] = None,
    **tok_kwargs
) -> Union[Dict[str, object], int, Dict[str, int]]:
    """
    Analyze the input ``text`` to compute
    aggregated metrics such as word count,
    character count, and word frequency, using
    ``text_analysis``.

    .. math::
       \\text{analysis} = \\bigl\\{
       \\mathrm{word\\_count}(\\text{text}),
       \\mathrm{char\\_count}(\\text{text}),
       \\mathrm{word\\_frequency}(\\text{text})
       \\bigr\\}

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        analyzed.
    excl_space : bool
        If `excl_space` is True, exclude all
        whitespace when computing the character
        count. Default False.
    mode : str, optional
        Determines which result to return:
          - If ``mode`` is None, returns a
            dictionary containing "word_count",
            "char_count", and "word_frequency".
          - If ``mode == 'char_count'``, returns
            only the integer character count.
          - If ``mode == 'word_count'``, returns
            only the integer word count.
          - If ``mode == 'frequency'``, returns
            only the dictionary of word frequencies.
    **tok_kwargs :
        Additional keyword arguments passed to
        :func:`tokenize` for tokenization
        customization.

    Returns
    -------
    Dict[str, object] or int or Dict[str, int]
        - If ``mode is None``, returns a dict with
          the keys:
          "word_count", "char_count",
          "word_frequency".
        - If ``mode == 'char_count'``, returns an
          integer representing the character count.
        - If ``mode == 'word_count'``, returns an
          integer representing the word count.
        - If ``mode == 'frequency'``, returns a
          dict mapping each word to its count.

    Notes
    -----
    1. If ``excl_space`` is True, whitespace is
       removed for the character count, but not
       for the tokenization step.
    2. Additional tokenization parameters (e.g.,
       `delim`, `to_lower`, `keep_punct`) can
       be passed to adapt word counting and
       frequencies as needed.

    Examples
    --------
    >>> from gofast.utils.text import text_analysis
    >>> data = ``"Hello world! Hello Universe."``   # double backticks
    >>> # Returning all metrics:
    >>> result = text_analysis(data,
    ...                        excl_space=True,
    ...                        to_lower=True,
    ...                        rm_empty=True,
    ...                        keep_punct=False)
    >>> print(result)
    {
      'word_count': 4,
      'char_count': 25,
      'word_frequency': {'hello': 2,
                         'world': 1,
                         'universe': 1}
    }

    >>> # Returning only word frequencies:
    >>> freq = text_analysis(data,
    ...                     mode='frequency',
    ...                     to_lower=True)
    >>> print(freq)
    {'hello': 2, 'world': 1, 'universe': 1}

    See Also
    --------
    :func:`tokenize` : Used internally to split
        the text and compute word frequencies
        and counts.

    """
    # Tokenize text based on user-specified kwargs.
    tokens = tokenize(text, **tok_kwargs)
    word_count_value = len(tokens)

    # Optionally exclude spaces for character count.
    text_for_char = re.sub(r'\s+', '', text) if excl_space else text
    char_count_value = len(text_for_char)

    # Build the word frequency dictionary.
    freq_dict = dict(Counter(tokens))

    # Decide what to return based on `mode`.
    if mode == "char_count":
        return char_count_value
    elif mode == "word_count":
        return word_count_value
    elif mode == "frequency":
        return freq_dict
    else:
        return {
            "word_count": word_count_value,
            "char_count": char_count_value,
            "word_frequency": freq_dict
        }

@to_text
def substring_pattern_match(
    text: str,
    *,
    mode: str = "find",
    substring: Optional[str] = None,
    pattern: Optional[str] = None,
    replacement: str = "",
    use_regex: bool = False,
    flags: int = 0,
    first_only: bool = False,
    overlap: bool = False,
    count: int = 0
) -> Union[List[int], str, bool]:
    """
    Perform substring and pattern matching operations
    on the given ``text`` via
    ``substring_pattern_match``.

    This function supports multiple modes:
      - ``"find"``
      - ``"replace"``
      - ``"contains"``

    In ``"find"`` mode, all start indices of the
    matching substring or pattern are returned.
    If ``first_only`` is True, only the first match
    index is returned in a single-element list.
    Overlapping matches are supported for plain
    substring finds if ``overlap`` is True. In
    ``"replace"`` mode, the function performs
    replacements of substring or pattern with the
    specified ``replacement``. In ``"contains"``
    mode, it returns a boolean indicating whether
    the text contains the substring or matches
    the pattern.

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        processed.
    mode : str
        The operation mode, one of:
        ``"find"``, ``"replace"``, or
        ``"contains"``. Default is ``"find"``.
    substring : str, optional
        A plain substring for matching if
        ``use_regex`` is False, or for use in
        ``"replace"`` mode (plain). Ignored
        if ``pattern`` is provided or
        ``use_regex`` is True.
    pattern : str, optional
        A regular expression to use if
        ``use_regex`` is True or if the function
        is in ``"replace"`` mode with regex.
    replacement : str
        The string to replace matches with
        when in ``"replace"`` mode.
    use_regex : bool
        If True, apply regex matching or
        replacement; if False, use plain
        substring matching. Default False.
    flags : int
        Regex flags (e.g., ``re.IGNORECASE``)
        for regex operations. Default 0.
    first_only : bool
        In ``"find"`` mode, if True, return only
        the first occurrence's index in a list.
    overlap : bool
        In ``"find"`` mode with plain substring
        matching, if True, allow overlapping
        matches. Ignored for regex matching.
    count : int
        In ``"replace"`` mode, the maximum number
        of replacements. A value of 0 means
        "replace all". Default 0.

    Returns
    -------
    list of int or str or bool
        - If ``mode == "find"``, returns a list
          of starting indices of matches (or
          a single-element list if
          ``first_only`` is True).
        - If ``mode == "replace"``, returns the
          resulting string after replacements.
        - If ``mode == "contains"``, returns a
          boolean indicating if the match was
          found.

    Notes
    -----
    1. When using plain substring matching, the
       function uses Python's built-in
       :meth:`str.find`. With overlapping
       enabled, the search index is advanced
       by 1 each time, rather than by the length
       of the substring.
    2. For regex operations, ensure that
       ``pattern`` is not None and
       ``use_regex`` is True.

    Examples
    --------
    >>> from gofast.utils.text import substring_pattern_match
    >>> text_data = ``"Hello Hello Hello"``  # double backticks
    >>> # Finding all occurrences of "Hello":
    >>> indices = substring_pattern_match(text_data,
    ...                                   mode="find",
    ...                                   substring="Hello",
    ...                                   overlap=True)
    >>> print(indices)
    [0, 6, 12]

    >>> # Checking if text contains "Hello":
    >>> contains_hello = substring_pattern_match(
    ...    text_data,
    ...    mode="contains",
    ...    substring="Hello"
    ... )
    >>> print(contains_hello)
    True

    >>> # Regex replace "Hello" with "Hi", limit to 2 replacements:
    >>> replaced_text = substring_pattern_match(
    ...    text_data,
    ...    mode="replace",
    ...    pattern=r"Hello",
    ...    replacement="Hi",
    ...    use_regex=True,
    ...    count=2
    ... )
    >>> print(replaced_text)
    Hi Hi Hello

    See Also
    --------
    :func:`substring_pattern_match` : Offers
        advanced substring or regex-based find,
        replace, and contains functionality.

    """
    # Normalize user input for mode.
    mode = mode.lower()

    # Handle "find" mode.
    if mode == "find":
        indices = []
        if use_regex:
            if pattern is None:
                raise ValueError(
                    "Pattern must be provided "
                    "for regex find."
                )
            for match_obj in re.finditer(pattern, text, flags=flags):
                indices.append(match_obj.start())
                if first_only:
                    break
        else:
            if substring is None:
                raise ValueError(
                    "Substring must be provided "
                    "for plain find."
                )
            start = 0
            sub_len = len(substring)
            while True:
                idx = text.find(substring, start)
                if idx == -1:
                    break
                indices.append(idx)
                if first_only:
                    break
                start = idx + (1 if overlap else sub_len)
        return indices

    # Handle "replace" mode.
    elif mode == "replace":
        if use_regex:
            if pattern is None:
                raise ValueError(
                    "Pattern must be provided "
                    "for regex replace."
                )
            # If count == 0, replace all occurrences.
            result = re.sub(pattern, replacement,
                            text, count=count, flags=flags)
        else:
            if substring is None:
                raise ValueError(
                    "Substring must be provided "
                    "for plain replace."
                )
            # str.replace does not accept count=0
            # as "all", so use -1 in that case.
            replace_count = count if count > 0 else -1
            result = text.replace(substring,
                                  replacement,
                                  replace_count)
        return result

    # Handle "contains" mode.
    elif mode == "contains":
        if use_regex:
            if pattern is None:
                raise ValueError(
                    "Pattern must be provided "
                    "for regex contains."
                )
            return re.search(pattern, text, flags=flags) is not None
        else:
            if substring is None:
                raise ValueError(
                    "Substring must be provided "
                    "for plain contains."
                )
            return substring in text

    else:
        raise ValueError(
            "Invalid mode. Choose from 'find', "
            "'replace', or 'contains'."
        )


@to_text
def slugify(
    text: str,
    *,
    to_lower: bool = True,
    delim: str = "-",
    allow_unicode: bool = False,
    custom_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Convert a string ``text`` into a URL-friendly
    slug via ``slugify``.

    .. math::
       \\text{slug} =
       \\mathrm{Strip}(\\mathrm{ReplaceNonAlnum}(
       \\mathrm{Normalize}(
       \\mathrm{Map}(\\text{text}))))\\dots

    Parameters
    ----------
    text : str
        The `parameter inline` input string
        to slugify.
    lower : bool
        If `lower` is True, convert the text
        to lowercase. Default True.
    delim : str
        The delimiter that replaces whitespace
        and non-alphanumeric sequences. Default
        ``"-"``.
    allow_unicode : bool
        If True, allow unicode characters and
        normalize with ``NFKC``. Otherwise,
        convert to ASCII via ``NFKD``. Default
        False.
    custom_map : dict, optional
        A dictionary of custom replacements
        applied before normalization. For
        instance,
        ``{"©": "c", "—": "-"}``.

    Returns
    -------
    str
        A URL-friendly slug string suitable for
        use in routes or filenames.

    Notes
    -----
    1. If `allow_unicode` is False, the function
       applies a ``NFKD`` normalization and then
       encodes to ASCII, discarding any
       non-ASCII bytes.
    2. The function replaces any sequence of
       spaces, underscores, or hyphens with a
       single instance of the chosen delimiter.
       Leading and trailing delimiters are also
       stripped.

    Examples
    --------
    >>> from gofast.utils.text import slugify
    >>> raw_text = ``"Hello, World! ~ 2025"``  # double backticks
    >>> s = slugify(raw_text,
    ...            lower=True,
    ...            delim="-",
    ...            allow_unicode=False,
    ...            custom_map={"~": "approximately"})
    >>> print(s)
    hello-world-approximately-2025

    See Also
    --------
    slugify: Ensures text is normalized,
        cleaned, and converted into a concise
        representation for URLs or filenames.

    """
    # Apply any custom replacements.
    if custom_map:
        for key, value in custom_map.items():
            text = text.replace(key, value)

    # Normalize text.
    if allow_unicode:
        text = unicodedata.normalize('NFKC', text)
    else:
        text = (
            unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('ascii')
        )

    # Optionally convert to lowercase.
    if to_lower:
        text = text.lower()

    # Remove unwanted characters: keep alphanumerics,
    # underscores, hyphens, and spaces.
    text = re.sub(r'[^\w\s-]', '', text)

    # Replace sequences of spaces, underscores,
    # or hyphens with a single delimiter.
    text = re.sub(r'[\s_-]+', delim, text).strip(delim)

    return text

@to_text
def reverse_text(
    text: str,
    *,
    reverse_by: str = "chars"
) -> str:
    """
    Reverse the input ``text`` by characters or
    words using ``reverse_text``.

    .. math::
       \\text{reverse}(\\text{string})
       = \\mathrm{ReverseOrder}(\\text{string})

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        reversed.
    reverse_by : str
        The mode of reversing:
          - ``"chars"``: Reverse the string by
            individual characters (default).
          - ``"words"``: Split the text on spaces
            and reverse the order of the resulting
            words.

    Returns
    -------
    str
        A string reversed according to
        ``reverse_by``.

    Notes
    -----
    1. When ``reverse_by="chars"``, the function
       returns the input string in reverse
       character order.
    2. When ``reverse_by="words"``, the function
       splits on whitespace (using :meth:`str.split`)
       and reverses the entire list of words.

    Examples
    --------
    >>> from gofast.utils.text import reverse_text
    >>> sample = ``"Hello World"``  # double backticks
    >>> rev_chars = reverse_text(sample, reverse_by="chars")
    >>> print(rev_chars)
    dlroW olleH

    >>> rev_words = reverse_text(sample, reverse_by="words")
    >>> print(rev_words)
    World Hello

    See Also
    --------
    :func:`reverse_text` : Reverses characters or
        words in a string, depending on the chosen
        mode.

    """
    if reverse_by == "words":
        words = text.split()
        return " ".join(words[::-1])
    # Default: reverse the string by characters.
    return text[::-1]


@to_text
def truncate(
    text: str,
    max_length: int,
    *,
    suffix: str = "...",
    word_boundaries: bool = False
) -> str:
    """
    Truncate the input ``text`` to a specified
    maximum length, appending a suffix if needed,
    using ``truncate``.

    .. math::
       \\text{truncated} =
       \\mathrm{Shorten}(\\text{text}) \\dots

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        truncated.
    max_length : int
        The maximum allowed length of the
        resulting string (including any suffix).
    suffix : str
        A string to append if truncation occurs.
        Default ``"..."``.
    word_boundaries : bool
        If True, the function attempts to avoid
        cutting in the middle of a word by
        locating the last whitespace before
        the cut. Default False.

    Returns
    -------
    str
        The truncated text. If the length of
        ``text`` is already within
        ``max_length``, the original text is
        returned unchanged.

    Notes
    -----
    1. If ``max_length <= len(suffix)``, the
       function truncates the text to
       ``max_length`` directly, which may
       eliminate the suffix if space is
       insufficient.
    2. If ``word_boundaries`` is True, the
       function cuts at the last space within
       the allowed length to avoid breaking
       a word.

    Examples
    --------
    >>> from gofast.utils.text import truncate
    >>> long_text = ``"This is a very long sentence..."`` # double backticks
    >>> short = truncate(long_text, 10,
    ...                  suffix="...")
    >>> print(short)
    This is...

    >>> # Using word boundaries
    >>> short_bound = truncate(long_text, 15,
    ...                        suffix="[cut]",
    ...                        word_boundaries=True)
    >>> print(short_bound)
    This is a[cut]

    See Also
    --------
    :func:`truncate` : Cuts a string to a
        maximum length, optionally respecting
        word boundaries.

    """
    if len(text) <= max_length:
        return text
    if max_length <= len(suffix):
        return text[:max_length]

    allowed = max_length - len(suffix)
    if word_boundaries:
        # Find the last whitespace before the cutoff.
        truncated = text[:allowed]
        last_space = truncated.rfind(" ")
        if last_space != -1:
            truncated = truncated[:last_space]
    else:
        truncated = text[:allowed]

    return truncated + suffix


@to_text
def extract_sentences(
    text: str,
    *,
    pattern: str = r'(?<=[.!?])\s+',
    keep_delim: bool = False,
    min_length: int = 0
) -> List[str]:
    """
    Split the input ``text`` into individual
    sentences using ``extract_sentences``.

    .. math::
       \\text{sentences} =
       \\mathrm{SplitOnRegex}(\\text{text}) \\dots

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        segmented into sentences.
    pattern : str
        A regular expression pattern for
        sentence splitting. Defaults to
        splitting whenever punctuation
        ('.', '!', '?') is followed by
        whitespace.
    keep_delim : bool
        If True, keeps the sentence delimiters
        attached to each sentence. This is
        achieved by capturing them in the
        regex pattern. Default False.
    min_length : int
        The minimum length a sentence must
        have to be included. Default 0.

    Returns
    -------
    List[str]
        A list of sentences extracted from
        the text. Sentences shorter than
        ``min_length`` are excluded.

    Notes
    -----
    1. If ``keep_delim`` is True, the function
       uses a regex that captures ending
       punctuation as part of each sentence,
       ensuring the delimiters remain attached.
    2. If you want finer-grained control over
       sentence splitting (e.g., handling
       abbreviations), consider a more advanced
       NLP library.

    Examples
    --------
    >>> from gofast.utils.text import extract_sentences
    >>> paragraph = ``"Hello world! This is an example. Test?"`` # double backticks
    >>> sents = extract_sentences(paragraph,
    ...                           keep_delim=False)
    >>> print(sents)
    ['Hello world!', 'This is an example.', 'Test?']

    >>> # Keep delimiters explicitly
    >>> sents_delim = extract_sentences(paragraph,
    ...                                 keep_delim=True)
    >>> print(sents_delim)
    ['Hello world!', ' This is an example.', ' Test?']

    See Also
    --------
    :func:`extract_sentences` : Splits text into
        smaller pieces by sentence-ending
        punctuation and optional whitespace.

    """
    if keep_delim:
        # Use a regex to capture sentences along
        # with their ending punctuation.
        sentences = re.findall(
            r'.*?[.!?](?:\s|$)',
            text,
            flags=re.DOTALL
        )
    else:
        # Split using the provided pattern.
        sentences = re.split(pattern, text)

    # Strip and filter out sentences shorter
    # than min_length.
    return [
        s.strip()
        for s in sentences
        if len(s.strip()) >= min_length
    ]


@to_text
def encode_text(
    text: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    add_bom: bool = False
) -> bytes:
    """
    Encode the input ``text`` into bytes using the
    specified encoding via ``encode_text``.

    .. math::
       \\text{encoded} =
       \\mathrm{Encode}(\\text{text}, \\text{encoding})

    Parameters
    ----------
    text : str
        The `parameter inline` string to encode.
    encoding : str
        The character encoding to use. Defaults
        to ``"utf-8"``.
    errors : str
        The error handling scheme. Possible
        values include ``"strict"``, ``"ignore"``,
        and ``"replace"``. Default is
        ``"strict"``.
    add_bom : bool
        If True, prepend a Byte Order Mark
        (BOM) for encodings that support it.
        Default False.

    Returns
    -------
    bytes
        The byte string resulting from encoding
        the input text.

    Notes
    -----
    1. Common BOMs include ``b'\\xef\\xbb\\xbf'``
       for UTF-8 and ``b'\\xff\\xfe'`` for
       UTF-16LE [1]_.
    2. If ``add_bom`` is True, the function
       looks up the BOM from a predefined map
       for common encodings (e.g., ``'utf-8'``,
       ``'utf-16'``, ``'utf-32'``, etc.).

    Examples
    --------
    >>> from gofast.utils.text import encode_text
    >>> text_data = ``"Hello world"``  # double backticks
    >>> encoded_bytes = encode_text(text_data,
    ...                              encoding="utf-8",
    ...                              add_bom=False)
    >>> print(encoded_bytes)
    b'Hello world'

    See Also
    --------
    :func:`encode_text` : Converts a Python
        string to a specified encoding, optionally
        inserting a BOM.

    References
    ----------
    .. [1] Unicode Consortium, "Byte Order
           Mark (BOM)", 2025.

    """
    encoded = text.encode(encoding, errors=errors)

    if add_bom:
        # Define BOMs for common encodings.
        boms = {
            "utf-8": b"\xef\xbb\xbf",
            "utf-16": b"\xff\xfe",  # Little-endian
            "utf-16le": b"\xff\xfe",
            "utf-16be": b"\xfe\xff",
            "utf-32": b"\xff\xfe\x00\x00",  # Little-endian
            "utf-32le": b"\xff\xfe\x00\x00",
            "utf-32be": b"\x00\x00\xfe\xff",
        }
        enc_lower = encoding.lower()
        if enc_lower in boms:
            encoded = boms[enc_lower] + encoded

    return encoded


@to_text
def decode_text(
    encoded_text: bytes,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    remove_bom: bool = False
) -> str:
    """
    Decode the byte sequence ``encoded_text``
    into a string using ``decode_text``.

    .. math::
       \\text{decoded} =
       \\mathrm{Decode}(\\text{encoded\\_text},
       \\text{encoding})

    Parameters
    ----------
    encoded_text : bytes
        The `parameter inline` byte sequence to
        decode.
    encoding : str
        The character encoding to use.
        Defaults to ``"utf-8"``.
    errors : str
        The error handling scheme: possible
        values include ``"strict"``, ``"ignore"``,
        and ``"replace"``. Default
        ``"strict"``.
    remove_bom : bool
        If True, remove a leading Byte Order
        Mark (BOM) if present. Default False.

    Returns
    -------
    str
        The decoded string resulting from
        interpreting the input bytes with the
        specified encoding.

    Notes
    -----
    1. If ``remove_bom`` is True, the function
       scans for known BOM sequences (e.g.,
       UTF-8, UTF-16, UTF-32) and strips them
       if found.
    2. The function then applies
       :meth:`bytes.decode` to convert the
       remaining bytes.

    Examples
    --------
    >>> from gofast.utils.text import decode_text
    >>> data = b'Hello world'
    >>> decoded = decode_text(data,
    ...                       encoding="utf-8",
    ...                       remove_bom=False)
    >>> print(decoded)
    Hello world

    See Also
    --------
    :func:`decode_text` : Converts a byte
        string to a Python string, optionally
        removing a BOM.

    """
    if remove_bom:
        # List of known BOMs.
        boms = [
            b"\xef\xbb\xbf",       # UTF-8 BOM
            b"\xff\xfe",           # UTF-16 LE BOM
            b"\xfe\xff",           # UTF-16 BE BOM
            b"\xff\xfe\x00\x00",   # UTF-32 LE BOM
            b"\x00\x00\xfe\xff",   # UTF-32 BE BOM
        ]
        for bom in boms:
            if encoded_text.startswith(bom):
                encoded_text = encoded_text[len(bom) :]
                break

    return encoded_text.decode(encoding, errors=errors)


@to_text (params=['s1', 's2'], allow_none=False)
def levenshtein_distance(
    s1: str,
    s2: str,
    *,
    case_sensitive: bool = True,
    weight_insert: int = 1,
    weight_delete: int = 1,
    weight_substitute: int = 1
) -> int:
    """
    Compute the Levenshtein distance between two
    strings ``s1`` and ``s2`` with optional custom
    operation weights using ``levenshtein_distance``.

    .. math::
       d(\\text{s1}, \\text{s2}) =
       \\mathrm{Levenshtein}(\\text{s1}, \\text{s2})

    Parameters
    ----------
    s1 : str
        The first `parameter inline` string.
    s2 : str
        The second `parameter inline` string.
    case_sensitive : bool
        If False, both strings are converted
        to lowercase before computing
        distances. Default True.
    weight_insert : int
        The cost of an insertion. Default 1.
    weight_delete : int
        The cost of a deletion. Default 1.
    weight_substitute : int
        The cost of a substitution. Default 1.

    Returns
    -------
    int
        The computed Levenshtein distance
        between ``s1`` and ``s2``.

    Notes
    -----
    1. The Levenshtein distance (also known as
       edit distance) is the minimum number
       of single-character edits (insertions,
       deletions, or substitutions) required
       to change one word into the other [1]_.
    2. Custom weights can be provided to
       reflect domain-specific costs for each
       edit operation.

    Examples
    --------
    >>> from gofast.utils.text import levenshtein_distance
    >>> dist = levenshtein_distance("cat", "cut",
    ...                             case_sensitive=True)
    >>> print(dist)
    1

    >>> # With weights:
    >>> dist_weighted = levenshtein_distance("cat", "cut",
    ...                                      weight_substitute=2)
    >>> print(dist_weighted)
    2

    References
    ----------
    .. [1] Vladimir Levenshtein, "Binary codes
           capable of correcting deletions,
           insertions and reversals", Soviet
           Physics Doklady, 1966.

    """
    if not case_sensitive:
        s1, s2 = s1.lower(), s2.lower()

    len1, len2 = len(s1), len(s2)

    # Initialize distance matrix.
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Base cases: distance from empty string.
    for i in range(len1 + 1):
        dp[i][0] = i * weight_delete
    for j in range(len2 + 1):
        dp[0][j] = j * weight_insert

    # Compute the distances.
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = (
                0
                if s1[i - 1] == s2[j - 1]
                else weight_substitute
            )
            dp[i][j] = min(
                dp[i - 1][j] + weight_delete,    # Deletion
                dp[i][j - 1] + weight_insert,    # Insertion
                dp[i - 1][j - 1] + cost          # Substitution
            )

    return dp[len1][len2]

@to_text
def is_palindrome(
    text: str,
    *,
    ignore_case: bool = True,
    ignore_spaces: bool = True,
    ignore_punctuation: bool = True
) -> bool:
    """
    Check if the given ``text`` is a palindrome,
    optionally ignoring case, spaces, and
    punctuation via ``is_palindrome``.

    .. math::
       \\mathrm{is\\_palindrome}(\\text{text})
       = (\\text{processed} = \\mathrm{Reverse}(
       \\text{processed}))

    Parameters
    ----------
    text : str
        The `parameter inline` string to check
        for palindromic symmetry.
    ignore_case : bool
        If True, comparison is case-insensitive
        (converts text to lowercase first).
        Default True.
    ignore_spaces : bool
        If True, all spaces are removed from
        `text` before checking. Default True.
    ignore_punctuation : bool
        If True, all punctuation is removed
        from `text` before checking. Default
        True.

    Returns
    -------
    bool
        True if the processed text is a palindrome;
        False otherwise.

    Notes
    -----
    1. Palindrome checks compare the processed
       string with its reverse. If they match,
       the text is palindromic.
    2. When ignoring punctuation, the function
       uses :obj:`string.punctuation` to remove
       ASCII punctuation.

    Examples
    --------
    >>> from gofast.utils.text import is_palindrome
    >>> result = is_palindrome(``"A man, a plan, a canal: Panama!"``)  # double backticks
    >>> print(result)
    True

    >>> # Retain punctuation and consider case:
    >>> result_case = is_palindrome(``"Madam"``,
    ...                             ignore_case=False,
    ...                             ignore_spaces=False,
    ...                             ignore_punctuation=False)
    >>> print(result_case)
    False

    See Also
    --------
    :func:`is_palindrome` : Checks if a string
        reads the same forward and backward,
        with optional filters for case, spaces,
        and punctuation.

    """
    processed = text
    # Optionally lower the text.
    if ignore_case:
        processed = processed.lower()
    # Optionally remove spaces.
    if ignore_spaces:
        processed = processed.replace(" ", "")
    # Optionally remove punctuation.
    if ignore_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        processed = processed.translate(translator)

    return processed == processed[::-1]


@to_text
def anonymize_text(
    text: str,
    sensitive_words: List[str],
    *,
    placeholder: str = "[REDACTED]",
    ignore_case: bool = True,
    regex: bool = False
) -> str:
    """
    Replace sensitive words in the given ``text``
    with a placeholder to anonymize content,
    using ``anonymize_text``.

    Parameters
    ----------
    text : str
        The `parameter inline` string to be
        anonymized.
    sensitive_words : list of str
        A list of words or patterns considered
        sensitive. These entries will be
        replaced by `placeholder`.
    placeholder : str
        The string used to replace sensitive
        words. Default ``"[REDACTED]"``.
    ignore_case : bool
        If True, matching is case-insensitive.
        Default True.
    regex : bool
        If True, each entry in `sensitive_words`
        is treated as a regular expression
        pattern. Otherwise, words are taken as
        literal strings. Default False.

    Returns
    -------
    str
        The anonymized text in which all
        occurrences of sensitive words or
        patterns have been replaced by
        `placeholder`.

    Notes
    -----
    1. If `regex` is True, advanced patterns
       (e.g. partial matches, groups) can be
       used in `sensitive_words`.
    2. If `ignore_case` is True, the function
       sets the :obj:`re.IGNORECASE` flag,
       making matches case-insensitive.

    Examples
    --------
    >>> from gofast.utils.text import anonymize_text
    >>> text_data = ``"This is John. He lives in Paris."``  # double backticks
    >>> redacted = anonymize_text(
    ...     text_data,
    ...     ["John", "Paris"],
    ...     placeholder="[HIDDEN]",
    ...     ignore_case=True
    ... )
    >>> print(redacted)
    This is [HIDDEN]. He lives in [HIDDEN].

    >>> # Using regex patterns:
    >>> text_data_2 = ``"Call me at 123-456-7890."``  # double backticks
    >>> redacted_2 = anonymize_text(
    ...     text_data_2,
    ...     [r"\\d{3}-\\d{3}-\\d{4}"],
    ...     regex=True
    ... )
    >>> print(redacted_2)
    Call me at [REDACTED].

    See Also
    --------
    :func:`anonymize_text` : Masks sensitive
        terms or patterns with a placeholder,
        optionally using regex for fine-grained
        matches.

    """
    flags = re.IGNORECASE if ignore_case else 0

    def _replacer(_match):
        return placeholder

    for word in sensitive_words:
        if regex:
            pattern = word
        else:
            # Escape the word for literal usage if
            # not using regex.
            pattern = re.escape(word)

        text = re.sub(
            pattern,
            _replacer,
            text,
            flags=flags
        )

    return text

@check_non_emptiness
@isdf 
def select_text_columns(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    threshold: float = 0.7,
    check_for_words: bool = True,
    return_frame: bool = False
) -> pd.DataFrame:
    """
    Select columns in a DataFrame that are likely
    to contain textual data.

    This function analyzes each column using a
    combination of heuristics to determine if
    it is "text-like" or not. By default, it
    returns a list of column names that qualify;
    if ``return_frame=True``, it returns the
    sub-DataFrame containing only those columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    sample_size : int, optional
        Number of rows to sample per column for
        textual analysis. Default is 50. If a
        column has fewer rows than `sample_size`,
        all rows in that column are used.
    threshold : float, optional
        The minimum fraction of sampled entries
        that must appear to be non-numeric text
        for the column to be considered text-like.
        Default is 0.7.
    check_for_words : bool, optional
        If True, the function additionally checks
        that a sampled value contains at least
        one alphabetic token (e.g., matches
        ``r"[A-Za-z]"``) to confirm textual
        content. This helps exclude columns of
        random codes or purely numeric data that
        happen to be stored as strings. Default
        True.
    return_frame : bool, optional
        If True, return a sub-DataFrame
        containing only the text-like columns.
        Otherwise, return a list of those column
        names. Default False.

    Returns
    -------
    list of str or pd.DataFrame
        - If ``return_frame=False`` (default),
          returns a list of column names that
          appear to be text-like.
        - If ``return_frame=True``, returns a
          sub-DataFrame containing those columns.

    Notes
    -----
    1. Columns with numeric dtypes (e.g. int, float)
       are automatically excluded from text-like
       detection.
    2. A column is considered text-like if at
       least ``threshold`` fraction of its sampled
       entries appear to be non-numeric strings.
    3. The sampling approach speeds up detection
       for large datasets but may misclassify
       columns with inconsistent data.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.text import select_text_columns
    >>> data = {
    ...     "name": ["Alice", "Bob", "Charlie"],
    ...     "age": [25, 30, 35],
    ...     "description": ["Works at ACME.", "Enjoys sports!", "N/A"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> text_cols = select_text_columns(df)
    >>> print(text_cols)
    ['name', 'description']

    >>> # Return just a sub-DataFrame:
    >>> text_df = select_text_columns(df, return_frame=True)
    >>> print(text_df)
           name       description
    0     Alice   Works at ACME.
    1       Bob   Enjoys sports!
    2  Charlie             N/A

    """
    threshold = assert_ratio (
        threshold, bounds=(0, 1),
        exclude_values= [0], 
        name="Threshold" 
        )
    text_like_columns = []

    for col in df.columns:
        # Exclude numeric dtypes immediately.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Sample values
        col_data = df[col].dropna().head(sample_size)
        if len(col_data) == 0:
            # Empty or all NaN, skip
            continue

        non_numeric_count = 0
        total_checked = 0

        for val in col_data:
            # Convert to string for analysis
            val_str = str(val).strip()

            # Attempt to detect if it is numeric.
            # If the entire string is numeric-like,
            # treat it as not text-like.
            if re.match(r"^-?\d+(\.\d+)?$", val_str):
                continue

            # If we require at least one word-like token:
            if check_for_words and not re.search(r"[A-Za-z]", val_str):
                continue

            # If we get here, we consider it "text".
            non_numeric_count += 1
            total_checked += 1

        # Compare fraction of non-numeric or word-like entries
        if total_checked > 0:
            fraction_text = non_numeric_count / len(col_data)
            if fraction_text >= threshold:
                text_like_columns.append(col)

    if return_frame:
        return df[text_like_columns]
    else:
        return text_like_columns
    
@check_non_emptiness
@isdf
def select_text_columns_in(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    threshold: float = 0.7,
    check_for_words: bool = True,
    return_frame: bool = False,
    **kw
) -> pd.DataFrame:
    """
    Select columns in ``df`` that are likely to
    contain textual data via
    ``select_text_columns_in``.

    .. math::
       \\text{Score}(\\text{column})
       = \\mathrm{HeuristicChecks}(\\text{column})

    This function implements a heuristic-based
    approach to detect columns that appear
    text-like [1]_. For each column, it samples
    up to ``sample_size`` rows, checks if those
    values are non-numeric, and optionally scans
    for alphabetic characters. If a sufficient
    fraction (defined by <threshold inline>)
    of the sampled entries appear textual, the
    column is considered text-like.

    Parameters
    ----------
    df : pd.DataFrame
        The `parameter inline` DataFrame to
        analyze.
    sample_size : int
        Number of rows to sample per column for
        textual analysis. Default 50. If a column
        has fewer rows, all rows are sampled.
    threshold : float
        The minimum fraction of sampled entries
        that must appear text-like for the column
        to qualify. Default 0.7.
    check_for_words : bool
        If True, a sampled entry must contain at
        least one alphabetical character
        (:math:`[A-Za-z]`) to be deemed textual.
        Default True.
    return_frame : bool
        If True, return a sub-DataFrame
        containing only columns that are
        considered text-like. If False (default),
        return a list of text-like column names.
    **kw :
        Additional keyword arguments for future
        extensions (currently unused).

    Returns
    -------
    list of str or pd.DataFrame
        - If ``return_frame=False`` (default),
          returns a list of column names that
          appear text-like.
        - If ``return_frame=True``, returns a
          sub-DataFrame containing only those
          columns.

    Notes
    -----
    1. Columns with numeric dtypes (e.g., int,
       float) are automatically excluded.
    2. If the fraction of text-like entries in
       the sampled data is greater or equal
       to <threshold inline>, the column is
       selected.
    3. Checking for words (if
       <check_for_words inline> is True) helps
       exclude codes or purely symbolic
       strings.

    Examples
    --------
    >>> from gofast.utils.text import select_text_columns_in
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['Alice', 'Bob', 'Charlie'],
    ...     'age': [25, 30, 35],
    ...     'notes': ['Loves reading', 'N/A', 'Enjoys traveling!']
    ... })
    >>> cols = select_text_columns_in(
    ...     df,
    ...     sample_size=2,
    ...     threshold=0.5,
    ...     check_for_words=True,
    ...     return_frame=False
    ... )
    >>> print(cols)
    ['name', 'notes']

    >>> # Returning a sub-DataFrame:
    >>> df_text = select_text_columns_in(
    ...     df,
    ...     return_frame=True
    ... )
    >>> print(df_text)
           name             notes
    0     Alice     Loves reading
    1       Bob                N/A
    2  Charlie  Enjoys traveling!

    See Also
    --------
    :func:`tokenize_text_columns` :
        Tokenizes the text in specific or
        autodetected text-like columns.

    References
    ----------
    .. [1] M. Hearst et al., *Searching and Mining
           Text*, Journal of Data Mining, 2023.

    """
    text_like_columns = []
    
    threshold = assert_ratio (
        threshold, bounds=(0, 1),
        exclude_values= [0], 
        name="Threshold" 
        )
    
    for col in df.columns:
        # Exclude numeric dtypes
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Drop NaN and sample
        col_data = df[col].dropna()
        sample_data = col_data.head(sample_size)
        if sample_data.empty:
            continue

        # Count text-like entries
        texty_count = 0
        for val in sample_data:
            val_str = str(val).strip()

            # Skip purely numeric entries
            if re.match(r"^-?\d+(\.\d+)?$", val_str):
                continue

            # Check for alphabetic character if needed
            if check_for_words and not re.search(r"[A-Za-z]", val_str):
                continue

            texty_count += 1

        # Fraction of text-like entries
        fraction_text = texty_count / len(sample_data)
        if fraction_text >= threshold:
            text_like_columns.append(col)

    if return_frame:
        return df[text_like_columns]
    else:
        return text_like_columns

@check_non_emptiness 
@isdf
def tokenize_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    delim: str = r"\W+",
    to_lower: bool = True,
    remove_empty: bool = True,
    autodetect_text: bool = True,
    sample_size: int = 50,
    threshold: float = 0.7,
    check_for_words: bool = True,
    inplace: bool = False,
    **kw
) -> pd.DataFrame:
    """
    Tokenize textual columns in ``df``, returning
    a new DataFrame (or modifying it in-place)
    via ``tokenize_text_columns``.

    .. math::
       \\text{tokens} =
       \\mathrm{Split}(\\text{text}, \\text{delim})

    This function either uses user-provided
    <columns inline> or autodetects text-like
    columns via
    ``select_text_columns_in`` [1]_. It then
    splits each cell's content into tokens,
    optionally lowercasing them and removing
    empty strings.

    Parameters
    ----------
    df : pd.DataFrame
        The `parameter inline` DataFrame with
        potentially text-based columns.
    columns : list of str, optional
        Specific columns to tokenize. If None
        and <autodetect_text inline> is True,
        the function automatically detects
        columns using
        ``select_text_columns_in``.
    delim : str
        A regex pattern for splitting text
        into tokens. Defaults to ``r"\\W+"``,
        meaning any sequence of non-word
        characters.
    to_lower : bool
        If True, convert tokens to lowercase.
        Default True.
    remove_empty : bool
        If True, discard empty tokens (which
        may appear when consecutive delimiters
        exist). Default True.
    autodetect_text : bool
        If True and <columns inline> is None,
        columns likely containing text are
        autodetected. Default True.
    sample_size : int
        The sample size used by
        ``select_text_columns_in`` if detecting
        text columns. Default 50.
    threshold : float
        The threshold used by
        ``select_text_columns_in`` for text
        detection. Default 0.7.
    check_for_words : bool
        If True, also used in
        ``select_text_columns_in`` to confirm
        the presence of alphabetic characters
        in a sampled entry. Default True.
    inplace : bool
        If True, modify <df inline> directly.
        Otherwise, return a new DataFrame
        (default).
    **kw :
        Additional keyword arguments for future
        extension (currently unused).

    Returns
    -------
    pd.DataFrame
        A DataFrame in which the chosen text
        columns are replaced by lists of tokens.
        If <inplace inline> is True, the changes
        occur in the original <df inline>.

    Notes
    -----
    1. By default, columns are converted to
       `str` before tokenization to avoid
       errors from non-string dtypes.
    2. Tokenization is performed row by row
       with :math:`\\mathrm{re.split}` on
       <delim inline>, producing lists of
       substrings.

    Examples
    --------
    >>> from gofast.utils.text import (
    ...     select_text_columns_in,
    ...     tokenize_text_columns
    ... )
    >>> import pandas as pd
    >>> data = {
    ...     "A": ["val1", "val2"],
    ...     "B": [1, 2],
    ...     "Observations": [
    ...         "First line of text!",
    ...         "Second line... more text?"
    ...     ]
    ... }
    >>> df = pd.DataFrame(data)

    >>> # Autodetect text columns and tokenize:
    >>> df_tokens = tokenize_text_columns(
    ...     df,
    ...     delim=r"\\s+",  # split on whitespace
    ...     to_lower=True,
    ...     remove_empty=True
    ... )
    >>> print(df_tokens["Observations"])
    [['first', 'line', 'of', 'text!'],
     ['second', 'line...', 'more', 'text?']]

    >>> # Specify columns manually:
    >>> df_tokens_2 = tokenize_text_columns(
    ...     df,
    ...     columns=["A"],
    ...     delim=r"\\W+",
    ...     to_lower=True
    ... )
    >>> print(df_tokens_2["A"])
    [['val1'], ['val2']]

    See Also
    --------
    select_text_columns_in :
        Identifies columns likely to contain
        textual data based on sampling and
        threshold checks.

    References
    ----------
    .. [1] M. Hearst et al., *Searching and
           Mining Text*, Journal of Data
           Mining, 2023.

    """
    # 1) Autodetect columns if not provided
    if columns is None and autodetect_text:
        columns = select_text_columns_in(
            df,
            sample_size=sample_size,
            threshold=threshold,
            check_for_words=check_for_words,
            return_frame=False,
            **kw
        )
    elif columns is None:
        # 2) If still None, raise an error
        raise ValueError(
            "No columns specified and "
            "autodetect_text=False. Either "
            "provide a list of columns or set "
            "autodetect_text=True."
        )

    # 3) Handle inplace or copy
    if not inplace:
        df = df.copy()

    # 4) Tokenize each target column
    for col in columns:
        if col not in df.columns:
            continue

        # Ensure string type
        df[col] = df[col].astype(str)

        tokenized_col = []
        for val in df[col]:
            tokens = re.split(delim, val)

            if to_lower:
                tokens = [t.lower() for t in tokens]

            if remove_empty:
                tokens = [t for t in tokens if t]

            tokenized_col.append(tokens)

        df[col] = tokenized_col

    return df

@check_non_emptiness
@isdf
def vectorize_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    vectorizer: str = "count",
    lowercase: bool = True,
    ngram_range: tuple = (1, 1),
    max_features: Optional[int] = None,
    stop_words: Optional[Union[str, List[str]]] = None,
    min_df: float = 1,
    max_df: float = 1.0,
    token_pattern: str = r"(?u)\b\w\w+\b",
    combine_result: bool = False,
    prefix: str = "vec_",
    autodetect_text: bool = False,
    sample_size: int = 50,
    threshold: float = 0.7,
    check_for_words: bool = True,
    inplace: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Convert tokenized text columns into numeric
    feature vectors, returning a DataFrame (or
    modifying it in-place) via
    ``vectorize_text_columns``.

    .. math::
       \\text{features} =
       \\mathrm{Vectorizer}(\\text{tokens})

    This function applies a scikit-learn-based
    vectorization strategy (either Count or TF-IDF)
    to columns of tokenized text [1]_. Each text
    column is transformed into one or more numeric
    columns, representing token frequencies,
    TF-IDF scores, or other features, depending on
    the chosen <vectorizer inline>.

    Parameters
    ----------
    df : pd.DataFrame
        The `parameter inline` DataFrame whose
        tokenized columns will be converted to
        numeric vectors.
    columns : list of str, optional
        The text columns to vectorize. If None,
        no default detection is performed; user
        must supply columns that contain lists
        of tokens.
    vectorizer : str
        The vectorization method: either
        ``"count"`` for :math:`\\mathrm{CountVectorizer}`
        or ``"tfidf"`` for
        :math:`\\mathrm{TfidfVectorizer}`.
        Default ``"count"``.
    lowercase : bool
        If True, convert tokens to lowercase
        before vectorization. Default True.
    ngram_range : tuple
        The lower and upper boundary of the n-gram
        range. For example, ``(1,2)`` includes
        unigrams and bigrams. Default ``(1,1)``.
    max_features : int, optional
        If not None, limit the vocabulary size
        to the top `max_features` tokens by
        frequency (or TF-IDF importance).
        Default None (unlimited).
    stop_words : str or list of str, optional
        Stop words to remove during feature
        extraction. If a string like ``"english"``,
        uses built-in English stop words. If a list,
        it should be a custom list of stopwords.
        Default None (no stop words).
    min_df : float
        Minimum document frequency. Terms that
        appear in fewer than <min_df inline> docs
        are excluded. Can be an absolute count
        or a proportion. Default 1.
    max_df : float
        Maximum document frequency. Terms that
        appear in more than <max_df inline> docs
        are excluded. Can be an absolute count
        or a proportion. Default 1.0 (no limit).
    token_pattern : str
        Regex to identify tokens. Default
        ``r"(?u)\\b\\w\\w+\\b"``. Used internally
        by scikit-learn for splitting.
    combine_result : bool
        If True, combine all vectorized columns
        into a single DataFrame. If False,
        produce separate vector columns for
        each text column. Default False.
    prefix : str
        A prefix used when naming new columns
        in the returned or modified DataFrame.
        Default ``"vec_"``.
    autodetect_text : bool
        If True and <columns inline> is None,
        columns likely containing text are
        autodetected. Default False.
    sample_size : int
        The sample size used by
        ``select_text_columns_in`` if detecting
        text columns. Default 50.
    threshold : float
        The threshold used by
        ``select_text_columns_in`` for text
        detection. Default 0.7.
    check_for_words : bool
        If True, also used in
        ``select_text_columns_in`` to confirm
        the presence of alphabetic characters
        in a sampled entry. Default True.
        
    inplace : bool
        If True, modify <df inline> directly by
        adding the new numeric columns (and
        possibly removing or replacing the
        original text columns). If False,
        return a new DataFrame. Default False.
    **kwargs :
        Additional keyword arguments passed to
        the underlying scikit-learn vectorizer
        (e.g., ``binary=True``, etc.).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the vectorized
        feature columns. If <inplace inline> is
        True, the changes occur in the original
        <df inline>; otherwise, a new DataFrame
        is returned.

    Notes
    -----
    1. The input <columns inline> should already
       contain lists of tokens (e.g., created
       by :func:`tokenize_text_columns`).
    2. When <combine_result inline> is False,
       the function appends separate numeric
       columns for each text column to the
       DataFrame. If True, it merges the vector
       outputs for all text columns into one set
       of feature columns, ensuring a single
       vocabulary is built across them.
    3. If a user wants entirely separate
       vectorizers for each column, set
       <combine_result inline> to False; if a
       single integrated vocabulary is desired,
       set <combine_result inline> to True.

    Examples
    --------
    >>> from gofast.utils.text import tokenize_text_columns
    >>> from gofast.utils.text import vectorize_text_columns
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'doc1': [['hello', 'world'],
    ...              ['hello', 'machine', 'learning']],
    ...     'doc2': [['foo', 'bar'], ['bar', 'baz']]
    ... })
    >>> # Vectorize columns doc1 and doc2:
    >>> df_vec = vectorize_text_columns(
    ...     df,
    ...     columns=['doc1', 'doc2'],
    ...     vectorizer='tfidf',
    ...     combine_result=False
    ... )
    >>> print(df_vec.columns)
    Index(['doc1', 'doc2', 'vec_doc1_0', 'vec_doc1_1', 'vec_doc1_2',
           'vec_doc2_0', 'vec_doc2_1', 'vec_doc2_2'],
          dtype='object')

    See Also
    --------
    tokenize_text_columns:
        Splits textual columns into lists of tokens
        suitable for vectorization.
    select_text_columns_in :
        Detects columns likely to be textual
        based on sampling and thresholds.
    gofast.dataops.transformation.summarize_text_columns: 
        Applies extractive summarization to specified text 
        columns in a pandas DataFrame. 

    References
    ----------
    .. [1] G. Salton and C. Buckley, "Term-Weighting
           Approaches in Automatic Text Retrieval",
           Information Processing & Management, 1988.

    """
    # 1) Autodetect columns if not provided
    if columns is None and autodetect_text:
        columns = select_text_columns_in(
            df,
            sample_size=sample_size,
            threshold=threshold,
            check_for_words=check_for_words,
            return_frame=False,
        )
        
    if columns is None:
        raise ValueError(
            "No columns specified. Please provide a list of "
            "columns containing tokenized text or set"
            " ``autodetect_text=True``."
        )
    # Decide which vectorizer to use
    if vectorizer.lower() == "count":
        vec_class = CountVectorizer
    elif vectorizer.lower() == "tfidf":
        vec_class = TfidfVectorizer
    else:
        raise ValueError(
            f"Unknown vectorizer type: {vectorizer}. "
            "Choose 'count' or 'tfidf'."
        )

    # Depending on inplace, either modify df or copy
    if not inplace:
        df = df.copy()

    if combine_result:
        # Combine text from all specified columns
        combined_corpus = []
        for idx, row in df.iterrows():
            # Merge tokens from all columns into one list
            merged_tokens = []
            for col in columns:
                if isinstance(row[col], list):
                    merged_tokens.extend(row[col])
                else:
                    # If the cell isn't a list,
                    # treat it as empty or skip
                    continue
            combined_corpus.append(
                " ".join(
                    token.lower() if lowercase else token
                    for token in merged_tokens
                )
            )

        # Build and fit the vectorizer
        vectorizer_obj = vec_class(
            lowercase=False,  # we already do .lower() above if needed
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            token_pattern=token_pattern,
            **kwargs
        )
        matrix = vectorizer_obj.fit_transform(combined_corpus)

        # Convert to DataFrame
        feature_names = vectorizer_obj.get_feature_names_out()
        matrix_df = pd.DataFrame(
            matrix.toarray(),
            columns=[f"{prefix}{name}" for name in feature_names]
        )

        # Concatenate new feature columns
        df = pd.concat([df.reset_index(drop=True),
                        matrix_df.reset_index(drop=True)],
                       axis=1)
    else:
        # Vectorize each column separately
        for col in columns:
            if col not in df:
                continue

            col_values = []
            for val in df[col]:
                if isinstance(val, list):
                    # Convert the list of tokens to a space-separated string
                    text_str = " ".join(
                        token.lower() if lowercase else token
                        for token in val
                    )
                else:
                    # If the cell isn't a list, treat as empty
                    text_str = ""

                col_values.append(text_str)

            # Build and fit the vectorizer for this column
            vectorizer_obj = vec_class(
                lowercase=False,
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words=stop_words,
                min_df=min_df,
                max_df=max_df,
                token_pattern=token_pattern,
                **kwargs
            )
            matrix = vectorizer_obj.fit_transform(col_values)

            # Convert to DataFrame
            feature_names = vectorizer_obj.get_feature_names_out()
            matrix_df = pd.DataFrame(
                matrix.toarray(),
                columns=[f"{prefix}{col}_{i}" for i in range(len(feature_names))]
            )

            # Merge back into df
            df = pd.concat([df.reset_index(drop=True),
                            matrix_df.reset_index(drop=True)],
                           axis=1)

    return df