"""General utility functions and classes."""

import re


def remove_digits(s: str) -> str:
    """Remove all digits, 0-9, from string.

    Parameters
    ----------
    s
        String from which any digits to be removed.

    Examples
    --------
    >>> remove_digits("asdf23ll2")
    'asdfll'
    >>> remove_digits("2s20a!sdf2s,2")
    'sa!sdfs,'
    """
    return re.sub(r"\d", "", s)


def remove_nondigits(s: str) -> str:
    """Remove all non-digits (i.e. not 0-9) from string.

    Parameters
    ----------
    s
        String from which any non-digits to be removed.

    Examples
    --------
    >>> remove_nondigits("asdf23ll2s")
    '232'
    >>> remove_nondigits("1a2b33cc4!!?/5£6€0")
    '12334560'
    """
    return re.sub(r"\D", "", s)
