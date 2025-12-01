from functools import lru_cache
from typing import Iterable, Sequence, Tuple
from wordfreq import top_n_list, iter_wordlist


def assert_positive(this_phrase: str, key_phrase: str) -> bool:
    return key_phrase in this_phrase


def assert_negative(this_phrase: str, key_phrase: str) -> bool:
    return key_phrase not in this_phrase


@lru_cache(maxsize=32)
def get_word_base(lang: str = "en", n: int = 100_000, lowercase: bool = True) -> Tuple[str, ...]:
    """
    Return a cached, immutable word list (tuple) of the top-N words for a language.
    - Uses wordfreq.top_n_list under the hood
    - Cached via lru_cache so repeated calls do not copy data
    """
    words = top_n_list(lang, n)
    if lowercase:
        words = [w.lower() for w in words]
    return tuple(words)


def iter_word_base(lang: str = "en") -> Iterable[str]:
    """
    Generator over a language's word list without loading everything into memory.
    Useful because we don't want to hold the full list in RAM.
    """
    return iter_wordlist(lang)