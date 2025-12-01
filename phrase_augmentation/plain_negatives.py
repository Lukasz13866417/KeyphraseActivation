"""
Utilities for generating phrase-level negatives that deliberately avoid the keyphrase.

We take a handful of public-domain English excerpts, build simple k-gram (Markov)
chains over their tokens, and sample fluent-ish sentences that should *not* contain
the provided keyphrase.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

# TODO add more excerpts
_DEFAULT_EXCERPTS: Tuple[str, ...] = (
    (
        "It is a truth universally acknowledged, that a single man in possession of "
        "a good fortune, must be in want of a wife. However little known the feelings "
        "or views of such a man may be on his first entering a neighbourhood, this "
        "truth is so well fixed in the minds of the surrounding families, that he is "
        "considered as the rightful property of someone or other of their daughters."
    ),
    (
        "It was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
        "incredulity, it was the season of Light, it was the season of Darkness, it was "
        "the spring of hope, it was the winter of despair."
    ),
    (
        "The mysterious affair had exercised a most potent fascination over her. "
        "She loved to hear Poirot talk about it, to see him arranging and re-arranging "
        "the facts in his orderly little grey cells, and she had the most complete "
        "confidence that sooner or later he would find the solution."
    ),
)

_PUNCT_RE = re.compile(r"[^\w\s']")
_TOKEN_RE = re.compile(r"[a-zA-Z']+|[.,;:!?]")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization preserving simple punctuation."""
    return _TOKEN_RE.findall(text.lower())


def _build_chain(tokens: Sequence[str], k: int) -> Dict[Tuple[str, ...], List[str]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(tokens) <= k:
        return {}
    chain: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for idx in range(len(tokens) - k):
        state = tuple(tokens[idx : idx + k])
        nxt = tokens[idx + k]
        chain[state].append(nxt)
    return chain


def _build_chain_from_text(text: str, k: int) -> Dict[Tuple[str, ...], List[str]]:
    tokens = _tokenize(text)
    return _build_chain(tokens, k)


@lru_cache(maxsize=64)
def _cached_chain(source_idx: int, k: int) -> Dict[Tuple[str, ...], List[str]]:
    excerpt = _DEFAULT_EXCERPTS[source_idx]
    return _build_chain_from_text(excerpt, k)


def _detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip().capitalize()


def _generate_once(chain: Dict[Tuple[str, ...], List[str]], k: int, max_words: int) -> Optional[str]:
    if not chain:
        return None
    state = random.choice(list(chain.keys()))
    generated = list(state)
    while len(generated) < max_words:
        state = tuple(generated[-k:])
        options = chain.get(state)
        if not options:
            break
        generated.append(random.choice(options))
    return _detokenize(generated)


def generate_plain_negatives(
    *,
    n: int,
    key_phrase: str,
    k: int = 2,
    max_words: int = 18,
    sources: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Generate up to `n` phrases that intentionally avoid `key_phrase`.
    It's a k-gram (markov chain babble)
    """
    if n <= 0:
        return []
    key_phrase_norm = key_phrase.strip().lower()
    if not key_phrase_norm:
        raise ValueError("key_phrase must be non-empty")
    excerpt_pool: Sequence[str] = tuple(sources) if sources else _DEFAULT_EXCERPTS
    if not excerpt_pool:
        raise ValueError("At least one text source is required.")

    results: List[str] = []
    attempts = 0
    max_attempts = n * 10

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        source_idx = random.randrange(len(excerpt_pool))
        if sources:
            chain = _build_chain_from_text(excerpt_pool[source_idx], k)
        else:
            chain = _cached_chain(source_idx, k)
        candidate = _generate_once(chain, k, max_words)
        if not candidate:
            continue
        if key_phrase_norm in candidate.lower():
            continue
        results.append(candidate)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate keyphrase-free negative sentences.")
    parser.add_argument("key_phrase", help="Key phrase to avoid")
    parser.add_argument("-n", type=int, default=5, help="Number of negatives")
    parser.add_argument("-k", type=int, default=2, help="Markov chain order")
    parser.add_argument("--max-words", type=int, default=18, help="Maximum words per phrase")
    args = parser.parse_args()

    negatives = generate_plain_negatives(
        n=args.n,
        key_phrase=args.key_phrase,
        k=args.k,
        max_words=args.max_words,
    )
    for line in negatives:
        print(line)

