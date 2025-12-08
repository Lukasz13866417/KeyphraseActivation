import random as py_random
import sys
from itertools import cycle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phrase_augmentation import augmenter, plain_negatives, punct_augmenter


def _fake_word_base(*_, **__) -> tuple[str, ...]:
    return ("alpha", "beta", "gamma", "delta", "epsilon")


def test_generate_positives_adds_leading_tokens(monkeypatch):
    py_random.seed(0)
    monkeypatch.setattr(augmenter, "get_word_base", _fake_word_base)

    phrases = augmenter.generate_positives(
        "wake word", 3, p_extra_start=1.0, p_extra_end=0.0
    )

    assert len(phrases) == 3
    for phrase in phrases:
        parts = phrase.split()
        assert parts[0] in _fake_word_base()
        assert "wake word" in phrase
        assert phrase.count("wake word") == 1


def test_generate_confusers_replaces_exactly_one_word(monkeypatch):
    py_random.seed(1)
    monkeypatch.setattr(augmenter, "get_word_base", _fake_word_base)
    monkeypatch.setattr(
        augmenter,
        "_nearest_confusers_for_word",
        lambda word, **_: [f"{word}_conf"],
    )

    confusers = augmenter.generate_confusers("wake word", 5)

    assert confusers
    base_words = ["wake", "word"]
    for phrase in confusers:
        words = phrase.split()
        assert len(words) == len(base_words)
        diff = sum(1 for left, right in zip(words, base_words) if left != right)
        assert diff == 1


def test_generate_inbetween_inserts_tokens(monkeypatch):
    py_random.seed(2)
    monkeypatch.setattr(augmenter, "get_word_base", _fake_word_base)
    monkeypatch.setattr(
        augmenter,
        "_nearest_confusers_for_word",
        lambda word, **_: [f"{word}_conf"],
    )

    phrases = augmenter.generate_inbetween(
        "wake friendly word", 2, max_inserts_per_gap=2, confuser_inbetween_prob=1.0
    )

    assert len(phrases) == 2
    for phrase in phrases:
        tokens = phrase.split()
        assert tokens[0] == "wake"
        assert tokens[-1] == "word"
        assert len(tokens) > 3


def test_add_punct_can_replace_existing(monkeypatch):
    random_values = iter([0.0, 0.0])

    def fake_random():
        try:
            return next(random_values)
        except StopIteration:
            return 1.0

    punct_cycle = cycle(["?", "!"])

    monkeypatch.setattr(punct_augmenter, "random", fake_random)
    monkeypatch.setattr(
        punct_augmenter,
        "choice",
        lambda _: next(punct_cycle),
    )

    augmented = next(punct_augmenter.add_punct("hello! world", 1, replace_existing=True))

    assert augmented == "hello? world!"


def test_plain_negatives_avoid_keyphrase(monkeypatch):
    py_random.seed(3)

    phrases = plain_negatives.generate_plain_negatives(
        n=3,
        key_phrase="wake word",
        k=2,
        sources=("alpha beta gamma delta epsilon zeta eta theta iota kappa",),
    )

    assert len(phrases) == 3
    for text in phrases:
        assert "wake word" not in text.lower()

