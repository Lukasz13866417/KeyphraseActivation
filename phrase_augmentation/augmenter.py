import json
from random import randint, random, choice, sample
from typing import List, Dict, Tuple, Optional

from .util import get_word_base
from .confusers_generator import phrase_phonemes, phoneme_edit_distance, phrase_metaphone
from rapidfuzz.distance import Levenshtein


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _nearest_confusers_for_word(
    word: str,
    *,
    word_base: Tuple[str, ...],
    max_candidates: int = 5000,
    top_k: int = 30,
) -> List[str]:
    """Return up to top_k words from the base that are phonetically closest to the given word.
    When possible, we use phoneme-distance based nearest. Otherwise, we fall back to string-level algorithms."""
    if not word:
        return []
    candidates = list(word_base[:max_candidates]) if len(word_base) >= max_candidates else list(word_base)
    if len(candidates) > max_candidates:
        candidates = sample(candidates, k=max_candidates)
    wl = word.lower()
    candidates = [w for w in candidates if w.lower() != wl] # Remove the original word (exact match)
    scored: List[Tuple[float, str]] = []
    try:
        w_phones = phrase_phonemes(word)
        for cand in candidates:
            try:
                c_phones = phrase_phonemes(cand)
                d = phoneme_edit_distance(w_phones, c_phones)
                scored.append((d, cand))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0])
        if scored:
            return [c for _, c in scored[:top_k]]
    except Exception:
        pass
    # Fallback: string-level metaphone/Levenshtein
    try:
        w_meta = phrase_metaphone(word)
        metas = []
        for cand in candidates:
            try:
                cm = phrase_metaphone(cand)
                metas.append((cand, cm))
            except Exception:
                continue
        same_meta = [c for (c, cm) in metas if cm == w_meta and c.lower() != word.lower()]
        if same_meta:
            # Rank same-meta by edit distance
            ranked = sorted(same_meta, key=lambda c: Levenshtein.distance(word, c))
            return ranked[:top_k]
        # Otherwise take global nearest by edit distance
        ranked_all = sorted((c for c, _ in metas), key=lambda c: Levenshtein.distance(word, c))
        return [c for c in ranked_all[:top_k]]
    except Exception:
        return []


def generate_positives(
    key_phrase: str,
    n: int,
    *,
    p_extra_start: float = 0.5,
    p_extra_end: float = 0.5,
    lang: str = "en",
    word_base_size: int = 100_000,
) -> List[str]:
    """
    Return n positives. Optionally add one extra word at the start/end with given probabilities.
    """
    base = get_word_base(lang, word_base_size)
    out: List[str] = []
    for _ in range(max(0, n)):
        parts: List[str] = []
        if random() < p_extra_start:
            parts.append(choice(base))
        parts.append(key_phrase)
        if random() < p_extra_end:
            parts.append(choice(base))
        out.append(" ".join(parts))
    return out


def generate_confusers(
    key_phrase: str,
    n: int,
    *,
    lang: str = "en",
    word_base_size: int = 100_000,
    per_word_topk: int = 20,
) -> List[str]:
    """Generate phrases that differ from the key phrase by replacing a single word with a close phonetic confuser."""
    words = [w for w in key_phrase.strip().split() if w]
    if not words:
        return []
    base = get_word_base(lang, word_base_size)
    out: List[str] = []
    attempts = 0
    # Try generating until we have n unique confusers or hit a cap
    while len(out) < n and attempts < n * 50:
        attempts += 1
        i = randint(0, len(words) - 1)
        target = words[i]
        confs = _nearest_confusers_for_word(target, word_base=base, top_k=per_word_topk)
        if not confs:
            continue
        repl = choice(confs)
        cand_words = list(words)
        cand_words[i] = repl
        cand = " ".join(cand_words)
        if cand != key_phrase:
            out.append(cand)
    return _unique_keep_order(out)[:n]


def generate_inbetween(
    key_phrase: str,
    n: int,
    *,
    lang: str = "en",
    word_base_size: int = 100_000,
    confuser_inbetween_prob: float = 0.5,
    per_word_topk: int = 15,
    max_inserts_per_gap: int = 2,
) -> List[str]:
    """
    Insert 1–2 words in random gaps of the key phrase; optionally use confuser words (phonetically close to neighbors).
    This is used to generate in-between phrases that are not too similar to the key phrase.
    """
    words = [w for w in key_phrase.strip().split() if w]
    if len(words) < 2:
        return [key_phrase] * max(0, n)
    base = get_word_base(lang, word_base_size)
    out: List[str] = []
    num_gaps = len(words) - 1
    for _ in range(n):
        forced_gap = randint(0, num_gaps - 1)
        assembled: List[str] = [words[0]]
        for gi in range(num_gaps):
            insert_here = (gi == forced_gap) or (gi > forced_gap and random() < 0.5)
            if insert_here:
                if gi == forced_gap:
                    base_inserts = 1 + (1 if random() < 0.5 else 0)
                else:
                    base_inserts = 1 if random() < 0.5 else 0
                    if base_inserts > 0 and random() < 0.5:
                        base_inserts += 1
                n_ins = max(1 if gi == forced_gap else 0, min(max_inserts_per_gap, base_inserts))
                for _k in range(n_ins):
                    if random() < confuser_inbetween_prob:
                        neighbor = words[gi] if random() < 0.5 else words[gi + 1]
                        confs = _nearest_confusers_for_word(neighbor, word_base=base, top_k=per_word_topk)
                        tok = choice(confs) if confs else choice(base)
                    else:
                        tok = choice(base)
                    assembled.append(tok)
            assembled.append(words[gi + 1])
        out.append(" ".join(assembled))
    return out


def generate_augmented_phrases(
    key_phrase: str,
    num_confusers: int,
    num_positives: int,
    num_inbetween: int,
    *,
    lang: str = "en",
    word_base_size: int = 100_000,
    confuser_inbetween_prob: float = 0.5,
    p_pos_extra_start: float = 0.5,
    p_pos_extra_end: float = 0.5,
    max_inserts_per_gap: int = 2,
) -> Dict[str, List[str]]:
    positives = generate_positives(
        key_phrase, num_positives,
        p_extra_start=p_pos_extra_start,
        p_extra_end=p_pos_extra_end,
        lang=lang,
        word_base_size=word_base_size,
    )
    confusers = generate_confusers(
        key_phrase, num_confusers, lang=lang, word_base_size=word_base_size
    )
    inbetween = generate_inbetween(
        key_phrase, num_inbetween, lang=lang, word_base_size=word_base_size,
        confuser_inbetween_prob=confuser_inbetween_prob,
        max_inserts_per_gap=max_inserts_per_gap,
    )
    return {
        "positives": positives,
        "confusers": confusers,
        "inbetween": inbetween,
    }


if __name__ == "__main__":
    kp = input("Key phrase: ").strip()
    try:
        n_conf = int(input("Number of confusers: ").strip() or "10")
    except Exception:
        n_conf = 10
    try:
        n_pos = int(input("Number of positives: ").strip() or "10")
    except Exception:
        n_pos = 10
    try:
        n_inb = int(input("Number of in-between phrases: ").strip() or "10")
    except Exception:
        n_inb = 10
    try:
        p_inb_conf = float(input("Probability inserted tokens are confusers [0-1] (default 0.5): ").strip() or "0.5")
    except Exception:
        p_inb_conf = 0.5
    try:
        p_pos_start = float(input("Positives: prob extra start word [0-1] (default 0.5): ").strip() or "0.5")
    except Exception:
        p_pos_start = 0.5
    try:
        p_pos_end = float(input("Positives: prob extra end word [0-1] (default 0.5): ").strip() or "0.5")
    except Exception:
        p_pos_end = 0.5
    try:
        max_ins = int(input("Max inserts per gap (default 2): ").strip() or "2")
    except Exception:
        max_ins = 2
    result = generate_augmented_phrases(
        kp, n_conf, n_pos, n_inb,
        confuser_inbetween_prob=p_inb_conf,
        p_pos_extra_start=p_pos_start,
        p_pos_extra_end=p_pos_end,
        max_inserts_per_gap=max_ins,
    )
    print(json.dumps(result, indent=2))


