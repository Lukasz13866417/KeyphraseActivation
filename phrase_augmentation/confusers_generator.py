from collections import defaultdict
from functools import lru_cache
import jellyfish
from rapidfuzz.distance import Levenshtein
from g2p_en import G2p

g2p = G2p()

def double_metaphone_word(w):
    # jellyfish has metaphone; for double metaphone you can use 'fuzzy' or 'Metaphone' pkg;
    # jellyfish.metaphone is fine as a fast proxy. If you want true double metaphone, use 'metaphone' lib.
    return jellyfish.metaphone(w)

def phrase_metaphone(phrase):
    return " ".join(double_metaphone_word(w) for w in phrase.lower().split())

@lru_cache(maxsize=100_000)
def phrase_phonemes(phrase):
    # g2p_en returns ['S', 'AH0', 'M', ' ... ', ' ']
    phones = [p for p in g2p(phrase) if p.strip() and p != ' ']
    return tuple(phones)

def phones_to_string(phones):
    return " ".join(phones)

SIMILAR = {
    # Stops (voicing pairs)
    ('P','B'):0.25, ('T','D'):0.25, ('K','G'):0.25,
    # Fricatives (voicing pairs)
    ('S','Z'):0.25, ('F','V'):0.25, ('TH','DH'):0.30,
    # Sibilants/affricates
    ('SH','ZH'):0.25, ('CH','JH'):0.35, ('CH','SH'):0.20, ('JH','ZH'):0.20,
    # Nasals and approximants (often confused in noise or coarticulation)
    ('M','N'):0.30, ('N','NG'):0.30, ('L','R'):0.40, ('W','V'):0.15, ('Y','IY'):0.15,
    # Vowels (close/neighboring)
    ('AA','AH'):0.35, ('EH','AE'):0.35, ('IH','IY'):0.35, ('UH','UW'):0.35,
    ('AO','AA'):0.30, ('AO','OW'):0.30, ('AH','AX'):0.40, ('EH','EY'):0.30,
}
def sub_cost(a,b):
    if a==b: return 0
    a0,b0 = a.split('0')[0].split('1')[0].split('2')[0], b.split('0')[0].split('1')[0].split('2')[0]
    if (a0,b0) in SIMILAR or (b0,a0) in SIMILAR:
        return SIMILAR.get((a0,b0), SIMILAR.get((b0,a0), 0.5))
    return 1.0

def phoneme_edit_distance(a, b):
    # RapidFuzz's Levenshtein doesn't support a matrix; do a tiny custom DP.
    # a,b are tuples of phones.
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        ai = a[i-1]
        for j in range(1, lb+1):
            pj = dp[j]
            cost_sub = sub_cost(ai, b[j-1])
            dp[j] = min(
                dp[j] + 1,         # deletion
                dp[j-1] + 1,       # insertion
                prev + cost_sub    # substitution
            )
            prev = pj
    return dp[-1]

class ConfuserIndex:
    def __init__(self, phrases):
        self.phrases = list(phrases)
        self.meta_index = defaultdict(list)     # metaphone token -> [ids]
        self.phonemes = []
        for i, ph in enumerate(self.phrases):
            meta = phrase_metaphone(ph)
            for tok in meta.split():
                self.meta_index[tok].append(i)
            self.phonemes.append(phrase_phonemes(ph))

    def candidates(self, query, max_candidates=1000):
        meta = phrase_metaphone(query)
        toks = meta.split()
        seen = set()
        cand = []
        for t in toks:
            for i in self.meta_index.get(t, []):
                if i not in seen:
                    seen.add(i)
                    cand.append(i)
                    if len(cand) >= max_candidates:
                        return cand
        return cand

    def nearest(self, query, k=20):
        q_phones = phrase_phonemes(query)
        cands = self.candidates(query, max_candidates=5000)  # cheap filter
        scored = []
        for i in cands:
            d = phoneme_edit_distance(q_phones, self.phonemes[i])
            scored.append((d, i))
        scored.sort(key=lambda x: x[0])
        return [(self.phrases[i], d) for d,i in scored[:k]]