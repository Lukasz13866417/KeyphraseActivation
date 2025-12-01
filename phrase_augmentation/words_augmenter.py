from typing import Generator
from random import choice, seed, random
from wordfreq import top_n_list, iter_wordlist

seed(12345) # for reproducibility

WORDS = top_n_list("en", 100_000)

def add_words_between_words(phrase: str, count: int) -> Generator[str, None, None]:
    """ Insert random words between words in a phrase."""
    words = phrase.split(" ")
    # If there are no gaps, just yield the original phrase
    if len(words) < 2:
        for _ in range(count):
            yield phrase
        return
    num_gaps = len(words) - 1  # gaps between words
    for _ in range(count):
        # Choose a random gap index where we MUST insert at least one word
        forced_gap = int(random() * num_gaps)  # 0 .. num_gaps-1
        out: list[str] = [words[0]]
        for gi in range(num_gaps):
            if gi == forced_gap:
                # Insert 1 or 2 words at the forced gap
                out.append(choice(WORDS))
                if random() < 0.5:
                    out.append(choice(WORDS))
            else:
                # After the forced gap, continue with the usual random insertion logic
                if gi > forced_gap and random() < 0.5:
                    out.append(choice(WORDS))
                    if random() < 0.5:
                        out.append(choice(WORDS))
            # Append the next original word
            out.append(words[gi + 1])
        yield " ".join(out)
