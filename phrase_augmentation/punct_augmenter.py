from typing import Generator
from random import choice, seed, random
from wordfreq import top_n_list, iter_wordlist
from words_augmenter import add_words_between_words

PUNCTUATION = [".", ",", "!", "?", "?!"]

def add_punct(phrase: str, count: int, replace_existing: bool = False) -> Generator[str, None, None]:
    """Add punctuation to a phrase.
    """
    places = phrase.split(" ")
    for _ in range(count):
        res = ""
        for place in places:
            res += place
            if random() < 0.5:
                if res[-1] in PUNCTUATION and replace_existing:
                    res = res[:-1] + choice(PUNCTUATION)
                elif res[-1] not in PUNCTUATION:
                    res += choice(PUNCTUATION)
            res += " "
        yield res.strip()

WORDS = top_n_list("en", 100_000)




if __name__ == "__main__":
    for phrase in add_punct("Hello world! This is a test.", 3, replace_existing=True):
        print(phrase)
    for phrase in add_words_between_words("Hello world! This is a test.", 30):
        print(phrase)