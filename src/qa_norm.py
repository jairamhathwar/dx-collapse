"""Answer normalization and grading for short-answer QA (AmbigQA).

normalize: SQuAD-style (lowercase, drop punctuation and articles, squeeze spaces).
Grader(case): maps an answer string to a key, so sampled answers can be counted exactly:
    "gold:<i>"  the string matches an alias of the case's i-th acceptable answer (one per interpretation)
    "txt:<s>"   otherwise, its normalized text
A key is correct if it starts with "gold:". Two strings share a key if they match the same acceptable answer or
normalize to the same text; wrong paraphrases ("NYC" / "New York City") stay separate, which overstates disagreement
among wrong answers (audit by hand before reporting).
"""
import re
import string

_ART = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s):
    s = _ART.sub(" ", s.lower().translate(_PUNCT))
    return " ".join(s.split())


class Grader:
    def __init__(self, case):
        self.alias = {}
        for i, aliases in enumerate(case["answers"]):
            for a in aliases:
                self.alias.setdefault(normalize(a), i)
        self.alias.pop("", None)

    def key(self, s):
        n = normalize(s)
        if not n:
            return None
        return f"gold:{self.alias[n]}" if n in self.alias else f"txt:{n}"

    @staticmethod
    def correct(key):
        return key is not None and key.startswith("gold:")
