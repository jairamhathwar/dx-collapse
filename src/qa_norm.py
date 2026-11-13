"""Answer normalization and grading for short-answer QA (AmbigQA).

normalize: SQuAD-style (lowercase, drop punctuation and articles, squeeze spaces).
Grader(case, lenient=True) maps the answers of one question's samples to keys, so they can be counted exactly:
    "gold:<i>"  the answer matches the case's i-th acceptable answer (one per interpretation)
    "txt:<s>"   otherwise, a representative normalized text
A key is correct if it starts with "gold:".

strict (lenient=False): exact match of normalized strings (the usual EM), no merging of wrong answers.
lenient (default), after a hand audit of exact match (16% of questions had a right answer graded wrong, 17% split
one wrong answer into paraphrases):
  - aliases joined by "|" in the data are split;
  - lead-ins are dropped ("voiced by X", "in 1998", "around 607");
  - dates are parsed to (year, month, day); an answer date matches a gold date if every field both give agrees and
    the answer gives at least the year, or month and day ("1998" matches "11 May 1998"; "January 6th" matches
    "January 6"; "1999" does not match "2000");
  - containment: two strings match if the shorter one's tokens appear in order, contiguously, in the longer one, the
    shorter has at least 3 characters and covers at least half of the longer one's tokens ("continental crust" /
    "continental"; "ronaldo" / "ronaldo nazario"; not "vogtle" / "vogtle electric generating plant").
  Wrong answers of one question are merged by the same date and containment rules (Grader.keys), most frequent first.
Lenient matching can accept a wrong answer that contains a right one's words; report strict and lenient together.
"""
import re
import string
import unicodedata
from collections import Counter

_ART = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_LEAD = re.compile(r"^(voiced by|played by|portrayed by|sung by|performed by|written by|directed by|composed by|"
                   r"created by|founded by|in|on|at|by|around|about|approximately|circa|roughly|nearly|over)\s+")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"])}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def normalize(s):
    s = _ART.sub(" ", s.lower().translate(_PUNCT))
    return " ".join(s.split())


def _fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def parse_date(raw):
    """(year, month, day) with None for missing fields, or None if the string is not just a date."""
    s = _fold(raw.lower()).strip()
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return (y, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    toks = re.sub(r"[,.]", " ", s).replace(" of ", " ").split()
    y = mo = d = None
    for t in toks:
        if t in _MONTHS and mo is None:
            mo = _MONTHS[t]
        elif re.fullmatch(r"\d{4}", t) and y is None and 1000 <= int(t) <= 2100:
            y = int(t)
        elif re.fullmatch(r"\d{1,2}", t) and d is None and 1 <= int(t) <= 31:
            d = int(t)
        else:
            return None
    if y is None and (mo is None or d is None):
        return None
    if d is not None and mo is None:
        return None
    return (y, mo, d)


def dates_match(a, b):
    return all(x == y for x, y in zip(a, b) if x is not None and y is not None) and \
        any(x is not None and y is not None for x, y in zip(a, b))


def canon(s):
    """Lenient text form: accents folded, lead-ins dropped, SQuAD-normalized."""
    n = normalize(_fold(s))
    for _ in range(2):
        n = _LEAD.sub("", n)
    return n


def contains(a, b):
    """Containment match between two canonical strings (see module docstring)."""
    if a == b:
        return True
    short, long_ = sorted((a.split(), b.split()), key=len)
    if not short or len("".join(short)) < 3 or 2 * len(short) < len(long_):
        return False
    return any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1))


def same(a, b):
    """a, b: (canonical text, parsed date or None)."""
    if a[1] and b[1]:
        return dates_match(a[1], b[1])
    if a[1] or b[1]:
        return a[0] == b[0]
    return contains(a[0], b[0])


class Grader:
    def __init__(self, case, lenient=True):
        self.lenient = lenient
        self.exact, self.gold = {}, []
        for i, aliases in enumerate(case["answers"]):
            for a in aliases:
                for part in (a.split("|") if lenient else [a]):
                    self.exact.setdefault(normalize(part), i)
                    if lenient and canon(part):
                        self.gold.append((i, (canon(part), parse_date(part))))
        self.exact.pop("", None)

    def _form(self, s):
        return (canon(s), parse_date(s))

    def _gold(self, s):
        n = normalize(s)
        if n in self.exact:
            return self.exact[n]
        if self.lenient:
            f = self._form(s)
            for i, g in self.gold:
                if f[0] and same(f, g):
                    return i
        return None

    def key(self, s):
        """Key of one answer on its own (no merging of wrong paraphrases)."""
        if not normalize(s):
            return None
        i = self._gold(s)
        if i is not None:
            return f"gold:{i}"
        return f"txt:{canon(s) if self.lenient else normalize(s)}"

    def keys(self, answers):
        """Keys for the answers of one question's samples (None stays None). Lenient: wrong answers are merged with
        the most frequent wrong answer they match."""
        ks = [self.key(a) if a is not None else None for a in answers]
        if not self.lenient:
            return ks
        wrong = Counter(a for a, k in zip(answers, ks) if k and k.startswith("txt:"))
        reps, rep_of = [], {}
        for a, _ in wrong.most_common():
            f = self._form(a)
            hit = next((r for r, rf in reps if same(f, rf)), None)
            if hit is None:
                reps.append((a, f))
                hit = a
            rep_of[a] = hit
        return [f"txt:{canon(rep_of[a])}" if k and k.startswith("txt:") else k for a, k in zip(answers, ks)]

    @staticmethod
    def correct(key):
        return key is not None and key.startswith("gold:")
