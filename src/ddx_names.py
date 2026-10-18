"""Canonical DDXPlus condition names, shared by evaluation scoring and training rewards.

Free-text names are mapped to one of the 49 conditions by, in order:
exact match after normalization, alias table, the same with qualifiers ("acute", "possible",
parentheticals) stripped, then fuzzy match (WRatio >= 88). Returns None when nothing matches.
"""
import re

from rapidfuzz import fuzz, process

ALIASES = {
    "nstemi": "Possible NSTEMI / STEMI", "stemi": "Possible NSTEMI / STEMI",
    "myocardial infarction": "Possible NSTEMI / STEMI", "acute coronary syndrome": "Possible NSTEMI / STEMI",
    "asthma": "Bronchospasm / acute asthma exacerbation", "asthma exacerbation": "Bronchospasm / acute asthma exacerbation",
    "bronchospasm": "Bronchospasm / acute asthma exacerbation",
    "copd exacerbation": "Acute COPD exacerbation / infection", "copd": "Acute COPD exacerbation / infection",
    "laryngospasm": "Larygospasm",  # DDXPlus's own condition name is misspelled
    "upper respiratory tract infection": "URTI", "common cold": "URTI",
    "pharyngitis": "Viral pharyngitis",
    "supraventricular tachycardia": "PSVT", "paroxysmal supraventricular tachycardia": "PSVT",
    "systemic lupus erythematosus": "SLE", "lupus": "SLE",
    "gastroesophageal reflux disease": "GERD", "gerd": "GERD",
    "boerhaave syndrome": "Boerhaave",
    "hiv": "HIV (initial infection)", "acute hiv infection": "HIV (initial infection)",
    "pneumothorax": "Spontaneous pneumothorax", "rib fracture": "Spontaneous rib fracture",
    "pulmonary edema": "Acute pulmonary edema", "otitis media": "Acute otitis media",
    "pertussis": "Whooping cough", "chagas disease": "Chagas", "ebola virus disease": "Ebola",
    "anemia": "Anemia", "anaemia": "Anemia", "afib": "Atrial fibrillation",
    "lung cancer": "Pulmonary neoplasm", "pancreatic cancer": "Pancreatic neoplasm",
    "guillain barre syndrome": "Guillain-Barré syndrome",
}
_QUALIFIERS = re.compile(r"^(acute|possible|suspected|probable)\s+")


def norm(s):
    s = s.lower().replace("é", "e")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


class Matcher:
    def __init__(self, conditions):
        self.conditions = frozenset(conditions)
        self.by_norm = {norm(c): c for c in conditions}
        self.by_norm.update({norm(k): v for k, v in ALIASES.items()})
        self.keys = list(self.by_norm)
        self._cache = {}

    def _exact(self, name):
        for variant in (name, re.sub(r"\([^)]*\)", "", name)):
            k = norm(variant)
            for key in (k, _QUALIFIERS.sub("", k)):
                if key in self.by_norm:
                    return self.by_norm[key]
        return None

    def __call__(self, name):
        if name in self.conditions:
            return name
        if name not in self._cache:
            hit = self._exact(name)
            if hit is None:
                f = process.extractOne(norm(name), self.keys, scorer=fuzz.WRatio, score_cutoff=88)
                hit = self.by_norm[f[0]] if f else None
            self._cache[name] = hit
        return self._cache[name]
