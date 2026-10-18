"""Map free-text disease names to Mondo IDs and grade them against a gold OMIM ID.

Grading tiers for one predicted name vs. a gold OMIM disease:
    exact    the name resolves to the gold disease's Mondo concept
    related  it resolves to a Mondo ancestor/descendant within `max_hops` (e.g. the
             disease series "hereditary spastic paraplegia" for SPG7)
    wrong    it resolves to some other concept
    unmapped no confident match in the alias index

Build once (pickled), then reuse:
    norm = DiseaseNormalizer.build(mondo_obo, hpoa)
    norm.grade("Hereditary spastic paraplegia type 7", "OMIM:607259")
"""
import pickle
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7",
         "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12"}
STOP = {"disease", "disorder", "syndrome", "the", "of", "type", "form", "related", "associated"}
SYN = re.compile(r'^synonym: "(.*)" (EXACT|RELATED|NARROW|BROAD)')


def norm_text(s, drop_stop=False):
    s = s.replace("’", "'").replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)          # TuberousSclerosisComplex
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)      # CVIDSyndrome -> CVID Syndrome
    s = re.sub(r"'s\b|s'(?=\s|$)", lambda m: "" if m.group(0) == "'s" else "s", s)  # Fahr's, Fuchs'
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [ROMAN.get(t, t) for t in s.split()]
    if drop_stop:
        toks = [t for t in toks if t not in STOP]
    return " ".join(toks)


def parse_mondo(path):
    terms, cur = {}, None
    for line in Path(path).open():
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"names": [], "omim": [], "parents": []}
            continue
        if line.startswith("[") or cur is None:
            if line.startswith("["):
                cur = None
            continue
        if line.startswith("id: "):
            cur["id"] = line[4:]
            terms[cur["id"]] = cur
        elif line.startswith("name: "):
            cur["names"].append(line[6:])
        elif m := SYN.match(line):
            if m.group(2) in ("EXACT", "RELATED"):
                cur["names"].append(m.group(1))
        elif line.startswith("xref: OMIM:") and "equivalentTo" in line:
            cur["omim"].append(line.split()[1])
        elif line.startswith("is_a: MONDO:"):
            cur["parents"].append(line.split()[1])
        elif line.startswith("is_obsolete: true"):
            terms.pop(cur.get("id"), None)
            cur = None
    return {k: v for k, v in terms.items() if k.startswith("MONDO:")}


class DiseaseNormalizer:
    def __init__(self, alias, omim2mondo, parents, children, labels):
        self.alias = alias            # normalized alias -> set(mondo ids)
        self.omim2mondo = omim2mondo  # OMIM:xxxx -> MONDO:yyyy
        self.parents, self.children, self.labels = parents, children, labels
        self._keys = list(alias)

    @classmethod
    def build(cls, mondo_obo, hpoa=None):
        terms = parse_mondo(mondo_obo)
        alias, omim2mondo, parents, children, labels = defaultdict(set), {}, {}, defaultdict(list), {}
        for mid, t in terms.items():
            labels[mid] = t["names"][0] if t["names"] else mid
            parents[mid] = t["parents"]
            for p in t["parents"]:
                children[p].append(mid)
            for o in t["omim"]:
                omim2mondo[o] = mid
            for n in t["names"]:
                for key in {norm_text(n), norm_text(n, drop_stop=True)}:
                    if key:
                        alias[key].add(mid)
        if hpoa:  # OMIM titles from HPO annotations, as extra aliases
            for line in Path(hpoa).open():
                if line.startswith(("#", "database_id")):
                    continue
                db, name = line.split("\t")[:2]
                if not db.startswith("OMIM:"):
                    continue
                if db not in omim2mondo:
                    # new OMIM entries not yet in Mondo: use the OMIM ID as its own node
                    omim2mondo[db] = db
                    labels[db] = name
                    parents[db] = []
                for key in {norm_text(name), norm_text(name, drop_stop=True)}:
                    alias[key].add(omim2mondo[db])
        return cls(dict(alias), omim2mondo, parents, dict(children), labels)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as fh:
            obj.__dict__.update(pickle.load(fh))
        return obj

    def resolve(self, name, fuzzy_cutoff=92):
        """Return (set of Mondo IDs, method) for a free-text disease name."""
        variants = [name]
        variants += re.findall(r"\(([^)]+)\)", name)          # "(SPG7)" etc.
        variants.append(re.sub(r"\([^)]*\)", "", name))
        for v in variants:
            for key in (norm_text(v), norm_text(v, drop_stop=True)):
                if key in self.alias:
                    return self.alias[key], "exact_alias"
        key = norm_text(re.sub(r"\([^)]*\)", "", name), drop_stop=True)
        if len(key) < 4:
            return set(), "unmapped"
        hit = process.extractOne(key, self._keys, scorer=fuzz.token_sort_ratio, score_cutoff=fuzzy_cutoff)
        if hit:
            return self.alias[hit[0]], f"fuzzy:{hit[1]:.0f}"
        return set(), "unmapped"

    def _up_hops(self, src, dst, max_hops):
        """Directed is_a distance from src up to its ancestor dst, up to max_hops."""
        frontier, seen = {src}, {src}
        for d in range(1, max_hops + 1):
            frontier = {p for n in frontier for p in self.parents.get(n, [])} - seen
            if dst in frontier:
                return d
            seen |= frontier
        return None

    def n_descendants(self, node, cap=10_000):
        cache = self.__dict__.setdefault("_ndesc", {})
        if node not in cache:
            seen, stack = set(), [node]
            while stack and len(seen) < cap:
                for c in self.children.get(stack.pop(), []):
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            cache[node] = len(seen)
        return cache[node]

    def grade(self, name, gold_omim, max_hops=2, max_group_size=150):
        """'related' = the prediction is a specific ancestor group of the gold disease
        (<= max_hops up, with <= max_group_size descendants) or a subtype of it.
        Siblings that only share a broad parent (e.g. Turner vs. Jacobsen) are 'wrong'."""
        gold = self.omim2mondo.get(gold_omim)
        mids, method = self.resolve(name)
        if not mids:
            return {"tier": "unmapped", "method": method, "mondo": None}
        if gold in mids:
            return {"tier": "exact", "method": method, "mondo": gold}
        if gold:
            for m in mids:
                h = self._up_hops(gold, m, max_hops)  # m is an ancestor group of gold
                if h is not None and self.n_descendants(m) <= max_group_size:
                    return {"tier": "related", "method": method, "mondo": m, "hops": h, "rel": "ancestor"}
                h = self._up_hops(m, gold, max_hops)  # m is a subtype of gold
                if h is not None:
                    return {"tier": "related", "method": method, "mondo": m, "hops": h, "rel": "subtype"}
        return {"tier": "wrong", "method": method, "mondo": sorted(mids)[0]}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--mondo", required=True)
    ap.add_argument("--hpoa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", help="cases.jsonl to report gold-ID coverage")
    a = ap.parse_args()
    n = DiseaseNormalizer.build(a.mondo, a.hpoa)
    n.save(a.out)
    print(f"aliases={len(n.alias)} omim->mondo={len(n.omim2mondo)} -> {a.out}")
    if a.cases:
        golds = [json.loads(l) for l in open(a.cases)]
        ids = {g["disease_id"] for g in golds}
        missing = [i for i in ids if i not in n.omim2mondo]
        print(f"gold diseases mapped to Mondo: {len(ids) - len(missing)}/{len(ids)}; missing e.g. {missing[:5]}")
        # sanity: gold label should grade exact against itself
        self_ok = sum(n.grade(g["disease_label"], g["disease_id"])["tier"] == "exact" for g in golds)
        print(f"self-consistency (gold label -> exact): {self_ok}/{len(golds)}")
        for s, gid in [("Hereditary spastic paraplegia type 7 (SPG7)", "OMIM:607259"),
                       ("Hereditary spastic paraplegia", "OMIM:607259"),
                       ("Marfan syndrome", "OMIM:607259")]:
            print(s, "->", n.grade(s, gid))
