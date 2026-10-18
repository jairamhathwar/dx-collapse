"""Convert Phenopacket Store JSON files into flat case records + clinical vignettes.

Usage:
    python prep_phenopackets.py --raw $THESIS/data/phenopackets/raw \
        --out $THESIS/data/phenopackets/cases.jsonl [--pubdates]

Each output record:
    case_id, disease_id (OMIM/MONDO), disease_label, genes, present_hpo, excluded_hpo,
    sex, age_years, pmid, pub_year (if --pubdates), vignette
"""
import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

ISO_DUR = re.compile(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?")


def iso_to_years(s):
    m = ISO_DUR.fullmatch(s or "")
    if not m:
        return None
    y, mo, w, d = (int(x) if x else 0 for x in m.groups())
    return y + mo / 12 + w / 52 + d / 365


def age_of(pp):
    subj = pp.get("subject", {})
    for key in ("timeAtLastEncounter",):
        a = subj.get(key, {}).get("age", {}).get("iso8601duration")
        if a:
            return iso_to_years(a)
    for dz in pp.get("diseases", []):
        a = dz.get("onset", {}).get("age", {}).get("iso8601duration")
        if a:
            return iso_to_years(a)
    return None


def fmt_age(y):
    if y is None:
        return None
    if y < 1 / 12:
        return "newborn"
    if y < 1:
        return f"{round(y * 12)}-month-old"
    if y < 2:
        return f"{round(y * 12)}-month-old" if y * 12 % 12 else "1-year-old"
    return f"{int(y)}-year-old"


def render_vignette(rec):
    who = {"MALE": "male", "FEMALE": "female"}.get(rec["sex"], "patient")
    age = fmt_age(rec["age_years"])
    phrase = f"{age} {who}" if age else who
    article = "An" if re.match(r"(8|11-|18-|[aeiou])", phrase) else "A"
    head = f"{article} {phrase}"
    lines = [f"{head} is evaluated. Observed clinical features:"]
    lines += [f"- {t}" for t in rec["present_hpo_labels"]]
    if rec["excluded_hpo_labels"]:
        lines.append("Features explicitly noted as absent:")
        lines += [f"- {t}" for t in rec["excluded_hpo_labels"]]
    return "\n".join(lines)


def parse(path):
    pp = json.loads(Path(path).read_text())
    diseases = pp.get("diseases", [])
    if len(diseases) != 1:
        return None, "n_diseases!=1"
    dz = diseases[0]["term"]
    present = [f["type"] for f in pp.get("phenotypicFeatures", []) if not f.get("excluded")]
    excluded = [f["type"] for f in pp.get("phenotypicFeatures", []) if f.get("excluded")]
    if len(present) < 3:
        return None, "lt3_features"
    genes = sorted({
        gi.get("variantInterpretation", {}).get("variationDescriptor", {}).get("geneContext", {}).get("symbol")
        for it in pp.get("interpretations", [])
        for gi in it.get("diagnosis", {}).get("genomicInterpretations", [])
    } - {None})
    pmids = [r["id"].split(":")[1] for r in pp.get("metaData", {}).get("externalReferences", [])
             if r.get("id", "").startswith("PMID:")]
    rec = {
        "case_id": pp["id"],
        "disease_id": dz["id"],
        "disease_label": dz["label"],
        "genes": genes,
        "present_hpo": [t["id"] for t in present],
        "present_hpo_labels": [t["label"] for t in present],
        "excluded_hpo": [t["id"] for t in excluded],
        "excluded_hpo_labels": [t["label"] for t in excluded],
        "sex": pp.get("subject", {}).get("sex", "UNKNOWN_SEX"),
        "age_years": age_of(pp),
        "pmid": pmids[0] if pmids else None,
    }
    rec["vignette"] = render_vignette(rec)
    return rec, "ok"


def fetch_pub_years(pmids, batch=200):
    """PubMed esummary -> {pmid: year}. Unauthenticated limit is 3 req/s."""
    years = {}
    pmids = sorted(set(p for p in pmids if p))
    for i in range(0, len(pmids), batch):
        ids = ",".join(pmids[i:i + batch])
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={ids}"
        with urllib.request.urlopen(url, timeout=60) as r:
            res = json.load(r)["result"]
        for p in res.get("uids", []):
            m = re.match(r"(\d{4})", res[p].get("sortpubdate") or res[p].get("pubdate", ""))
            if m:
                years[p] = int(m.group(1))
        time.sleep(0.4)
    return years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pubdates", action="store_true")
    args = ap.parse_args()

    recs, reasons = [], Counter()
    for f in sorted(Path(args.raw).rglob("*.json")):
        rec, why = parse(f)
        reasons[why] += 1
        if rec:
            recs.append(rec)

    if args.pubdates:
        years = fetch_pub_years([r["pmid"] for r in recs])
        for r in recs:
            r["pub_year"] = years.get(r["pmid"])

    with open(args.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    n_dz = len({r["disease_id"] for r in recs})
    print(f"parsed: {dict(reasons)}")
    print(f"kept {len(recs)} cases, {n_dz} diseases -> {args.out}")
    print("disease id prefixes:", Counter(r["disease_id"].split(":")[0] for r in recs))
    if args.pubdates:
        yrs = Counter(r.get("pub_year") for r in recs)
        print("pub_year (>=2023):", {y: c for y, c in sorted(yrs.items(), key=lambda x: (x[0] is None, x[0])) if y is None or y >= 2023})


if __name__ == "__main__":
    main()
