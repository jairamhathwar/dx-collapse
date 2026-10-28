"""Drop vague entries from the extracted authors' differentials (extract_medcase_ddx.py output).

    python clean_authors_ddx.py authors_ddx.jsonl authors_ddx_clean.jsonl

An alternative is vague if every word of it, after normalization, is a category or qualifier word (VAGUE below or a
stop word), e.g. "infection", "viral infection", "malignancy", "metastatic disease", "drug-induced". Item numbers
are kept as in the judge's numbered list: item 1 is the final diagnosis, items 2.. the alternatives; the output adds
"keep" (the item numbers of the alternatives that remain) and "n_kept".
"""
import json
import sys

from disease_normalizer import STOP, norm_text

VAGUE = set("""infection infections infectious inflammatory inflammation bacterial bacteria viral virus viruses fungal
parasitic malignancy malignancies malignant neoplasm neoplasms neoplastic neoplasia tumor tumors tumour tumours cancer
cancers disease diseases disorder disorders process processes cause causes etiology etiologies other others drug drugs
medication medications toxic toxicity toxin toxins genetic congenital hereditary autoimmune cardiac
endocrine metabolic immunologic immunological immune allergic allergy embolic vascular degenerative idiopathic
metastatic metastasis metastases infiltrative systemic primary secondary acute chronic benign lesion lesions mass masses
condition conditions reaction reactions related induced origin unknown nonspecific non specific of the and or a an
due to with without""".split())


def vague(name):
    toks = norm_text(name).split()
    return not toks or all(t in VAGUE or t in STOP for t in toks)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_in = n_out = n_case = 0
    dropped = {}
    with open(dst, "w") as fh:
        for r in map(json.loads, open(src)):
            keep = [j + 1 for j, x in enumerate(r["ddx"]) if j >= 1 and not vague(x)]
            for j, x in enumerate(r["ddx"]):
                if j >= 1 and vague(x):
                    dropped[norm_text(x)] = dropped.get(norm_text(x), 0) + 1
            n_in += r["n_alternatives"]
            n_out += len(keep)
            n_case += bool(keep)
            fh.write(json.dumps({**r, "keep": keep, "n_kept": len(keep)}) + "\n")
    print(f"alternatives {n_in} -> {n_out} ({n_in - n_out} vague dropped); cases with at least one: {n_case}")
    print(sorted(dropped.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
