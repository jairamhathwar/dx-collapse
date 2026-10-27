"""Pooled effects across model pairs and equivalence tests (pure Python; reads review2.py JSONs).

    python3 meta_tost.py --zs review2_zs_50k.json --fs review2_fs.json --third review2_third.json --rf review2_rf.json --out meta.json

Random-effects meta-analysis (DerSimonian-Laird) of each base -> post-trained difference across pairs, with standard
errors taken from the condition-clustered 95% intervals (width / 3.92), for three prompt settings: zero-shot T=0.7
(five pairs), worked examples (five pairs + Mistral) and worked examples from another model (six pairs). Reports the
pooled effect with a 95% interval, tau (between-pair SD), I^2, and a Qwen vs other-family subgroup test (Q_between,
1 df). Equivalence (TOST at the 5% level, i.e. the 90% interval inside the margin) uses fixed margins: 5 points for
pass@k and recall, 0.02 for TV, 0.03 for AUROC.
"""
import argparse
import json
import math

METRICS = ["pass32", "distinct", "rec5", "tv_list", "cond_tv", "sev_named", "auroc_agree32", "auroc_stated"]
MARGIN = {"pass32": 0.05, "pass1": 0.05, "rec5": 0.05, "tv_list": 0.02, "cond_tv": 0.02, "auroc_agree32": 0.03, "auroc_stated": 0.03}


def chi2_sf_1(q):
    return math.erfc(math.sqrt(max(q, 0) / 2))


def dl(ys, ses):
    w = [1 / s ** 2 for s in ses]
    yf = sum(wi * yi for wi, yi in zip(w, ys)) / sum(w)
    q = sum(wi * (yi - yf) ** 2 for wi, yi in zip(w, ys))
    df = len(ys) - 1
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    ws = [1 / (s ** 2 + tau2) for s in ses]
    y = sum(wi * yi for wi, yi in zip(ws, ys)) / sum(ws)
    se = math.sqrt(1 / sum(ws))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return {"est": y, "ci": [y - 1.96 * se, y + 1.96 * se], "se": se, "tau": math.sqrt(tau2), "I2": i2, "k": len(ys)}


def effects(d, keys, metric):
    out = []
    for k in keys:
        v = d["pairs"][k][metric]
        lo, hi = v["cluster_ci"]
        out.append((k, v["diff"], (hi - lo) / 3.92))
    return out


def tost(diff, se, margin):
    lo, hi = diff - 1.645 * se, diff + 1.645 * se
    return {"diff": diff, "ci90": [lo, hi], "margin": margin, "equivalent": lo > -margin and hi < margin,
            "noninferior": lo > -margin, "nonsuperior": hi < margin}


def main():
    ap = argparse.ArgumentParser()
    for x in ["zs", "fs", "third", "rf", "out"]:
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    Z, F, T, RF = (json.load(open(p)) for p in (a.zs, a.fs, a.third, a.rf))
    zs_keys = [k for k in Z["pairs"] if "_t1" not in k and "pair25_t1" not in k]
    fs_keys = list(F["pairs"])
    mis = [k for k in T["pairs"] if "mistral" in k]
    rf_keys = list(RF["pairs"])
    sets = {"zero-shot": [(Z, k) for k in zs_keys], "worked examples": [(F, k) for k in fs_keys] + [(T, k) for k in mis],
            "examples from another model": [(RF, k) for k in rf_keys]}
    report = {"meta": {}, "tost": {}}
    for name, items in sets.items():
        report["meta"][name] = {}
        for m in METRICS:
            rows = [(k, d["pairs"][k][m]["diff"], (d["pairs"][k][m]["cluster_ci"][1] - d["pairs"][k][m]["cluster_ci"][0]) / 3.92)
                    for d, k in items if m in d["pairs"][k]]
            if len(rows) < 2:
                continue
            allp = dl([r[1] for r in rows], [r[2] for r in rows])
            q = [r for r in rows if "qwen" in r[0]]
            o = [r for r in rows if "qwen" not in r[0]]
            res = {"pooled": allp, "pairs": rows}
            if len(q) >= 1 and len(o) >= 1:
                pq = dl([r[1] for r in q], [r[2] for r in q])
                po = dl([r[1] for r in o], [r[2] for r in o])
                qb = (pq["est"] - po["est"]) ** 2 / (pq["se"] ** 2 + po["se"] ** 2)
                res.update({"qwen": pq, "other": po, "Q_between": qb, "p_between": chi2_sf_1(qb)})
            report["meta"][name][m] = res
            print(f"{name:28s} {m:14s} pooled {allp['est']:+.3f} [{allp['ci'][0]:+.3f},{allp['ci'][1]:+.3f}] tau {allp['tau']:.3f} I2 {allp['I2']:.2f}"
                  + (f" | qwen {res['qwen']['est']:+.3f} other {res['other']['est']:+.3f} p_between {res['p_between']:.3f}" if "qwen" in res else ""))
    # equivalence for "no difference" claims
    claims = [("thinking vs base, pass@32", T, [k for k in T["pairs"] if "think" in k and "base" in k.split(">")[0]], "pass32"),
              ("32B zero-shot, pass@32", Z, [k for k in zs_keys if "32b" in k], "pass32"),
              ("OLMo 3 7B zero-shot, recall@5", Z, [k for k in zs_keys if "olmo3-base" in k], "rec5"),
              ("OLMo 3.1 32B zero-shot, agreement AUROC", Z, [k for k in zs_keys if "olmo31" in k], "auroc_agree32"),
              ("OLMo 3 7B zero-shot, agreement AUROC", Z, [k for k in zs_keys if "olmo3-base" in k], "auroc_agree32")]
    for label, d, keys, m in claims:
        for k in keys:
            v = d["pairs"][k][m]
            se = (v["cluster_ci"][1] - v["cluster_ci"][0]) / 3.92
            t = tost(v["diff"], se, MARGIN[m])
            report["tost"][f"{label} | {k}"] = t
            print(f"TOST {label:42s} {k[:40]:40s} {t['diff']:+.3f} ci90 [{t['ci90'][0]:+.3f},{t['ci90'][1]:+.3f}] margin {t['margin']} "
                  f"equiv={t['equivalent']} noninf={t['noninferior']}")
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
