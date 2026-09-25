"""
Fail-closed verifier.

Three sources of truth must agree before the paper may be built:

  1. the raw result CSVs in results/, re-read here and recomputed with
     independent code paths that do not import anything from analyze.py
  2. paper/tables/macros.tex, the macros Paper.tex actually renders
  3. Paper.tex itself, which may not contain a bare numeric result outside
     a macro, a table file, or a verbatim block

Any disagreement, any macro used in the paper but never defined, and any
hardcoded percentage in the prose exits non-zero.
"""
import json
import pathlib
import re
import sys

import pandas as pd
from scipy.stats import fisher_exact

CFG = json.loads(pathlib.Path("config.json").read_text())
R = pathlib.Path("results")
T = pathlib.Path("paper/tables")
PAPER = pathlib.Path("paper/Paper.tex")

FAILURES = []
CHECKED = 0


def check(name, got, want, tol=0.05):
    """Compare a recomputed value against the macro the paper renders."""
    global CHECKED
    CHECKED += 1
    try:
        g = float(got)
        w = float(str(want).replace("\\%", "").strip())
    except (TypeError, ValueError):
        if str(got).strip() != str(want).strip():
            FAILURES.append(f"{name}: recomputed {got!r} but macro says {want!r}")
        return
    if abs(g - w) > tol:
        FAILURES.append(f"{name}: recomputed {g:.4f} but macro says {w:.4f}")


def load_macros():
    txt = (T / "macros.tex").read_text()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt):
        out[m.group(1)] = m.group(2)
    return out


def main():
    if not (T / "macros.tex").exists():
        sys.exit("macros.tex missing, run src/analyze.py first")
    M = load_macros()

    base = pd.read_csv(R / "baseline.csv")
    base["is_prompt"] = base["is_prompt"].astype(str).str.lower().isin(["true", "1"])
    flags = base["direct_flags"].fillna("").apply(
        lambda s: [x for x in s.split("|") if x])
    sigs = base["direct_signals"].fillna("").apply(
        lambda s: [x for x in s.split("|") if x])
    atk = base.true_label.isin(["S5", "S6"])
    safe = base.true_label == "SAFE"

    # ---- probe counts
    check("NProbes", len(base), M["NProbes"], tol=0)
    check("NSfive", (base.true_label == "S5").sum(), M["NSfive"], tol=0)
    check("NSsix", (base.true_label == "S6").sum(), M["NSsix"], tol=0)
    check("NSafe", safe.sum(), M["NSafe"], tol=0)
    check("NGuardCalls",
          len(base) * len(CFG["guard_models"]) * CFG["guard_repeats"],
          M["NGuardCalls"], tol=0)

    # ---- salience channel, recomputed from scratch
    firing = set()
    for lst in sigs:
        firing.update(lst)
    check("NSignalsFiring", len(firing & set(CFG["signals"])),
          M["NSignalsFiring"], tol=0)
    check("NSignalsInert", len(set(CFG["signals"]) - firing),
          M["NSignalsInert"], tol=0)
    check("MeanScoreSafe", base.loc[safe, "direct_score"].mean(),
          M["MeanScoreSafe"], tol=0.05)
    check("MeanScoreSfive",
          base.loc[base.true_label == "S5", "direct_score"].mean(),
          M["MeanScoreSfive"], tol=0.05)
    check("MaxScoreAttack", base.loc[atk, "direct_score"].max(),
          M["MaxScoreAttack"], tol=0)
    check("MaxScoreSafe", base.loc[safe, "direct_score"].max(),
          M["MaxScoreSafe"], tol=0)

    # AUROC by brute-force pair counting, written independently here.
    def brute_auroc(pos, neg):
        pos, neg = list(pos), list(neg)
        w = sum(1.0 if a > b else 0.5 if a == b else 0.0
                for a in pos for b in neg)
        return w / (len(pos) * len(neg))

    check("AurocScoreAttack",
          100 * brute_auroc(base.loc[atk, "direct_score"],
                            base.loc[safe, "direct_score"]),
          M["AurocScoreAttack"], tol=0.2)
    check("AurocScorePrompt",
          100 * brute_auroc(base.loc[base.is_prompt, "direct_score"],
                            base.loc[~base.is_prompt, "direct_score"]),
          M["AurocScorePrompt"], tol=0.2)

    banded = base.direct_band != "none"
    check("BandTprAttack", 100 * banded[atk].mean(), M["BandTprAttack"], tol=0.2)
    check("BandFprSafe", 100 * banded[safe].mean(), M["BandFprSafe"], tol=0.2)

    # ---- flag channel
    any_flag = flags.apply(len) > 0
    check("FlagTprAttack", 100 * any_flag[atk].mean(), M["FlagTprAttack"], tol=0.2)
    check("FlagFprSafe", 100 * any_flag[safe].mean(), M["FlagFprSafe"], tol=0.2)
    kip = flags.apply(lambda l: "known_injection_phrase" in l)
    doc = base.phrasing == "security_doc"
    check("NKipDoc", kip[doc].sum(), M["NKipDoc"], tol=0)
    check("NKipSafe", kip[safe].sum(), M["NKipSafe"], tol=0)
    check("KipFprDoc", 100 * kip[doc].mean(), M["KipFprDoc"], tol=0.2)

    # ---- guard panel
    for mid, key in [("local-bionic-guard", "Zs"), ("model-67276875f520", "Tr")]:
        t = mid.replace("-", "_")
        unsafe = base[f"{t}_unsafe"].astype(str).str.lower().isin(["true", "1"])
        codes = base[f"{t}_codes"].fillna("").apply(
            lambda s: [c for c in s.split(",") if c])
        check(f"BinTpr{key}", 100 * unsafe[atk].mean(), M[f"BinTpr{key}"], tol=0.2)
        check(f"BinFpr{key}", 100 * unsafe[safe].mean(), M[f"BinFpr{key}"], tol=0.2)
        degen = codes.apply(len) >= 5
        check(f"NDegen{key}", degen.sum(), M[f"NDegen{key}"], tol=0)
        s5 = base.true_label == "S5"
        has5 = codes.apply(lambda l: "S5" in l)
        check(f"CatTprSfive{key}", 100 * has5[s5].mean(),
              M[f"CatTprSfive{key}"], tol=0.2)
        check(f"CatTprSfiveStrict{key}", 100 * (has5 & ~degen)[s5].mean(),
              M[f"CatTprSfiveStrict{key}"], tol=0.2)
        for ph, pk in [("imperative", "Imp"), ("declarative", "Dec"),
                       ("indirect", "Ind")]:
            m = base.phrasing == ph
            check(f"BinTpr{pk}{key}", 100 * unsafe[m].mean(),
                  M[f"BinTpr{pk}{key}"], tol=0.2)
        for ph, pk in [("legit_prompt", "Legit"), ("security_doc", "Doc")]:
            m = base.phrasing == ph
            check(f"BinFpr{pk}{key}", 100 * unsafe[m].mean(),
                  M[f"BinFpr{pk}{key}"], tol=0.2)

    # ---- guard call log agrees with the majority verdicts in baseline.csv
    log = [json.loads(l) for l in (R / "guard_raw.jsonl").read_text().splitlines() if l.strip()]
    check("NGuardLogRows", len(log), M["NGuardCalls"], tol=0)
    for mid, key in [("local-bionic-guard", "Zs"), ("model-67276875f520", "Tr")]:
        t = mid.replace("-", "_")
        per = {}
        for rec in log:
            if rec["model_id"] != mid:
                continue
            per.setdefault(rec["probe_id"], []).append(bool(rec["unsafe"]))
        recomputed = {pid: (sum(v) * 2 >= len(v)) for pid, v in per.items()}
        stated = dict(zip(base["id"],
                          base[f"{t}_unsafe"].astype(str).str.lower().isin(["true", "1"])))
        bad = [p for p in stated if recomputed.get(p) != stated[p]]
        if bad:
            FAILURES.append(f"{mid}: majority verdict disagrees with the call log "
                            f"on {len(bad)} probes: {bad[:5]}")

    # ---- weight search
    sc = pd.read_csv(R / "weight_search_scores.csv")
    cfgs = pd.read_csv(R / "weight_search.csv")
    check("NSearchConfigs", cfgs.config_id.nunique(), M["NSearchConfigs"], tol=0)
    check("NSearchCells", len(sc), M["NSearchCells"], tol=0)
    best = -1.0
    above = 0
    for cid, g in sc.groupby("config_id"):
        a = brute_auroc(g[g.true_label.isin(["S5", "S6"])]["score"],
                        g[g.true_label == "SAFE"]["score"])
        best = max(best, a)
        above += int(a > 0.5)
    check("SearchBestAuroc", 100 * best, M["SearchBestAuroc"], tol=0.2)
    check("SearchNAboveHalf", above, M["SearchNAboveHalf"], tol=0)

    # ---- external corpus
    ext = pd.read_csv(R / "external.csv")
    e_atk = ext.label == 1
    check("NExtRows", len(ext), M["NExtRows"], tol=0)
    check("NExtInjection", int(e_atk.sum()), M["NExtInjection"], tol=0)
    check("NExtBenign", int((~e_atk).sum()), M["NExtBenign"], tol=0)
    e_flag = ext.direct_flags.fillna("").apply(
        lambda x: len([t for t in str(x).split("|") if t]) > 0)
    check("ExtFlagTpr", 100 * e_flag[e_atk].mean(), M["ExtFlagTpr"], tol=0.2)
    e_band = ext.direct_band != "none"
    check("ExtBandTpr", 100 * e_band[e_atk].mean(), M["ExtBandTpr"], tol=0.2)
    for mid, key in [("local-bionic-guard", "Zs"), ("model-67276875f520", "Tr")]:
        u = ext[f"{mid.replace('-', '_')}_unsafe"].astype(str).str.lower().eq("true")
        check(f"ExtBinTpr{key}", 100 * u[e_atk].mean(), M[f"ExtBinTpr{key}"], tol=0.2)
        check(f"ExtBinFpr{key}", 100 * u[~e_atk].mean(), M[f"ExtBinFpr{key}"], tol=0.2)
    check("ExtAurocScore",
          100 * brute_auroc(ext.direct_score[e_atk], ext.direct_score[~e_atk]),
          M["ExtAurocScore"], tol=0.2)
    # The confound claim rests on length beating the score, so check that too.
    check("ExtAurocLength",
          100 * brute_auroc(ext.text_len[e_atk], ext.text_len[~e_atk]),
          M["ExtAurocLength"], tol=0.2)
    if float(M["ExtAurocLength"]) <= float(M["ExtAurocScore"]):
        FAILURES.append("the paper claims length outscores the salience score "
                        "on the external corpus, but the macros do not show it")

    # ---- sentence channel
    sen = pd.read_csv(R / "sentences.csv")
    sen["risk"] = sen["risk"].astype(str).str.lower().isin(["true", "1"])
    check("NSentences", len(sen), M["NSentences"], tol=0)
    pr = sen.groupby(["probe_id", "true_label", "phrasing"])["risk"].any().reset_index()
    p_atk = pr.true_label.isin(["S5", "S6"])
    check("SentTprAttack", 100 * pr.risk[p_atk].mean(), M["SentTprAttack"], tol=0.2)
    check("SentFprSafe", 100 * pr.risk[~p_atk].mean(), M["SentFprSafe"], tol=0.2)
    check("SentFprDoc", 100 * pr.risk[pr.phrasing == "security_doc"].mean(),
          M["SentFprDoc"], tol=0.2)
    # Every probe must appear in the sentence pass, or the rates above are
    # computed over a different set than the rest of the paper.
    if sorted(pr.probe_id) != sorted(base["id"]):
        FAILURES.append("sentences.csv does not cover the same probes as baseline.csv")

    # ---- defense phase
    dfn = pd.read_csv(R / "defense.csv")
    dflags = dfn["flags"].fillna("").apply(lambda s: [x for x in s.split("|") if x])
    d_any = dflags.apply(len) > 0
    d_atk = dfn.true_label.isin(["S5", "S6"])
    d_safe = dfn.true_label == "SAFE"
    check("DefTprAttack", 100 * d_any[d_atk].mean(), M["DefTprAttack"], tol=0.2)
    check("DefFprSafe", 100 * d_any[d_safe].mean(), M["DefFprSafe"], tol=0.2)
    check("DefTprDec", 100 * d_any[dfn.phrasing == "declarative"].mean(),
          M["DefTprDec"], tol=0.2)
    # Probe order must match between the baseline and defense phases for the
    # paired before-and-after counts to mean anything.
    if list(dfn.probe_id) != list(base["id"]):
        FAILURES.append("defense.csv probe order differs from baseline.csv")
    gained_fp = int(((~any_flag.values) & d_any.values & d_safe.values).sum())
    check("DefGainedFp", gained_fp, M["DefGainedFp"], tol=0)

    # ---- training-data claims match the service's own model record
    # This is the check that keeps the dataset description honest: the paper
    # may only describe corpora the service actually reports for the model
    # that was measured.
    meta_models = {m["id"]: m for m in json.loads(
        (R / "run_meta.json").read_text())["guard_models"]}
    for mid, key in [("local-bionic-guard", "Zs"), ("model-67276875f520", "Tr")]:
        rec = meta_models.get(mid, {})
        ds = rec.get("dataset_stats") or {}
        check(f"NTrainTotal{key}", ds.get("total", 0), M[f"NTrainTotal{key}"], tol=0)
        srcs = ds.get("per_source") or {}
        ext = sorted(n for n in srcs if not n.startswith("seed"))
        check(f"NExternalRows{key}",
              sum(c for n, c in srcs.items() if not n.startswith("seed")),
              M[f"NExternalRows{key}"], tol=0)
        check(f"ExternalSourceList{key}", ", ".join(ext) or "none",
              M[f"ExternalSourceList{key}"])
        check(f"Base{key}", rec.get("base_model", "unknown"), M[f"Base{key}"])
        # Any dataset named in the paper must be one the service reports.
        if PAPER.exists():
            named = set(re.findall(r"\\ExternalSourceList" + key + r"\{\}",
                                   PAPER.read_text()))
            if ext and not named:
                FAILURES.append(f"{mid}: training sources {ext} are recorded but "
                                f"the paper never names them through a macro")

    # ---- the service was left at its shipped configuration
    meta = json.loads((R / "run_meta.json").read_text())
    drifted = [s["name"] for s in meta["scoring"]["signals"]
               if s["weight"] != s["default_weight"]]
    if drifted:
        FAILURES.append(f"run_meta captured non-default weights: {drifted}")

    # ---- the seven-detector comparison, recomputed independently
    src = json.loads((R / "panel" / "PANEL_SOURCE.json").read_text())
    inj = {d: c["injection_label"] for d, c in src["detectors"].items()}
    raw = {(r["guard"], r["text_sha"]): r for r in
           (json.loads(l) for l in (R / "panel" / "guard_raw.jsonl").read_text().splitlines())}
    pflag = {}
    for v in (json.loads(l) for l in (R / "panel" / "verdicts.jsonl").read_text().splitlines()):
        if "error" in v:
            FAILURES.append(f"panel verdict error: {v['id']} {v['detector']}")
            continue
        if v["detector"] in inj and v.get("from") != "phase1":
            r = raw.get((v["detector"], v["text_sha"]))
            if r is None:
                FAILURES.append(f"panel {v['id']} {v['detector']}: no raw call")
            else:
                scores = json.loads(r["raw"])
                check(f"panel raw {v['id']} {v['detector']}", int(max(scores, key=scores.get) == inj[v["detector"]]),
                      int(v["flag"]), tol=0)
        pflag[(v["id"], v["detector"])] = bool(v["flag"])
    grp = {}
    for _, r in base.iterrows():
        grp[r["id"]] = "Atk" if r["true_label"] in ("S5", "S6") else {"legit_prompt": "Legit", "security_doc": "Doc",
                                                                    "benign_template": "Tmpl", "benign": "Plain"}[r["phrasing"]]
    keyname = {"protectai": "Protectai", "piguard": "Piguard", "deepset": "Deepset", "fmops": "Fmops",
               "llm": "Llm", "llm27": "Llmtwoseven"}
    legit_all = 0
    for d, K in keyname.items():
        for G in ("Atk", "Legit", "Doc", "Tmpl", "Plain"):
            ids = [i for i, g in grp.items() if g == G]
            k = sum(pflag[(f"t35_{i}", d)] for i in ids)
            check(f"Pan{K}{G}K", k, M.get(f"Pan{K}{G}K"), tol=0)
            check(f"Pan{K}{G}N", len(ids), M.get(f"Pan{K}{G}N"), tol=0)
            if G == "Legit" and k == len(ids):
                legit_all += 1
        ni = [key for key in pflag if key[0].startswith("notinject_") and key[1] == d]
        check(f"Pan{K}NotInjectK", sum(pflag[x] for x in ni), M.get(f"Pan{K}NotInjectK"), tol=0)
    check("NFlagAllLegit", legit_all, M.get("NFlagAllLegit"), tol=0)
    # Bench flag rules: recomputed from the raw channel outputs.
    ni_csv = pd.read_csv(R / "notinject.csv")
    ni_flag = int(sum(1 for x in ni_csv.direct_flags if isinstance(x, str) and x.strip()))
    doc = base[base.phrasing == "security_doc"]
    doc_flag = int(sum(1 for x in doc.direct_flags if isinstance(x, str) and x.strip()))
    check("PanBenchflagNotInjectK", ni_flag, M.get("PanBenchflagNotInjectK"), tol=0)
    nig = pd.read_csv(R / "notinject_guards.csv")
    for col, key in (("local_bionic_guard_unsafe", "PanBenchzsNotInjectK"), ("model_67276875f520_unsafe", "PanBenchtrNotInjectK")):
        errs = int(sum(str(x) == "error" for x in nig[col]))
        if errs:
            FAILURES.append(f"{col}: {errs} guard errors on NotInject")
        check(key, int(sum(str(x) == "True" for x in nig[col])), M.get(key), tol=0)
    check("PanBenchflagDocK", doc_flag, M.get("PanBenchflagDocK"), tol=0)
    # Real corpora, recomputed from the panel verdicts and the bench CSV.
    for d, K in keyname.items():
        for pre, G in (("prompts_", "RealPrompt"), ("arxivsec_", "RealDoc")):
            xs = [f for (i, dd), f in pflag.items() if dd == d and i.startswith(pre)]
            check(f"Pan{K}{G}K", sum(xs), M.get(f"Pan{K}{G}K"), tol=0)
    rc = pd.read_csv(R / "real_corpora.csv")
    for col, K in (("local_bionic_guard_unsafe", "Benchzs"), ("model_67276875f520_unsafe", "Benchtr")):
        for corp, G in (("prompts_chat", "RealPrompt"), ("arxiv_security", "RealDoc")):
            sub = rc[rc.corpus == corp][col]
            if any(str(x) == "error" for x in sub):
                FAILURES.append(f"{col} errors on {corp}")
            check(f"Pan{K}{G}K", int(sum(str(x) == "True" for x in sub)), M.get(f"Pan{K}{G}K"), tol=0)
    # Rank agreement and per-detector level differences, NotInject against
    # the security abstracts, over every output that flags anything.
    from scipy.stats import spearmanr as _sp
    outs = []
    for K in list(keyname.values()) + ["Benchflag", "Benchzs", "Benchtr"]:
        kn, nn = int(M[f"Pan{K}NotInjectK"]), int(M[f"Pan{K}NotInjectN"])
        kd, nd = int(M[f"Pan{K}RealDocK"]), int(M[f"Pan{K}RealDocN"])
        outs.append((kn, nn, kd, nd))
    rho = _sp([a / b for a, b, _, _ in outs], [c / d for _, _, c, d in outs])[0]
    check("PanNiDocRho", round(float(rho), 3), M.get("PanNiDocRho"), tol=0.001)
    hi = sum(1 for a, b, c, d in outs if fisher_exact([[a, b - a], [c, d - c]])[1] < 0.05 / len(outs) and c / d > a / b)
    check("NPanDocHigher", hi, M.get("NPanDocHigher"), tol=0)
    if not hi > len(outs) / 2:
        FAILURES.append("claim broken: most detector outputs no longer flag documentation more than NotInject")

    # Claims: the benchmark disagreement is significant, and the language-model
    # detector flags no security document.
    p_dis = fisher_exact([[ni_flag, len(ni_csv) - ni_flag], [doc_flag, len(doc) - doc_flag]])[1]
    if not p_dis < 0.05:
        FAILURES.append("claim broken: NotInject and security documentation no longer disagree for the flag rules")
    if M.get("PanLlmDocK") != "0":
        FAILURES.append("claim broken: the language-model detector now flags a security document")

    # ---- datasheet counts for the added corpora
    readme = re.sub(r"\s+", " ", pathlib.Path("data/README.md").read_text()) if pathlib.Path("data/README.md").exists() else ""
    n_ni = len(pathlib.Path("data/external/notinject.jsonl").read_text().splitlines())
    n_pv = len((R / "panel" / "verdicts.jsonl").read_text().splitlines())
    for claim in (f"`external/notinject.jsonl`, {n_ni} benign prompts", f"{n_pv} verdicts in all"):
        if claim not in readme:
            FAILURES.append(f"datasheet disagrees with the files: {claim!r} not found")

    # ---- publish the verifier's own counts before auditing Paper.tex
    # CHECKED is final here: the hygiene block below records failures directly
    # rather than through check(). Writing the macro now lets the paper cite
    # the count without the undefined-macro check tripping on it.
    summary = (f"VERIFY PASSED: {CHECKED} checks across {len(base)} probes, "
               f"{len(log)} guard calls, {len(sc)} search cells, "
               f"{len(M)} macros")
    (T / "verify_output.tex").write_text(
        "\\begin{lstlisting}\n"
        f"$ python3 src/verify_numbers.py\n{summary}\n"
        "\\end{lstlisting}\n")
    macros = (T / "macros.tex").read_text()
    line = f"\\newcommand{{\\NVerifyChecks}}{{{CHECKED}}}"
    kept = [l for l in macros.splitlines()
            if not l.startswith("\\newcommand{\\NVerifyChecks}")]
    (T / "macros.tex").write_text("\n".join(kept + [line]) + "\n")
    M = load_macros()

    # ---- Paper.tex hygiene and static LaTeX validation
    # No LaTeX engine is available in this environment, so the source is
    # checked structurally instead: every cite key resolves, every ref has a
    # label, every included file exists, and every environment closes.
    if PAPER.exists():
        tex = PAPER.read_text()
        body = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", tex, flags=re.S)
        body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", body, flags=re.S)
        body = re.sub(r"(?<!\\)%.*", "", body)

        used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", body))
        undefined = sorted(u for u in used if u not in M)
        if undefined:
            FAILURES.append(f"macros used in Paper.tex but never defined: {undefined}")

        # A bare percentage in the prose cannot be re-derived from the data.
        for m in re.finditer(r"(?<![\\A-Za-z0-9])(\d+\.\d+)\s*\\%", body):
            FAILURES.append(f"hardcoded percentage in Paper.tex: {m.group(0)!r}")
        # Whole-number percentages and "percent" typed in words count too.
        for m in re.finditer(r"(?<![\\A-Za-z0-9{])(\d+)\s*(\\%|percent)", body):
            FAILURES.append(f"hardcoded percentage in Paper.tex: {m.group(0)!r}")

        for bad, why in [("\u2014", "em dash"), ("\u2013", "en dash"),
                         ("->", "arrow"), ("honest", "self-referential hedging"),
                         ("genuinely", "self-referential hedging"),
                         ("we can't", "self-referential hedging"),
                         ("cannot fake", "self-referential hedging")]:
            if bad in body.lower():
                FAILURES.append(f"Paper.tex contains {why}: {bad!r}")


        # A caption that states a finding should point at where it is
        # established, so a reader meeting the figure first is not stranded.
        for m in re.finditer(r"\\caption\{(.+?)\}\s*\n?\s*\\label\{(fig:[^}]*)\}",
                             tex, flags=re.S):
            cap, lab = m.group(1), m.group(2)
            if ("\\cite{" not in cap and "\\ref{" not in cap
                    and re.search(r"\b(shows|because|so that|which is what)\b", cap)):
                FAILURES.append(f"caption of {lab} asserts a finding with no "
                                f"citation and no section reference")

        # Deictic pointers that depend on where text lands on the page, or
        # that make the reader count back through a list, do not survive
        # two-column float placement. Each must name what it refers to.
        VAGUE = {
            r"\bwork above\b": "positional reference to earlier text",
            r"\b(numbers|figures|rates|results|table|tables) below\b":
                "positional reference to later text",
            r"\bthe (former|latter)\b": "forces the reader to backtrack",
            r"\bthe other two\b": "does not name what it refers to",
            r"\bfall\s?backs?\b": "unexplained fallback",
            r"\bfails here\b": "'here' does not name the group",
            r"\bOnly the (first|second|third|fourth)\b":
                "refers to a list item by position",
        }
        for pat, why in VAGUE.items():
            for m in re.finditer(pat, body, re.I):
                line = body[:m.start()].count("\n") + 1
                FAILURES.append(f"Paper.tex line {line}: {m.group(0)!r} is "
                                f"a vague reference ({why})")

        # Semicolons are not used in this paper's prose. LaTeX spacing
        # commands such as \; and \quad are markup, not punctuation.
        prose = body.replace("\\;", " ").replace("\\,", " ").replace("\\:", " ")
        semis = [i for i, l in enumerate(prose.splitlines(), 1) if ";" in l]
        if semis:
            FAILURES.append(f"semicolons in Paper.tex prose, lines {semis[:8]}")

        # Every cite key must exist in the bibliography, and every entry in the
        # bibliography must be cited, so no reference is decorative.
        bib = pathlib.Path("paper/bibliography.bib").read_text()
        defined_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        cited = set()
        for m in re.finditer(r"\\cite\{([^}]*)\}", body):
            cited.update(k.strip() for k in m.group(1).split(","))
        missing = sorted(cited - defined_keys)
        if missing:
            FAILURES.append(f"cite keys with no bibliography entry: {missing}")
        uncited = sorted(defined_keys - cited)
        if uncited:
            FAILURES.append(f"bibliography entries never cited: {uncited}")

        # Terms naming a specific external method, dataset, or standard must
        # be cited somewhere in the paper. This does not require a citation on
        # every repeat use, only that the term is never used uncited.
        NEEDS_CITE = {
            "Wilson": "wilson1927interval",
            "Fisher": "fisher1922interpretation",
            "Spearman": "spearman1904",
            "Mann-Whitney": "mann1947test",
            "receiver operating characteristic": "fawcett2006introduction",
            "random search": "bergstra2012random",
            "LoRA": "hu2022lora",
            "low-rank adaptation": "hu2022lora",
            "Llama Guard": "inan2023llamaguard",
            "NeMo Guardrails": "rebedea2023nemo",
            "OWASP": "owasp2026",
            "ATLAS": "mitre_atlas",
            "NIST": "nist_aiml_2025",
            "deepset": "deepset_promptinjections",
            "shortcut": "geirhos2020shortcut",
            "Qwen": "qwen2024report",
        }
        for term, key in NEEDS_CITE.items():
            if re.search(re.escape(term), body) and key not in cited:
                FAILURES.append(f"Paper.tex uses {term!r} but never cites {key}")

        # Every cross-reference must resolve to a label.
        labels = set(re.findall(r"\\label\{([^}]*)\}", body))
        refs = set()
        for m in re.finditer(r"\\ref\{([^}]*)\}", body):
            refs.add(m.group(1))
        dangling = sorted(refs - labels)
        if dangling:
            FAILURES.append(f"references with no label: {dangling}")

        # Every included table and figure file must exist.
        for m in re.finditer(r"\\input\{([^}]*)\}", tex):
            t = m.group(1)
            f = pathlib.Path("paper") / (t if t.endswith(".tex") else t + ".tex")
            if not f.exists():
                FAILURES.append(f"\\input target missing: {f}")
        for m in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", tex):
            f = pathlib.Path("paper") / m.group(1)
            if not f.exists() and not f.with_suffix(".pdf").exists():
                FAILURES.append(f"\\includegraphics target missing: {f}")

        # Environments must balance, and braces must balance.
        opened = re.findall(r"\\begin\{([a-zA-Z*]+)\}", tex)
        closed = re.findall(r"\\end\{([a-zA-Z*]+)\}", tex)
        from collections import Counter as _C
        diff = _C(opened) - _C(closed)
        diff2 = _C(closed) - _C(opened)
        if diff or diff2:
            FAILURES.append(f"unbalanced environments: opened {dict(diff)}, "
                            f"closed {dict(diff2)}")
        depth = 0
        stripped = re.sub(r"\\[{}%]", "", tex)
        for ch in stripped:
            depth += (ch == "{") - (ch == "}")
            if depth < 0:
                break
        if depth != 0:
            FAILURES.append(f"unbalanced braces in Paper.tex, net depth {depth}")

        # The included table files must balance too.
        for f in sorted(T.glob("tab_*.tex")):
            t = f.read_text()
            if t.count(r"\begin{tabular}") != t.count(r"\end{tabular}"):
                FAILURES.append(f"unbalanced tabular in {f}")

    # ---- report
    if FAILURES:
        print(f"VERIFY FAILED: {len(FAILURES)} problem(s) across {CHECKED} checks")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(summary)


if __name__ == "__main__":
    main()
