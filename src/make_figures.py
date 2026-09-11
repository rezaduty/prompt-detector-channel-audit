"""
Render the data charts from the result CSVs.

Concept diagrams live in paper/figures/*.html and are rendered separately by
paper/figures/render_html_figs.sh. Charts here carry no baked-in result text:
bars and axes are the data, and every interpretation stays in the caption.
"""
import json
import pathlib
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CFG = json.loads(pathlib.Path("config.json").read_text())
R = pathlib.Path("results")
F = pathlib.Path("paper/figures")
F.mkdir(parents=True, exist_ok=True)

SIGNALS = CFG["signals"]
INK = "#1f2937"
BLUE = "#245b81"
RED = "#c23a4f"
GREEN = "#2f7d68"
AMBER = "#d97706"
GREY = "#8a93a2"

# Figures are drawn at the exact width they are rendered at, so nothing is
# downscaled in the PDF and label text keeps its true point size.
COL = 3.45      # IEEEtran single column, inches
FULL = 7.10     # IEEEtran two-column span, inches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def save(fig, name):
    # pad keeps a margin between the outermost label ink and the figure edge,
    # so no tick label or legend touches the crop box.
    fig.tight_layout(pad=0.55)
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"  wrote {F/name}.pdf")


def wilson(k, n, z=1.96):
    """Wilson score interval. Group sizes here are small, so a rate without
    an interval would overstate what the chart can support."""
    if n == 0:
        return (0.0, 0.0)
    import math
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def rate_with_ci(mask):
    """Return (rate, lower error, upper error) for a boolean series."""
    k, n = int(mask.sum()), int(len(mask))
    r = k / n if n else 0.0
    lo, hi = wilson(k, n)
    return r, max(0.0, r - lo), max(0.0, hi - r)


def legend_above(ax, ncol, fontsize=7.5):
    """Put the legend in its own band above the axes.

    Placing it inside risks overlapping a bar or curve that reaches the top of
    the data range, which is exactly what happens on the rate charts where
    several groups sit at one.
    """
    ax.legend(frameon=False, ncol=ncol, fontsize=fontsize,
              loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12),
              mode="expand", borderaxespad=0, handlelength=1.4,
              columnspacing=1.4, handletextpad=0.5)


def auroc(pos, neg):
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return float("nan")
    w = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def load_baseline():
    b = pd.read_csv(R / "baseline.csv")
    b["is_prompt"] = b["is_prompt"].astype(str).str.lower().isin(["true", "1"])
    b["flag_list"] = b["direct_flags"].fillna("").apply(
        lambda s: [x for x in s.split("|") if x])
    b["sig_list"] = b["direct_signals"].fillna("").apply(
        lambda s: [x for x in s.split("|") if x])
    for mid in CFG["guard_models"]:
        t = mid.replace("-", "_")
        b[f"{t}_u"] = b[f"{t}_unsafe"].astype(str).str.lower().isin(["true", "1"])
    return b


def fig_signal_coverage(b):
    """How often each weighted signal fires on the probe traffic."""
    c = Counter()
    for lst in b.sig_list:
        c.update(lst)
    names = SIGNALS
    vals = [c[s] for s in names]
    colors = [BLUE if v > 0 else GREY for v in vals]
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    ax.barh(range(len(names)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("probes on which the signal fires")
    ax.set_xlim(0, max(vals) * 1.12 + 1)
    save(fig, "fig_signal_coverage")


def fig_score_distributions(b):
    """Salience score by group, ordered by whether the text is prompt shaped."""
    groups = [("imperative", RED), ("declarative", RED), ("indirect", RED),
              ("roleplay", AMBER), ("encoding", AMBER), ("benign", GREY),
              ("security doc", GREY), ("benign template", GREEN),
              ("legit prompt", GREEN)]
    key = {"security doc": "security_doc", "benign template": "benign_template",
           "legit prompt": "legit_prompt"}
    fig, ax = plt.subplots(figsize=(FULL, 2.7))
    for i, (label, color) in enumerate(groups):
        ph = key.get(label, label)
        vals = b.loc[b.phrasing == ph, "direct_score"].values
        # Deterministic jitter so equal scores do not stack into one dot and
        # hide how many probes the group holds.
        rng = __import__("numpy").random.default_rng(11)
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(i + jitter, vals, s=15, color=color, alpha=0.75,
                   edgecolors="none")
        ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=INK, linewidth=1.4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("salience score")
    ax.axhline(15, color=RED, linestyle="--", linewidth=1.2,
               label="band threshold")
    ax.set_ylim(-1.2, 18)
    ax.margins(x=0.03)
    legend_above(ax, 1)
    save(fig, "fig_score_distributions")


def fig_weight_search():
    """Where random weight vectors land on the separation axis."""
    sc = pd.read_csv(R / "weight_search_scores.csv")
    aus = []
    for _, g in sc.groupby("config_id"):
        aus.append(auroc(g[g.true_label.isin(["S5", "S6"])]["score"],
                         g[g.true_label == "SAFE"]["score"]))
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.hist(aus, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color=RED, linestyle="--", linewidth=1.2, label="chance")
    ax.set_xlabel("area under the curve, attack against benign")
    ax.set_ylabel("weight vectors")
    ax.set_xlim(0, 1)
    legend_above(ax, 1)
    save(fig, "fig_weight_search")


def fig_signal_sweep():
    """Separation as a function of each signal's weight."""
    sw = pd.read_csv(R / "signal_sweep.csv")
    fig, ax = plt.subplots(figsize=(FULL, 2.6))
    moving = []
    for s in SIGNALS:
        sub = sw[sw.signal == s]
        if sub.groupby("weight")["score"].sum().nunique() <= 1:
            continue
        moving.append(s)
        ws = sorted(sub.weight.unique())
        ys = []
        for w in ws:
            g = sub[sub.weight == w]
            ys.append(auroc(g[g.true_label.isin(["S5", "S6"])]["score"],
                            g[g.true_label == "SAFE"]["score"]))
        ax.plot(ws, ys, marker="o", markersize=3, linewidth=1.3,
                label=s.replace("_", " "))
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.2, label="chance")
    ax.set_xlabel("signal weight")
    ax.set_ylabel("area under the curve")
    ax.set_ylim(0, 1)
    ax.margins(x=0.02)
    legend_above(ax, len(moving) + 1)
    save(fig, "fig_signal_sweep")


def fig_phrasing(b):
    """Detection rate by attack phrasing, per detector."""
    phr = [("imperative", "imperative"), ("declarative", "declarative"),
           ("indirect", "indirect"), ("roleplay", "roleplay"),
           ("encoding", "encoding")]
    t0 = CFG["guard_models"][0].replace("-", "_")
    t1 = CFG["guard_models"][1].replace("-", "_")
    series = [
        ("risk flags", GREY, lambda m: rate_with_ci((b.flag_list.apply(len) > 0)[m])),
        ("zero-shot guard", BLUE, lambda m: rate_with_ci(b[f"{t0}_u"][m])),
        ("trained guard", GREEN, lambda m: rate_with_ci(b[f"{t1}_u"][m])),
    ]
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    width = 0.26
    for j, (label, color, fn) in enumerate(series):
        xs = [i + (j - 1) * width for i in range(len(phr))]
        stats = [fn(b.phrasing == ph) for _, ph in phr]
        ys = [t[0] for t in stats]
        err = [[t[1] for t in stats], [t[2] for t in stats]]
        ax.bar(xs, ys, width=width, color=color, label=label)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2, capthick=0.8)
    ax.set_xticks(range(len(phr)))
    ax.set_xticklabels([p[0] for p in phr])
    ax.set_ylabel("detection rate")
    ax.set_ylim(0, 1.18)
    ax.margins(x=0.04)
    legend_above(ax, 3)
    save(fig, "fig_phrasing")


def fig_false_positives(b):
    """False positive rate by benign group, per detector."""
    grp = [("benign", "benign"), ("legit prompt", "legit_prompt"),
           ("security doc", "security_doc"), ("benign template", "benign_template")]
    t0 = CFG["guard_models"][0].replace("-", "_")
    t1 = CFG["guard_models"][1].replace("-", "_")
    series = [
        ("risk flags", GREY, lambda m: rate_with_ci((b.flag_list.apply(len) > 0)[m])),
        ("zero-shot guard", BLUE, lambda m: rate_with_ci(b[f"{t0}_u"][m])),
        ("trained guard", GREEN, lambda m: rate_with_ci(b[f"{t1}_u"][m])),
    ]
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    width = 0.26
    for j, (label, color, fn) in enumerate(series):
        xs = [i + (j - 1) * width for i in range(len(grp))]
        stats = [fn(b.phrasing == ph) for _, ph in grp]
        ys = [t[0] for t in stats]
        err = [[t[1] for t in stats], [t[2] for t in stats]]
        ax.bar(xs, ys, width=width, color=color, label=label)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2, capthick=0.8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([g[0] for g in grp])
    ax.set_ylabel("false positive rate")
    ax.set_ylim(0, 1.18)
    ax.margins(x=0.06)
    legend_above(ax, 3)
    save(fig, "fig_false_positives")


def fig_defense(b):
    """Flag-channel coverage before and after the added rules."""
    dfn = pd.read_csv(R / "defense.csv")
    dfn["flag_list"] = dfn["flags"].fillna("").apply(
        lambda s: [x for x in s.split("|") if x])
    grp = [("imperative", "imperative"), ("declarative", "declarative"),
           ("indirect", "indirect"), ("benign", "benign"),
           ("legit prompt", "legit_prompt"), ("security doc", "security_doc"),
           ("benign template", "benign_template")]
    bstats = [rate_with_ci((b.flag_list.apply(len) > 0)[b.phrasing == ph])
              for _, ph in grp]
    astats = [rate_with_ci((dfn.flag_list.apply(len) > 0)[dfn.phrasing == ph])
              for _, ph in grp]
    before = [t[0] for t in bstats]
    after = [t[0] for t in astats]
    berr = [[t[1] for t in bstats], [t[2] for t in bstats]]
    aerr = [[t[1] for t in astats], [t[2] for t in astats]]
    fig, ax = plt.subplots(figsize=(FULL, 2.6))
    xs = range(len(grp))
    ax.bar([x - 0.19 for x in xs], before, width=0.38, color=GREY,
           label="shipped rules")
    ax.bar([x + 0.19 for x in xs], after, width=0.38, color=BLUE,
           label="with added rules")
    ax.errorbar([x - 0.19 for x in xs], before, yerr=berr, fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2, capthick=0.8)
    ax.errorbar([x + 0.19 for x in xs], after, yerr=aerr, fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2, capthick=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([g[0] for g in grp], rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("flag rate")
    ax.set_ylim(0, 1.18)
    ax.margins(x=0.03)
    legend_above(ax, 2)
    save(fig, "fig_defense")


def fig_guard_codes(b):
    """How many taxonomy codes each guard returns per verdict."""
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    labels = ["zero-shot guard", "trained guard"]
    colors = [BLUE, GREEN]
    width = 0.38
    maxc = 0
    for j, mid in enumerate(CFG["guard_models"]):
        t = mid.replace("-", "_")
        c = Counter(b[f"{t}_ncodes"].astype(int))
        maxc = max(maxc, max(c))
        xs = sorted(c)
        ax.bar([x + (j - 0.5) * width for x in xs], [c[x] for x in xs],
               width=width, color=colors[j], label=labels[j])
    ax.set_xticks(range(0, maxc + 1))
    ax.set_xlabel("taxonomy codes returned in one verdict")
    ax.set_ylabel("probes")
    legend_above(ax, 2)
    save(fig, "fig_guard_codes")


# ---------------------------------------------------------------- span figure

# Tag colours. Ordering matters: the first match wins when spans overlap.
TAG_STYLE = [
    ("injection_phrase", "#fdeef1", "#c23a4f", "injection phrase"),
    ("role_assertion",   "#fdf0e3", "#b4690e", "role assertion"),
    ("directive",        "#eaf1f8", "#245b81", "directive"),
    ("output_format",    "#eef3f1", "#2f7d68", "output format"),
    ("placeholder",      "#f2f3f5", "#6b7280", "placeholder"),
]
TAG_COLOR = {t: (bg, fg, lbl) for t, bg, fg, lbl in TAG_STYLE}

# The probes shown. The first three pairs put an attack directly above the
# benign text that matches on the same span, which is where the vocabulary
# rules cannot separate the two. The last two rows are benign text the
# sentence channel does leave alone.
SPAN_EXAMPLES = [
    ("s5_imp_01", "attack"),
    ("hn_d_01",   "benign"),
    ("s5_dec_02", "attack"),
    ("hn_p_04",   "benign"),
    ("s6_rp_01",  "attack"),
    ("hn_d_07",   "benign"),
    ("hn_p_03",   "benign"),
    ("hn_t_01",   "benign"),
]


def esc_html(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_spans(text, sentences):
    """Wrap every matched span in a coloured mark, leaving the rest plain."""
    marks = []
    for sent in sentences:
        for m in sent.get("matches", []):
            if m.get("tag") in TAG_COLOR:
                marks.append((m["start"], m["end"], m["tag"]))
    marks.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    out, cursor = [], 0
    for start, end, tag in marks:
        if start < cursor:            # overlapping span, first one wins
            continue
        out.append(esc_html(text[cursor:start]))
        bg, fg, _ = TAG_COLOR[tag]
        out.append(f'<mark style="background:{bg};color:{fg};'
                   f'border:1px solid {fg}">{esc_html(text[start:end])}</mark>')
        cursor = end
    out.append(esc_html(text[cursor:]))
    return "".join(out)


def fig_spans(b):
    """Worked examples: what the detector matches, span by span."""
    raw = {}
    for line in (R / "sentences_raw.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            raw[rec["probe_id"]] = rec

    base = b.set_index("id")
    t0 = CFG["guard_models"][0].replace("-", "_")
    t1 = CFG["guard_models"][1].replace("-", "_")

    rows = []
    for pid, side in SPAN_EXAMPLES:
        rec = raw[pid]
        row = base.loc[pid]
        sents = rec["sentences"]
        risky = any(s.get("risk") for s in sents)
        tags = []
        for s in sents:
            for t in (s.get("tags") or []):
                if t in TAG_COLOR and t not in tags:
                    tags.append(t)
        # A missing cell reads as NaN, which is truthy, so an "or" default
        # would mark every unflagged probe as flagged.
        raw_flags = row["direct_flags"]
        flags = [] if pd.isna(raw_flags) else [
            x for x in str(raw_flags).split("|") if x]
        flagged = bool(flags)
        raw_codes = row[f"{t1}_codes"]
        codes = "" if pd.isna(raw_codes) else str(raw_codes).strip()
        rows.append({
            "side": side,
            "group": row["phrasing"].replace("_", " "),
            "html": render_spans(rec["text"], sents),
            "sentence": "risk" if risky else "no risk",
            "flags": "flagged" if flagged else "not flagged",
            "guard": codes if codes else "safe",
            "truth": "attack" if side == "attack" else "benign",
        })

    legend = " ".join(
        f'<span class="lg"><span class="sw" style="background:{bg};'
        f'border-color:{fg}"></span>{lbl}</span>'
        for _, bg, fg, lbl in TAG_STYLE)

    body = []
    for r in rows:
        cls = "atk" if r["side"] == "attack" else "ben"
        body.append(f'''
  <div class="row {cls}">
    <div class="truth">{r["truth"]}<div class="grp">{r["group"]}</div></div>
    <div class="txt">{r["html"]}</div>
    <div class="verd">
      <span class="chip">sentence: {r["sentence"]}</span>
      <span class="chip">rules: {r["flags"]}</span>
      <span class="chip">guard: {r["guard"]}</span>
    </div>
  </div>''')

    # Row height measured from the rendered output: 100px per row, plus the
    # title, legend, and page padding.
    H = 120 + 101 * len(rows)
    html = f'''<style>
  *{{box-sizing:border-box;margin:0;padding:0;
     font-family:Arial,Helvetica,sans-serif}}
  @page{{size:1100px {H}px;margin:0}}
  body{{width:1100px;height:{H}px;background:#fff;color:#1f2733;
        padding:20px 24px}}
  .ttl{{font-size:18px;letter-spacing:2px;text-transform:uppercase;
        color:#8a93a2;font-weight:bold;margin-bottom:6px}}
  .leg{{margin-bottom:14px}}
  .lg{{font-size:14px;color:#6b7280;margin-right:18px;white-space:nowrap}}
  .sw{{display:inline-block;width:14px;height:14px;border-radius:3px;
       border:1px solid;vertical-align:-2px;margin-right:6px}}
  .row{{display:flex;align-items:center;gap:14px;padding:10px 12px;
        border-radius:10px;margin-bottom:8px}}
  .atk{{background:#fdf7f8;border:1px solid #f0d3d9}}
  .ben{{background:#f7faf9;border:1px solid #d3e4de}}
  .truth{{width:110px;flex:none;font-size:15px;font-weight:bold}}
  .grp{{font-size:12px;font-weight:normal;color:#6b7280;margin-top:2px}}
  .txt{{flex:1;font-size:15px;line-height:1.55}}
  mark{{border-radius:4px;padding:1px 3px}}
  .verd{{width:250px;flex:none;display:flex;flex-direction:column;gap:3px}}
  .chip{{font-size:12px;color:#374151;background:#fff;border:1px solid #d9dde4;
         border-radius:6px;padding:2px 7px;white-space:nowrap}}
</style>

<div class="ttl">What the detector matches, span by span</div>
<div class="leg">{legend}</div>
{"".join(body)}
'''
    (F / "fig_spans.html").write_text(html)
    print(f"  wrote {F}/fig_spans.html")


def main():
    b = load_baseline()
    fig_signal_coverage(b)
    fig_score_distributions(b)
    fig_weight_search()
    fig_signal_sweep()
    fig_phrasing(b)
    fig_false_positives(b)
    fig_defense(b)
    fig_guard_codes(b)
    fig_spans(b)




if __name__ == "__main__":
    main()
