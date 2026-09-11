"""
Run every measurement phase against the live local detection service.

Phases
  P1 baseline      default configuration, both detection channels plus the
                   guard-model panel, R repeats per probe per model
  P2 signal sweep  one signal weight at a time across the full 0..100 range
  P3 weight search random search over the joint 10-dimensional weight space
  P4 threshold     noise-floor sweep at default weights
  P5 defense       custom risk flags targeting declarative and indirect
                   phrasing, added, measured, then removed

Server state (signal weights, thresholds, custom flags) is mutated by P2,
P3, P4 and P5. Every phase restores defaults in a finally block, and main()
restores again on exit, so an interrupted run does not leave the service
mis-configured.

Outputs (results/):
  baseline.csv            one row per probe
  guard_raw.jsonl         one row per guard call, kept for manual audit
  signal_sweep.csv        probe score per signal per weight
  weight_search.csv       one row per sampled weight vector
  weight_search_scores.csv  probe score per sampled weight vector
  threshold_sweep.csv     probe band per noise-floor value
  defense.csv             one row per probe under the custom flag set
  run_meta.json           service configuration captured at run time
"""
import csv
import json
import os
import pathlib
import random
import sys
import time

import requests

CFG = json.loads(pathlib.Path("config.json").read_text())
BASE = CFG["base_url"]
PROBE_FILE = pathlib.Path(CFG["probe_file"])
RESULTS = pathlib.Path("results")
GUARD_MODELS = CFG["guard_models"]
REPEATS = CFG["guard_repeats"]
SIGNALS = CFG["signals"]
SWEEP_WEIGHTS = CFG["sweep_weights"]
SEARCH_N = CFG["weight_search_n"]
SEARCH_SEED = CFG["weight_search_seed"]
NOISE_FLOORS = CFG["noise_floor_sweep"]
CUSTOM_FLAGS = CFG["custom_risk_flags"]
CARRIERS = CFG["carriers"]

S = requests.Session()


# ---------------------------------------------------------------- API calls

def policy_test(text, filename="pasted.txt"):
    r = S.post(f"{BASE}/api/prompt-scans/policy-test",
               json={"text": text, "filename": filename}, timeout=30)
    r.raise_for_status()
    return r.json()


def guard_classify(text, model_id, explain=True):
    r = S.post(f"{BASE}/api/prompt-scans/guard-classify",
               json={"text": text, "model_ids": [model_id],
                     "combine": "or", "explain": explain}, timeout=120)
    r.raise_for_status()
    return r.json()


def set_weight(name, value):
    r = S.put(f"{BASE}/api/prompt-scans/scoring/weight/{name}",
              json={"weight": int(value)}, timeout=30)
    r.raise_for_status()


def reset_weight(name):
    r = S.delete(f"{BASE}/api/prompt-scans/scoring/weight/{name}", timeout=30)
    if r.status_code not in (200, 204, 404):
        r.raise_for_status()


def set_threshold(key, value):
    r = S.put(f"{BASE}/api/prompt-scans/scoring/threshold/{key}",
              json={"value": int(value)}, timeout=30)
    r.raise_for_status()


def reset_threshold(key):
    r = S.delete(f"{BASE}/api/prompt-scans/scoring/threshold/{key}", timeout=30)
    if r.status_code not in (200, 204, 404):
        r.raise_for_status()


def add_custom_flag(spec):
    r = S.post(f"{BASE}/api/prompt-scans/risk-flags/custom", json=spec, timeout=30)
    if r.status_code not in (200, 201):
        r.raise_for_status()


def delete_custom_flag(name):
    r = S.delete(f"{BASE}/api/prompt-scans/risk-flags/custom/{name}", timeout=30)
    if r.status_code not in (200, 204, 404):
        r.raise_for_status()


def get_scoring():
    r = S.get(f"{BASE}/api/prompt-scans/scoring", timeout=30)
    r.raise_for_status()
    return r.json()


def restore_defaults():
    """Return the service to its shipped configuration."""
    for name in SIGNALS:
        reset_weight(name)
    for key in ["noise_floor", "band_high", "band_medium", "band_low",
                "linguistic_per_hit", "multiline_min_chars",
                "multiline_min_lines", "max_score"]:
        reset_threshold(key)
    for spec in CUSTOM_FLAGS:
        delete_custom_flag(spec["name"])


# ---------------------------------------------------------------- helpers

def tag(mid):
    return mid.replace("-", "_")


def load_probes():
    return [json.loads(l) for l in PROBE_FILE.read_text().splitlines() if l.strip()]


def channels(resp):
    """Split a policy-test response into its two detection channels.

    direct  whole-text scoring, the path a chat input rail exercises
    node    prompt candidates recovered by the structural detector
    """
    direct = resp.get("direct") or {}
    nodes = resp.get("nodes") or []
    top = max(nodes, key=lambda n: n.get("score", 0)) if nodes else {}
    return direct, nodes, top


def redact_paths(obj):
    """Replace absolute local paths in captured metadata before it is written.

    The service reports on-disk locations for model artifacts. Those say
    nothing about the measurement and would publish the local directory layout,
    so they are reduced to the trailing component.
    """
    home = os.path.expanduser("~")
    if isinstance(obj, dict):
        return {k: redact_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_paths(v) for v in obj]
    if isinstance(obj, str) and (obj.startswith(home) or obj.startswith("/Users")
                                 or obj.startswith("/home")):
        return "<redacted>/" + os.path.basename(obj.rstrip("/"))
    return obj


def assert_defaults_match():
    """Fail closed if the shipped configuration is not what config.json records."""
    sc = get_scoring()
    live = {s["name"]: s["default_weight"] for s in sc["signals"]}
    for name in SIGNALS:
        if name not in live:
            raise SystemExit(f"signal {name} missing from live service")
    cur = {s["name"]: s["weight"] for s in sc["signals"]}
    bad = {k: (cur[k], live[k]) for k in live if cur[k] != live[k]}
    if bad:
        raise SystemExit(f"service not at default weights after reset: {bad}")
    return sc


# ---------------------------------------------------------------- phases

def phase1_baseline(probes):
    print(f"P1 baseline: {len(probes)} probes, {len(GUARD_MODELS)} guard models, "
          f"{REPEATS} repeats")
    rows = []
    guard_log = open(RESULTS / "guard_raw.jsonl", "w")
    for i, p in enumerate(probes, 1):
        resp = policy_test(p["text"])
        direct, nodes, top = channels(resp)

        row = {
            "id": p["id"],
            "true_label": p["true_label"],
            "phrasing": p["phrasing"],
            "is_prompt": p["is_prompt"],
            "direct_score": direct.get("score", -1),
            "direct_band": direct.get("band", "error"),
            "direct_signals": "|".join(s["name"] for s in direct.get("signals", [])),
            "direct_flags": "|".join(direct.get("risk_flags", [])),
            "n_nodes": len(nodes),
            "node_score": top.get("score", 0) if top else 0,
            "node_band": top.get("band", "none") if top else "none",
            "node_signals": "|".join(s["name"] for s in top.get("signals", [])) if top else "",
        }

        for mid in GUARD_MODELS:
            verdicts = []
            for rep in range(REPEATS):
                g = guard_classify(p["text"], mid, explain=True)
                det = (g.get("details") or [{}])[0]
                rec = {
                    "probe_id": p["id"], "true_label": p["true_label"],
                    "phrasing": p["phrasing"], "model_id": mid, "repeat": rep,
                    "unsafe": g.get("unsafe"), "codes": g.get("codes", []),
                    "raw": det.get("raw", ""), "reason": det.get("reason", ""),
                    "model_name": det.get("name", ""), "note": g.get("note", ""),
                    "text": p["text"],
                }
                guard_log.write(json.dumps(rec) + "\n")
                verdicts.append(rec)

            # Majority verdict over repeats; ties resolve to unsafe.
            n_unsafe = sum(1 for v in verdicts if v["unsafe"])
            maj_unsafe = n_unsafe * 2 >= REPEATS
            code_sets = [tuple(sorted(v["codes"])) for v in verdicts]
            stable = len(set(code_sets)) == 1
            # Representative run for code-level analysis: first repeat whose
            # verdict matches the majority.
            rep_v = next((v for v in verdicts if bool(v["unsafe"]) == maj_unsafe), verdicts[0])
            tag = mid.replace("-", "_")
            row[f"{tag}_unsafe"] = maj_unsafe
            row[f"{tag}_codes"] = ",".join(rep_v["codes"])
            row[f"{tag}_ncodes"] = len(rep_v["codes"])
            row[f"{tag}_stable"] = stable
            row[f"{tag}_reason"] = rep_v["reason"].replace("\n", " ").strip()

        rows.append(row)
        if i % 10 == 0:
            print(f"   {i}/{len(probes)}")
    guard_log.close()

    with open(RESULTS / "baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"   wrote {RESULTS/'baseline.csv'} ({len(rows)} rows)")
    return rows


def phase2_signal_sweep(probes):
    total = len(SIGNALS) * len(SWEEP_WEIGHTS) * len(probes)
    print(f"P2 signal sweep: {len(SIGNALS)} signals x {len(SWEEP_WEIGHTS)} weights "
          f"x {len(probes)} probes = {total} scans")
    path = RESULTS / "signal_sweep.csv"
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal", "weight", "probe_id", "true_label", "phrasing",
                        "is_prompt", "score", "band", "fired"])
            for sig in SIGNALS:
                for wt in SWEEP_WEIGHTS:
                    set_weight(sig, wt)
                    for p in probes:
                        d, _, _ = channels(policy_test(p["text"]))
                        names = [s["name"] for s in d.get("signals", [])]
                        w.writerow([sig, wt, p["id"], p["true_label"], p["phrasing"],
                                    p["is_prompt"], d.get("score", -1),
                                    d.get("band", "error"), int(sig in names)])
                    reset_weight(sig)
                print(f"   {sig} done")
    finally:
        for sig in SIGNALS:
            reset_weight(sig)
    print(f"   wrote {path}")


def phase3_weight_search(probes):
    print(f"P3 weight search: {SEARCH_N} random weight vectors over "
          f"{len(SIGNALS)} signals x {len(probes)} probes")
    rng = random.Random(SEARCH_SEED)
    cfg_path = RESULTS / "weight_search.csv"
    sc_path = RESULTS / "weight_search_scores.csv"
    try:
        with open(cfg_path, "w", newline="") as cf, open(sc_path, "w", newline="") as sf:
            cw = csv.writer(cf)
            cw.writerow(["config_id"] + [f"w_{s}" for s in SIGNALS])
            sw = csv.writer(sf)
            sw.writerow(["config_id", "probe_id", "true_label", "phrasing",
                         "is_prompt", "score"])
            for cid in range(SEARCH_N):
                vec = {s: rng.randint(0, 100) for s in SIGNALS}
                for s, v in vec.items():
                    set_weight(s, v)
                cw.writerow([cid] + [vec[s] for s in SIGNALS])
                for p in probes:
                    d, _, _ = channels(policy_test(p["text"]))
                    sw.writerow([cid, p["id"], p["true_label"], p["phrasing"],
                                 p["is_prompt"], d.get("score", -1)])
                if (cid + 1) % 100 == 0:
                    print(f"   {cid+1}/{SEARCH_N}")
    finally:
        for s in SIGNALS:
            reset_weight(s)
    print(f"   wrote {cfg_path} and {sc_path}")


def phase4_threshold_sweep(probes):
    """Sweep the two thresholds that gate banding.

    noise_floor discards candidates below it, but band_low is what decides
    whether a surviving score is labelled at all, so both have to move
    together for the band label to reach low-scoring traffic.
    """
    print(f"P4 threshold sweep: {len(NOISE_FLOORS)} threshold values")
    path = RESULTS / "threshold_sweep.csv"
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "probe_id", "true_label", "phrasing",
                        "is_prompt", "score", "band"])
            for nf in NOISE_FLOORS:
                set_threshold("noise_floor", nf)
                set_threshold("band_low", nf)
                for p in probes:
                    d, _, _ = channels(policy_test(p["text"]))
                    w.writerow([nf, p["id"], p["true_label"], p["phrasing"],
                                p["is_prompt"], d.get("score", -1),
                                d.get("band", "error")])
            reset_threshold("noise_floor")
            reset_threshold("band_low")
    finally:
        reset_threshold("noise_floor")
        reset_threshold("band_low")
    print(f"   wrote {path}")


def phase5_defense(probes):
    print(f"P5 defense: {len(CUSTOM_FLAGS)} custom risk flags")
    path = RESULTS / "defense.csv"
    try:
        for spec in CUSTOM_FLAGS:
            delete_custom_flag(spec["name"])
            add_custom_flag(spec)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe_id", "true_label", "phrasing", "is_prompt",
                        "score", "band", "flags", "custom_flags"])
            custom_names = {s["name"] for s in CUSTOM_FLAGS}
            for p in probes:
                d, _, _ = channels(policy_test(p["text"]))
                flags = d.get("risk_flags", [])
                w.writerow([p["id"], p["true_label"], p["phrasing"], p["is_prompt"],
                            d.get("score", -1), d.get("band", "error"),
                            "|".join(flags),
                            "|".join(x for x in flags if x in custom_names)])
    finally:
        for spec in CUSTOM_FLAGS:
            delete_custom_flag(spec["name"])
    print(f"   wrote {path}")


def phase6_carriers(probes):
    """Does the carrier filename change what the scanner can see?

    The structural signals are recovered by a code-aware detector, so the same
    text may or may not produce a prompt candidate depending on the filename
    the service is told to treat it as.
    """
    print(f"P6 carriers: {len(CARRIERS)} filenames x {len(probes)} probes")
    path = RESULTS / "carriers.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "probe_id", "true_label", "phrasing", "is_prompt",
                    "direct_score", "n_nodes", "node_score", "node_band",
                    "node_signals"])
        for fn in CARRIERS:
            for p in probes:
                d, nodes, top = channels(policy_test(p["text"], filename=fn))
                w.writerow([fn, p["id"], p["true_label"], p["phrasing"], p["is_prompt"],
                            d.get("score", -1), len(nodes),
                            top.get("score", 0) if top else 0,
                            top.get("band", "none") if top else "none",
                            "|".join(s["name"] for s in top.get("signals", []))
                            if top else ""])
    print(f"   wrote {path}")


def phase7_sentences(probes):
    """Collect the per-sentence analysis the full scan pipeline produces.

    policy-test returns whole-text scoring only. The sentence channel is
    produced by the stored-scan path, so each probe is submitted as a scan,
    read back, and then deleted so the service is left as it was found.
    """
    print(f"P7 sentence analysis: {len(probes)} probes")
    path = RESULTS / "sentences.csv"
    raw = open(RESULTS / "sentences_raw.jsonl", "w")
    created = []
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe_id", "true_label", "phrasing", "is_prompt",
                        "sent_index", "n_sentences", "start", "end", "text",
                        "tags", "score", "risk", "n_matches"])
            for i, p in enumerate(probes, 1):
                r = S.post(f"{BASE}/api/prompt-scans",
                           json={"mode": "paste", "filename": "pasted.txt",
                                 "text": p["text"]}, timeout=60)
                r.raise_for_status()
                sid = r.json()["id"]
                created.append(sid)
                d = S.get(f"{BASE}/api/prompt-scans/{sid}", timeout=60).json()
                res = d.get("result", d)
                nodes = res.get("prompt_nodes") or []
                sents = []
                for n in nodes:
                    sents.extend(n.get("sentence_analysis") or [])
                raw.write(json.dumps({"probe_id": p["id"],
                                      "true_label": p["true_label"],
                                      "phrasing": p["phrasing"],
                                      "text": p["text"],
                                      "sentences": sents}) + "\n")
                for j, sent in enumerate(sents):
                    w.writerow([p["id"], p["true_label"], p["phrasing"],
                                p["is_prompt"], j, len(sents),
                                sent.get("start", -1), sent.get("end", -1),
                                sent.get("text", "").replace("\n", " "),
                                "|".join(sent.get("tags") or []),
                                sent.get("score", 0), bool(sent.get("risk")),
                                len(sent.get("matches") or [])])
                S.delete(f"{BASE}/api/prompt-scans/{sid}", timeout=60)
                created.remove(sid)
                if i % 20 == 0:
                    print(f"   {i}/{len(probes)}")
    finally:
        raw.close()
        for sid in list(created):
            try:
                S.delete(f"{BASE}/api/prompt-scans/{sid}", timeout=30)
            except Exception:
                print(f"   could not delete scan {sid}")
    print(f"   wrote {path}")


def load_external():
    """Load the deepset prompt-injections test split from the local cache.

    This is a third-party labelled corpus, so it tests the channels on text
    nobody involved in this paper wrote. Label 1 is injection, 0 is benign.
    """
    import glob
    base = os.path.expanduser(CFG["external_dataset_glob"])
    hits = glob.glob(base, recursive=True)
    if not hits:
        raise SystemExit(f"external dataset not found at {base}")
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(sorted(hits)[0]) as src:
        table = ipc.open_stream(src).read_all()
    return table.to_pylist()


def phase8_external(probes):
    """Replay the channel comparison on a third-party labelled corpus."""
    rows = load_external()
    print(f"P8 external validation: {len(rows)} rows from the deepset test split")
    path = RESULTS / "external.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["idx", "label", "text_len", "direct_score", "direct_band",
                "direct_signals", "direct_flags"]
        for mid in GUARD_MODELS:
            cols += [f"{tag(mid)}_unsafe", f"{tag(mid)}_codes"]
        w.writerow(cols)
        for i, r in enumerate(rows):
            text = r["text"]
            d, _, _ = channels(policy_test(text))
            out = [i, int(r["label"]), len(text), d.get("score", -1),
                   d.get("band", "error"),
                   "|".join(x["name"] for x in d.get("signals", [])),
                   "|".join(d.get("risk_flags", []))]
            for mid in GUARD_MODELS:
                try:
                    g = guard_classify(text, mid, explain=False)
                    out += [bool(g.get("unsafe")), ",".join(g.get("codes", []))]
                except Exception as e:
                    out += ["error", str(e)[:40]]
            w.writerow(out)
            if (i + 1) % 25 == 0:
                print(f"   {i+1}/{len(rows)}")
    print(f"   wrote {path}")


# ---------------------------------------------------------------- main

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    probes = load_probes()
    print(f"Loaded {len(probes)} probes from {PROBE_FILE}")

    restore_defaults()
    scoring = assert_defaults_match()

    guard_prompt = S.get(f"{BASE}/api/prompt-scans/guard-prompt", timeout=30).json()
    risk_flags = S.get(f"{BASE}/api/prompt-scans/risk-flags", timeout=30).json()
    models = S.get(f"{BASE}/api/trained-models", timeout=30).json()
    models = models if isinstance(models, list) else models.get("models", [])
    meta = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": BASE,
        "scoring": scoring,
        "guard_prompt": guard_prompt,
        "risk_flags": risk_flags,
        "guard_models": [m for m in models if m.get("id") in GUARD_MODELS],
        "n_probes": len(probes),
        "repeats": REPEATS,
    }
    (RESULTS / "run_meta.json").write_text(
        json.dumps(redact_paths(meta), indent=1))
    print(f"   wrote {RESULTS/'run_meta.json'}")

    only = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    t0 = time.time()
    try:
        if "1" in only:
            phase1_baseline(probes)
        if "2" in only:
            phase2_signal_sweep(probes)
        if "3" in only:
            phase3_weight_search(probes)
        if "4" in only:
            phase4_threshold_sweep(probes)
        if "5" in only:
            phase5_defense(probes)
        if "6" in only:
            phase6_carriers(probes)
        if "7" in only:
            phase7_sentences(probes)
        if "8" in only:
            phase8_external(probes)
    finally:
        restore_defaults()
        assert_defaults_match()
        print("   service restored to default configuration")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
