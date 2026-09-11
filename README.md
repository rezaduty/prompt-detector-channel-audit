# Calibration Limits of Weighted Rule Signals and Taxonomy Label Inflation in Guard Model Classification of Prompt Injection

Measurement artifact for the paper of the same name. Everything reported in the
paper is reproduced from this repository, and a fail-closed verifier re-derives
every number from the raw outputs before the paper can be built.

Paper: [`paper/Paper.pdf`](paper/Paper.pdf) · Dataset documentation:
[`data/README.md`](data/README.md) · Licence: MIT (code), CC-BY-4.0 (data)

---

## What the study measures

A deployed layered prompt injection detector returns three things for one
input: a weighted salience score with a confidence band, a set of boolean risk
flags, and a guard-model taxonomy verdict. Operators see them through one
interface. This study drives the detector's own configuration interface to
determine what question each output actually answers.

| Finding | Evidence |
| --- | --- |
| The weighted score measures whether text is a **prompt**, not whether it is an **attack** | separates prompt-shaped text at 97.8% AUROC, attacks at 32.5%, below chance |
| Weight calibration cannot fix that | 1000 sampled weight vectors peak at 45.0% AUROC, never above chance; 7 of 10 weights never fire on chat traffic |
| The rule channel matches vocabulary, not intent | 8 of the 9 benign items its injection-phrase rule flags are security documentation quoting attack phrases |
| Guard models carry the binary verdict but not the label | neither assigns the jailbreak code to any jailbreak probe, despite 40 jailbreak rows in the trained model's training set |
| Degenerate multi-code verdicts inflate per-category recall | by 10.3 points on the zero-shot guard |
| A model's own recorded metric can be uninformative | recorded binary F1 of 1.0 against a 35.0% benign false-positive rate measured here, 75.0% on an external corpus |
| Public benchmarks can flatter a detector through a length confound | on the external corpus length alone scores 79.3% AUROC, higher than the score's 63.0%; stratifying by length collapses it to near chance |

## Repository layout

    config.json          endpoints, guard model ids, sweep ranges, added rules
    build.sh             regenerate everything, then the PDF
    requirements.txt

    data/
      README.md          dataset documentation, schema, provenance, limitations
      probes/probes.jsonl  the 91-probe diagnostic set

    src/
      build_probes.py    regenerates the probe set from literals
      run_experiment.py  the eight measurement phases
      analyze.py         metrics, LaTeX macros, tables
      verify_numbers.py  fail-closed verifier
      make_figures.py    data charts and the span figure
      check_figures.py   diagram geometry and figure/data agreement

    results/             raw CSVs, all 546 guard calls, captured configuration
    paper/               IEEEtran LaTeX, generated macros and tables, figures

## Measurement phases

| Phase | What it does | Scale |
| --- | --- | --- |
| P1 | baseline, both rule channels plus the guard panel | 91 probes, 546 guard calls |
| P2 | per-signal weight sweep, 0 to 100 | 14,560 scored cells |
| P3 | random search over the joint weight space | 1,000 vectors, 91,000 cells |
| P4 | noise-floor and band-threshold sweep | 16 values |
| P5 | rule extension, added then removed | 3 custom rules |
| P6 | carrier variation, does the filename change what is seen | 455 cells |
| P7 | per-sentence analysis, tags and spans | 111 sentences |
| P8 | external validation on a third-party corpus | 116 items |

## Running it

Requires the detection service on `http://127.0.0.1:8000`. Its interactive API
documentation is at `/docs`, with `/redoc` and `/openapi.json` alongside.

    pip install -r requirements.txt
    python3 src/build_probes.py       # regenerate data/probes/probes.jsonl
    python3 src/run_experiment.py     # all eight phases
    ./build.sh                        # analyze, verify, figures, PDF

Individual phases: `python3 src/run_experiment.py 1 8`.

### The experiment mutates server state

Phases P2 to P5 change signal weights, thresholds, and risk flags on the live
service. Every phase restores the shipped configuration in a cleanup block, the
run re-reads the live configuration afterwards and aborts if any weight differs
from its default, and the verifier checks the captured configuration again.

## Verification

`src/verify_numbers.py` is the gate, and the build refuses to run without it.
It re-derives every number from the raw result files using code written
independently of `analyze.py`, and additionally:

- recomputes every majority guard verdict from the log of all 546 individual calls
- confirms the probe order matches across phases, without which the paired
  before-and-after counts would compare different probes
- confirms every training corpus the paper names for a guard model appears in
  the service's own record of that model
- confirms the captured configuration carries no leftover weight overrides
- fails if the paper's length-confound claim stops matching the data
- rejects an undefined macro, a bare percentage typed into the prose, a cite key
  with no bibliography entry, a bibliography entry never cited, a
  cross-reference with no label, a missing figure or table file, and any of 16
  named methods or datasets used without a citation

`src/check_figures.py` checks the diagrams for canvas overflow, box overlap,
labels too large for their box, and type below 6.5pt, and re-reads the
worked-example figure's verdict chips to confirm they match the result CSVs.

Current run: **72 verifier checks** across 91 probes, 546 guard calls and
91,000 search cells, plus the figure checks.

## Building the paper

`./build.sh` runs analysis, verification, figures, then `pdflatex`/`bibtex`.
Any TeX distribution works. The build was done with TinyTeX plus
`ieeetran orcidlink cite booktabs caption listings placeins courier pgf breakurl`.

## Citing

See [`CITATION.cff`](CITATION.cff).

```bibtex
@article{rashidi2026calibration,
  author  = {Rashidi, Mohammadreza},
  title   = {Calibration Limits of Weighted Rule Signals and Taxonomy Label
             Inflation in Guard Model Classification of Prompt Injection},
  year    = {2026},
  note    = {Artifact: https://github.com/rezaduty/prompt-detector-channel-audit}
}
```

## Scope and limitations

Results describe one deployed detector and its two guard backends. Probes are
non-adaptive, so every detection rate is an upper bound. The probe set carries
one annotator and small per-group sizes; per-group comparisons are directional
unless a significance test is quoted. The trained guard was fine tuned on 60
rows from the external corpus, so its numbers on that corpus are an optimistic
bound. See the paper's limitations section and
[`data/README.md`](data/README.md).

## Ethics

All measurements ran against a detection service on the author's own machine
using benign canary-style payloads. No third-party service was exercised and no
harmful content was generated.
