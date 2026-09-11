"""
Geometry check for the HTML concept diagrams.

Label overflow and collisions are not visible in a build log, so they are
checked here instead of by eye. For each diagram this script parses the
absolutely positioned elements and fails if:

  1. any element extends past the canvas
  2. any two boxes overlap
  3. a box sits outside the zone it belongs to
  4. the estimated text height exceeds the box height, meaning the label
     would be clipped or would spill over its border
  5. the rendered PDF would place any text below a minimum point size

Text height is estimated from the font size, line height, and an average
glyph width for the sans-serif face, which is conservative enough to catch
a label that does not fit.

Exits non-zero on any failure.
"""
import math
import pathlib
import re
import sys

FIGURES = pathlib.Path("paper/figures")

# Width of the rendered figure on the page, in inches, per diagram. Used to
# convert a CSS pixel size into the point size the reader actually sees.
RENDER_WIDTH_IN = {
    "fig_channels": 7.10,
    "fig_protocol": 7.10,
}
MIN_POINT_SIZE = 6.5
AVG_GLYPH_RATIO = 0.52   # average advance width as a fraction of font size
PAD_X = 10               # .box horizontal padding, each side
PAD_Y = 8                # .box vertical padding, each side

FAILURES = []


def style_dict(style):
    out = {}
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def px(v):
    m = re.match(r"(-?[\d.]+)px", (v or "").strip())
    return float(m.group(1)) if m else None


def parse(html):
    """Pull out canvas size, css font sizes, and positioned elements."""
    canvas = re.search(r"body\{width:(\d+)px;height:(\d+)px", html)
    W, H = (int(canvas.group(1)), int(canvas.group(2))) if canvas else (0, 0)

    fonts = {}
    for cls, size in re.findall(r"\.([a-zA-Z]+)\{[^}]*font-size:([\d.]+)px", html):
        fonts[cls] = float(size)

    els = []
    for m in re.finditer(
            r'<div class="([^"]+)"\s+style="([^"]*)"\s*>(.*?)</div>\s*(?=<div|\Z)',
            html, flags=re.S):
        classes = m.group(1).split()
        st = style_dict(m.group(2))
        inner = m.group(3)
        x, y = px(st.get("left")), px(st.get("top"))
        w, h = px(st.get("width")), px(st.get("height"))
        if x is None or y is None:
            continue
        title = re.search(r'<div class="t">(.*?)</div>', inner, flags=re.S)
        sub = re.search(r'<div class="s">(.*?)</div>', inner, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", inner)
        text = re.sub(r"\s+", " ", text).strip()
        els.append({
            "classes": classes, "x": x, "y": y, "w": w, "h": h,
            "title": (title.group(1).strip() if title else ""),
            "sub": (sub.group(1).strip() if sub else ""),
            "text": text, "is_box": "box" in classes, "is_zone": "zone" in classes,
        })
    return W, H, fonts, els


def wrapped_lines(text, font_px, inner_w):
    if not text or not inner_w:
        return 0
    per_line = max(1, int(inner_w / (font_px * AVG_GLYPH_RATIO)))
    words, lines, cur = text.split(), 1, 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur + add > per_line:
            lines += 1
            cur = len(word)
        else:
            cur += add
    return lines


def check_figure(name):
    path = FIGURES / f"{name}.html"
    if not path.exists():
        FAILURES.append(f"{name}: html source missing")
        return
    html = path.read_text()
    W, H, fonts, els = parse(html)
    if not W or not H:
        FAILURES.append(f"{name}: could not read canvas size")
        return

    scale = RENDER_WIDTH_IN.get(name, 7.10) * 72.0 / W
    for cls, size in sorted(fonts.items()):
        pt = size * scale
        if pt < MIN_POINT_SIZE:
            FAILURES.append(f"{name}: class .{cls} renders at {pt:.1f}pt, "
                            f"below the {MIN_POINT_SIZE}pt floor")

    boxes = [e for e in els if e["is_box"]]
    zones = [e for e in els if e["is_zone"]]

    # 1. inside the canvas
    for e in els:
        if e["w"] is None or e["h"] is None:
            continue
        if e["x"] < 0 or e["y"] < 0 or e["x"] + e["w"] > W or e["y"] + e["h"] > H:
            FAILURES.append(
                f"{name}: element {e['title'] or e['text'][:28]!r} at "
                f"({e['x']},{e['y']},{e['w']},{e['h']}) leaves the "
                f"{W}x{H} canvas")

    # 2. no two boxes overlap
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                    and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]):
                FAILURES.append(
                    f"{name}: boxes {a['title']!r} and {b['title']!r} overlap")

    # 3. a box that sits within a zone's span must sit within it fully
    for bx in boxes:
        for z in zones:
            cx, cy = bx["x"] + bx["w"] / 2, bx["y"] + bx["h"] / 2
            inside_centre = (z["x"] <= cx <= z["x"] + z["w"]
                             and z["y"] <= cy <= z["y"] + z["h"])
            if not inside_centre:
                continue
            if not (bx["x"] >= z["x"] and bx["y"] >= z["y"]
                    and bx["x"] + bx["w"] <= z["x"] + z["w"]
                    and bx["y"] + bx["h"] <= z["y"] + z["h"]):
                FAILURES.append(
                    f"{name}: box {bx['title']!r} crosses its zone border")

    # 4. text must fit inside the box
    ft = fonts.get("t", 13.0)
    fs = fonts.get("s", 10.0)
    for bx in boxes:
        if not bx["w"] or not bx["h"]:
            continue
        inner_w = bx["w"] - 2 * PAD_X
        h = 0.0
        if bx["title"]:
            h += wrapped_lines(bx["title"], ft, inner_w) * ft * 1.25
        if bx["sub"]:
            h += 5 + wrapped_lines(bx["sub"], fs, inner_w) * fs * 1.3
        need = h + 2 * PAD_Y
        if need > bx["h"]:
            FAILURES.append(
                f"{name}: label in box {bx['title']!r} needs about "
                f"{need:.0f}px but the box is {bx['h']:.0f}px tall")

    print(f"  {name}: {len(boxes)} boxes, {len(zones)} zones, "
          f"canvas {W}x{H}, smallest type "
          f"{min(fonts.values()) * scale:.1f}pt")


def check_span_figure():
    """The worked-example figure states a verdict per probe. Re-read those
    chips out of the generated HTML and compare them against the result CSVs,
    so a data-reading bug in the figure generator cannot reach the paper."""
    import csv as _csv
    path = FIGURES / "fig_spans.html"
    if not path.exists():
        FAILURES.append("fig_spans: html source missing")
        return
    html = path.read_text()
    base = {}
    with open("results/baseline.csv") as f:
        for row in _csv.DictReader(f):
            base[row["id"]] = row

    rows = re.findall(r'<div class="row [^"]*">(.*?)</div>\s*</div>', html, flags=re.S)
    texts = re.findall(r'<div class="txt">(.*?)</div>', html, flags=re.S)
    chips = re.findall(r'rules: (flagged|not flagged)', html)
    guards = re.findall(r'guard: ([^<]+)</span>', html)
    if not (len(texts) == len(chips) == len(guards)):
        FAILURES.append(f"fig_spans: {len(texts)} rows but {len(chips)} rule "
                        f"chips and {len(guards)} guard chips")
        return

    # Recover which probe each row shows by matching its plain text.
    def plain(h):
        t = re.sub(r"<[^>]+>", "", h)
        for ent, ch in (("&quot;", '"'), ("&amp;", "&"),
                        ("&lt;", "<"), ("&gt;", ">")):
            t = t.replace(ent, ch)
        return re.sub(r"\s+", " ", t).strip()

    import json as _json
    probe_text = {}
    for line in pathlib.Path("data/probes/probes.jsonl").read_text().splitlines():
        if line.strip():
            r = _json.loads(line)
            probe_text[re.sub(r"\s+", " ", r["text"]).strip()] = r["id"]

    for t, chip, guard in zip(texts, chips, guards):
        pid = probe_text.get(plain(t))
        if pid is None:
            FAILURES.append(f"fig_spans: row text does not match any probe: "
                            f"{plain(t)[:50]!r}")
            continue
        row = base[pid]
        want_flag = "flagged" if (row["direct_flags"] or "").strip() else "not flagged"
        if chip != want_flag:
            FAILURES.append(f"fig_spans: {pid} shows rules '{chip}' but "
                            f"baseline.csv says '{want_flag}'")
        codes = (row["model_67276875f520_codes"] or "").strip()
        want_guard = codes if codes else "safe"
        if guard.strip() != want_guard:
            FAILURES.append(f"fig_spans: {pid} shows guard '{guard.strip()}' "
                            f"but baseline.csv says '{want_guard}'")
    print(f"  fig_spans: {len(texts)} worked examples, chips match baseline.csv")


def main():
    print("Checking concept diagram geometry")
    for name in sorted(RENDER_WIDTH_IN):
        check_figure(name)
    check_span_figure()
    if FAILURES:
        print(f"FIGURE CHECK FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("FIGURE CHECK PASSED")


if __name__ == "__main__":
    main()
