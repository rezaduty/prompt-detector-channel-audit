#!/bin/sh
# Rebuild everything from the raw results, then the PDF.
#
# The order matters. analyze.py regenerates paper/tables/macros.tex, and
# verify_numbers.py appends the \NVerifyChecks macro that Paper.tex cites, so
# the paper cannot be built until the verifier has run and passed.
set -e
cd "$(dirname "$0")"

# TinyTeX installs outside the default PATH.
if ! command -v pdflatex >/dev/null 2>&1; then
  PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
  export PATH
fi
command -v pdflatex >/dev/null 2>&1 || {
  echo "pdflatex not found. Install TinyTeX, MacTeX, or TeX Live." >&2
  exit 1
}

python3 src/analyze.py
python3 src/verify_numbers.py
python3 src/make_figures.py
python3 src/check_figures.py
sh paper/figures/render_html_figs.sh

cd paper
pdflatex -interaction=nonstopmode Paper.tex >/dev/null
bibtex Paper >/dev/null
pdflatex -interaction=nonstopmode Paper.tex >/dev/null
pdflatex -interaction=nonstopmode Paper.tex >/dev/null

if grep -qiE 'Warning.*(undefined|Citation|Reference)' Paper.log; then
  echo "build produced reference warnings:" >&2
  grep -iE 'Warning.*(undefined|Citation|Reference)' Paper.log >&2
  exit 1
fi
echo "built $(pwd)/Paper.pdf"
