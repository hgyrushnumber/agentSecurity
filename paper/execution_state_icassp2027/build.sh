#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
if command -v tectonic >/dev/null 2>&1; then
  tectonic --untrusted --keep-logs --outdir build main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  (cd build && BIBINPUTS=..: BSTINPUTS=..: bibtex main)
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
else
  echo 'Install Tectonic or a TeX distribution with pdflatex and bibtex; alternatively upload the source files to Overleaf.' >&2
  exit 1
fi
echo 'PDF: build/main.pdf'
