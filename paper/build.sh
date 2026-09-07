#!/usr/bin/env bash
# Run from any directory. Only paper/build contains generated TeX output.
set -euo pipefail
PAPER_BUILD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PAPER_BUILD_DIR"
"${PAPER_PYTHON_BIN:-python3}" check_paper.py

if command -v latexmk >/dev/null 2>&1; then
    mkdir -p build
    latexmk -norc -pdf -interaction=nonstopmode -halt-on-error \
        -pdflatex='pdflatex -no-shell-escape %O %S' -outdir=build main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    mkdir -p build
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    BIBINPUTS="$PAPER_BUILD_DIR:${BIBINPUTS:-}" bibtex build/main
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
else
    printf '%s\n' 'No LaTeX compiler found. Install latexmk or pdflatex + bibtex and the required TeX packages.' >&2
    exit 2
fi
printf 'Draft PDF: %s/build/main.pdf\n' "$PAPER_BUILD_DIR"
printf '%s\n' 'Still required: PDF page/visual checks and resolution of all submission placeholders.'
