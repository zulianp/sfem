# The valve-containment report. Figures, tables and number macros are GENERATED from data/.
#
# make            build the pdf
# make figures    regenerate every figure, table and macro from data/
# make selftest   the generator's own checks
# make check      clean build, no undefined references, no LaTeX errors, no overfull boxes
# make clean

PY      ?= python3
FIGDIR   = figures
TABDIR   = tables

.PHONY: all figures check clean selftest

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard $(FIGDIR)/*.tex) $(wildcard $(TABDIR)/*.tex)
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	$(PY) python/ldp_figures.py --out $(FIGDIR) --tables $(TABDIR)

selftest:
	$(PY) python/ldp_figures.py --selftest

check: selftest main.pdf
	@echo "--- undefined references / citations"
	@! grep -qaiE "undefined (citation|reference)" main.log || \
	  { grep -aiE "undefined (citation|reference)" main.log | sort -u | head; exit 1; }
	@echo "--- LaTeX errors"
	@! grep -qaE "^! " main.log || { grep -aE "^! " main.log | head; exit 1; }
	@echo "--- overfull boxes wider than 5pt"
	@! grep -aoE "Overfull \\\\hbox \(([0-9.]+)pt" main.log | awk -F'[(]' '$$2+0>5' | grep -q . || \
	  { grep -aoE "Overfull \\\\hbox \(([0-9.]+)pt" main.log | awk -F'[(]' '$$2+0>5' | head; exit 1; }
	@echo "all checks passed"

clean:
	latexmk -C main.tex
	rm -f main.bbl
