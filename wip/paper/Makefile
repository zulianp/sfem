# The paper. Figures and tables are GENERATED; nothing here edits them by hand.
#
# make            build the pdf
# make figures    regenerate every figure and table input from the measurements
# make check      the submission gates: clean build, no undefined refs, page limit
# make clean

PY      ?= python3
FIGDIR   = figures
TABDIR   = tables
# PASC27: 10 pages including figures, tables and appendices, EXCLUDING references.
# No extensions are granted, so this is a gate rather than a guideline.
PAGELIMIT ?= 10

.PHONY: all figures check clean selftest

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard $(FIGDIR)/*.tex) $(wildcard $(TABDIR)/*.tex)
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	$(PY) python/packed_format_figures.py --out $(FIGDIR)
	@if [ -f python/perf_figures.py ]; then \
	  $(PY) python/perf_figures.py --out $(FIGDIR) --tables $(TABDIR); \
	fi

selftest:
	$(PY) python/packed_format_figures.py --selftest
	@if [ -f python/perf_figures.py ]; then $(PY) python/perf_figures.py --selftest; fi

# The gates, in the order a failure is cheapest to fix.
check: selftest main.pdf
	@echo "--- undefined references / citations"
	@! grep -qaiE "undefined (citation|reference)" main.log || \
	  { grep -aiE "undefined (citation|reference)" main.log | sort -u | head; exit 1; }
	@echo "--- LaTeX errors"
	@! grep -qaE "^! " main.log || { grep -aE "^! " main.log | head; exit 1; }
	@echo "--- overfull boxes wider than 5pt"
	@! grep -aoE "Overfull \\\\hbox \(([0-9.]+)pt" main.log | awk -F'[(]' '$$2+0>5' | grep -q . || \
	  { grep -aoE "Overfull \\\\hbox \(([0-9.]+)pt" main.log | awk -F'[(]' '$$2+0>5' | head; exit 1; }
	@echo "--- page limit"
	@$(PY) python/check_page_limit.py --limit $(PAGELIMIT)
	@echo "all checks passed"

clean:
	latexmk -C main.tex
	rm -f main.bbl
