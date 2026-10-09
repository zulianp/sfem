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
	  $(PY) python/perf_figures.py --out $(FIGDIR) --tables $(TABDIR) || \
	    echo "perf_figures: no measurements yet -- see README.md"; \
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
	@# PASC counts 10 pages EXCLUDING references, so the gate must too. The body ends on the
	@# page carrying the References heading -- unless that page holds nothing but references,
	@# in which case the body ended on the page before it. Counting a references-only page as a
	@# body page reported 11 for a paper whose body was 10 and would have forced a needless cut.
	@# "Nothing but references" is judged by what precedes the heading on that page: only the
	@# running header, which is one line.
	@total=$$(pdfinfo main.pdf | awk '/^Pages/{print $$2}'); \
	  refpage=$$(for i in $$(seq 1 $$total); do \
	      if pdftotext -f $$i -l $$i main.pdf - 2>/dev/null | grep -qaE '^References[[:space:]]*$$'; \
	        then echo $$i; break; fi; done); \
	  body=$${refpage:-$$total}; \
	  if [ -n "$$refpage" ]; then \
	    before=$$(pdftotext -f $$refpage -l $$refpage main.pdf - 2>/dev/null \
	              | sed -n '1,/^References[[:space:]]*$$/p' | grep -acvE '^[[:space:]]*$$'); \
	    [ "$$before" -le 2 ] && body=$$((refpage - 1)); \
	  fi; \
	  echo "    $$body body pages + references to page $$total (limit $(PAGELIMIT), excl. refs)"; \
	  [ "$$body" -le "$(PAGELIMIT)" ] || \
	    { echo "    OVER THE LIMIT -- PASC grants no extensions"; exit 1; }
	@echo "all checks passed"

clean:
	latexmk -C main.tex
	rm -f main.bbl
