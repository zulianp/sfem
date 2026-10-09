# The long-form companion. No generated inputs: everything here is written, not measured.
PY      ?= python3

.PHONY: all clean check

all: main.pdf

main.pdf: main.tex $(wildcard sections/*.tex)
	latexmk -pdf -interaction=nonstopmode main.tex

# The gates that matter for a document with no generated numbers: it builds, it has no
# undefined references, and no section is an empty stub.
check: main.pdf
	@echo "--- undefined references"
	@! grep -qaiE "undefined (citation|reference)" main.log || \
	  { grep -aiE "undefined (citation|reference)" main.log | sort -u | head; exit 1; }
	@echo "--- LaTeX errors"
	@! grep -qaE "^! " main.log || { grep -aE "^! " main.log | head; exit 1; }
	@echo "--- stub sections"
	@! grep -rqa "TODO\|FIXME\|\\\\stub" sections/ || \
	  { grep -rna "TODO\|FIXME\|\\\\stub" sections/ | head; exit 1; }
	@echo "all checks passed"

clean:
	latexmk -C main.tex 2>/dev/null || true
