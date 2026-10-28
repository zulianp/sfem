# Build the article; `make tables` regenerates the cost-model inputs first.
ROOT := ../..
PYTHON ?= python3

# The format diagrams: TikZ sources in figures/formats, built to figures/fmt_*.pdf.
FORMATS := $(patsubst figures/formats/%.tex,figures/%.pdf,$(wildcard figures/formats/fmt_*.tex))

main.pdf: main.tex $(wildcard tables/*.tex) $(wildcard figures/*.pdf) $(FORMATS)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep -E "Warning|Error" || true

figures/fmt_%.pdf: figures/formats/fmt_%.tex figures/formats/common.tex
	cd figures/formats && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.. fmt_$*.tex >/dev/null
	rm -f figures/fmt_$*.aux figures/fmt_$*.log

tables:
	$(PYTHON) $(ROOT)/spikes/inexact_apply_compare/qp/cost_model.py tables

.PHONY: tables
