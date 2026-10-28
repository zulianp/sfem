# Build the article; `make tables` regenerates the cost-model inputs first.
ROOT := ../..
PYTHON ?= python3

main.pdf: main.tex $(wildcard tables/*.tex) $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep -E "Warning|Error" || true

tables:
	$(PYTHON) $(ROOT)/spikes/inexact_apply_compare/qp/cost_model.py tables

.PHONY: tables
