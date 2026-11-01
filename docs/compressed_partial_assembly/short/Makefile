# The PASC27 short form; figures and tables come from the long form (make there first).
main.pdf: main.tex ../refs.bib $(wildcard ../tables/*.tex) $(wildcard ../figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	bibtex main >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep -E "Warning|Error" || true
