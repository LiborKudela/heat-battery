# Makefile pro Návod k softwaru HeatBattery (PDF)
#
# Standardní použití:
#   make                  # přeloží navod.tex pomocí xelatexu (3 průchody)
#                         # pokud je PDF aktuální, nedělá nic
#   make rebuild          # vynutí rebuild i když je PDF aktuální
#   make ENGINE=lualatex  # přeloží přes lualatex (alternativa)
#   make watch            # latexmk v continuous-watch módu (vyžaduje latexmk)
#   make view             # otevře vygenerované PDF (xdg-open)
#   make clean            # smaže mezivýsledky, ponechá PDF
#   make distclean        # smaže i PDF
#
# Engine musí podporovat fontspec/polyglossia (xelatex nebo lualatex).
# pdflatex nestačí — vyhodí cannot-use-pdftex.

ENGINE ?= xelatex
DOC    := navod
SRC    := $(DOC).tex
PDF    := $(DOC).pdf

# Soubory mazané při `clean`
AUX_FILES := \
  $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
  $(DOC).fdb_latexmk $(DOC).fls $(DOC).synctex.gz \
  $(DOC).bbl $(DOC).blg $(DOC).nav $(DOC).snm \
  $(DOC).vrb

.PHONY: all rebuild watch view clean distclean help

all: $(PDF)

# Vynucený rebuild i kdy7 je PDF aktualnejsi nez zdroj
rebuild:
	@$(MAKE) -B all

$(PDF): $(SRC)
	@echo ">>> Překládám $(SRC) přes $(ENGINE) (3 průchody pro obsah a křížové odkazy)"
	$(ENGINE) -interaction=nonstopmode -halt-on-error $(SRC)
	$(ENGINE) -interaction=nonstopmode -halt-on-error $(SRC)
	$(ENGINE) -interaction=nonstopmode -halt-on-error $(SRC)
	@echo ">>> Hotovo: $@"

watch:
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk není dostupný"; exit 1; }
	latexmk -$(ENGINE) -pvc -interaction=nonstopmode $(SRC)

view: $(PDF)
	@command -v xdg-open >/dev/null 2>&1 && xdg-open $(PDF) || \
	  command -v open      >/dev/null 2>&1 && open $(PDF)     || \
	  echo "Otevřete ručně $(PDF)"

clean:
	@echo ">>> Mažu mezivýsledky"
	@rm -f $(AUX_FILES)

distclean: clean
	@rm -f $(PDF)

help:
	@echo "Cíle: all, rebuild, watch, view, clean, distclean"
	@echo "Proměnné: ENGINE=$(ENGINE)  (xelatex nebo lualatex)"
