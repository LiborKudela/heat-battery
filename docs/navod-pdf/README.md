# Návod (PDF verze)

Tahle složka obsahuje LaTeX zdroj českého návodu (`navod.tex`), ze kterého
si vyrenderujete PDF lokálně. Důvod: některá prostředí (Cursor, GitHub
preview, různé Markdown rendery) nezobrazují všechny LaTeX vzorce
konzistentně, PDF je všude stejné.

## Co je potřeba

- TeX Live (typicky `texlive-full` na Ubuntu nebo MacTeX) — nutné kvůli
  `polyglossia`, `fontspec`, `microtype`, `tcolorbox`, `tikz`, `listings`.
- Engine: **`xelatex`** (preferováno) nebo **`lualatex`**. `pdflatex`
  v aktuální podobě **nestačí**, protože zdroj používá `fontspec`
  + `polyglossia`.
- TeX Gyre fonty (Termes / Heros / Cursor) jsou součástí `texlive-fonts-extra`
  / `texlive-full`. Pokud je nemáš, smaž v hlavičce `navod.tex` tři řádky
  `\setmainfont{...}`, `\setsansfont{...}`, `\setmonofont{...}` — TeX si
  vybere default, dokument zkompiluje pořád.

## Sestavení

### Varianta A: latexmk (nejpohodlnější)

```bash
cd docs/navod-pdf
latexmk -xelatex navod.tex      # nebo: latexmk -lualatex navod.tex
latexmk -c                      # úklid mezivýsledků (volitelně)
```

### Varianta B: Makefile (přiloženo)

```bash
cd docs/navod-pdf
make                # implicitně používá xelatex
make ENGINE=lualatex
make clean
```

### Varianta C: ručně

```bash
cd docs/navod-pdf
xelatex navod.tex
xelatex navod.tex      # druhý průchod kvůli odkazům a obsahu
```

(Pro `lualatex` postup stejný, jen místo `xelatex`.)

## Výstup

`navod.pdf` ve stejné složce. Obsahuje:

- celý český návod včetně architekturního diagramu (TikZ),
- správně vykreslené rovnice (slabá formulace, Newtonovo ochlazování,
  $\theta$-schéma…),
- syntax-highlighted ukázky kódu (Python, Bash, YAML),
- klikatelné křížové odkazy a obsah.

## Synchronizace s `README_NAVOD.md`

Zdroj pravdy zůstává `README_NAVOD.md` v kořeni repa. Tenhle `.tex` je
ručně udržovaná „zrcadlová“ verze pro PDF výstup. Pokud upravíš
`README_NAVOD.md`, projdi i `navod.tex` a doplň stejné změny — text
i strukturu jsem se snažil držet 1:1, takže `diff` mezi nimi by měl být
zhruba čitelný.
