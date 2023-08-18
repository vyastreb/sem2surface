# MD to PDF converter

To convert markdown documentation to PDF use `pandoc`
```
$ pandoc SEM2surface_Documentation.md -o Doc.pdf --pdf-engine=xelatex --filter pandoc-crossref --include-in-header=setup.tex
```
