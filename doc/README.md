# Documentation `sem2surface`

+ The theoretical documentation is available in `sem2surface.pdf`.
+ The source file is `SEM2surface_Documentation.md`.
+ To convert markdown documentation to PDF use `pandoc`
```
$ pandoc SEM2surface_Documentation.md -o sem2surface.pdf --pdf-engine=xelatex   --filter pandoc-crossref   --bibliography=references.bib   --csl=eng_bib_style.csl
```
