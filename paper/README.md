# Paper (LaTeX, IEEE conference format)

## Compile on Overleaf (recommended)
1. Overleaf -> New Project -> Upload Project -> upload `paper.zip` (this folder zipped).
2. Menu -> Compiler: **pdfLaTeX**. Main document: `main.tex`. Overleaf has IEEEtran built in.

## Add the result figures (from the university PC)
Copy these three files into `paper/figures/`:
    work\runs\msgf_f9\eval_subset9\froc.png
    work\runs\msgf_f9\eval_subset9\reliability.png
    work\runs\msgf_f9\eval_subset9\triage.png
Until they are there, the PDF shows grey boxes instead.

## Red [PENDING: ...] text
Marks numbers that come from runs that are not finished yet (step 10 p-values, LUNA25,
candidate recall, app screenshots, author list/email). Search the .tex for "PENDING".

## References
All references are in `refs.bib`. Check every entry against the publisher page before submission.
