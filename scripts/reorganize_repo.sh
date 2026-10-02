#!/usr/bin/env bash
# Moves the existing research files into the documented folder structure.
# Uses `git mv`, so file history is kept. Run once from the repo root (Git Bash on Windows):
#   bash scripts/reorganize_repo.sh
# Review with `git status` before committing.
set -euo pipefail

mv_if() {
  if [ -e "$1" ]; then mkdir -p "$(dirname "$2")"; git mv "$1" "$2"; echo "moved: $1 -> $2"
  else echo "skip (not found): $1"; fi
}

# Byte-identical "(1)" duplicate downloads
for f in "Journals/Next Likely Cancer Development Region/cancers-16-03097 (1).pdf" \
         "Journals/Confidence-Aware Decision Support for Radiologists/s41597-025-05742-x (1).pdf"; do
  if [ -e "$f" ]; then git rm -q "$f"; echo "removed duplicate: $f"; fi
done

# Proposal
mv_if "lung cancer proposal.docx"                         "docs/proposal/lung cancer proposal.docx"
mv_if "Lung Cancer Proposal Presentation Template.pptx"   "docs/proposal/Lung Cancer Proposal Presentation Template.pptx"

# Report / thesis chapters
mv_if "Chapter 1.rtf"                    "docs/thesis/Chapter 1.rtf"
mv_if "Chapter 2.rtf"                    "docs/thesis/Chapter 2.rtf"
mv_if "Methodology.docx"                 "docs/thesis/Methodology.docx"
mv_if "Research gap analysis.docx"       "docs/thesis/Research gap analysis.docx"
mv_if "Research gap analysis para.rtf"   "docs/thesis/Research gap analysis para.rtf"

# Working drafts
mv_if "Chapter1 gpt.rtf"                     "docs/drafts/Chapter1 gpt.rtf"
mv_if "Chapter 2 gpt.rtf"                    "docs/drafts/Chapter 2 gpt.rtf"
mv_if "Methodology gpt.docx"                 "docs/drafts/Methodology gpt.docx"
mv_if "Research gap analysis gpt.rtf"        "docs/drafts/Research gap analysis gpt.rtf"
mv_if "Research gap analysis para gpt.rtf"   "docs/drafts/Research gap analysis para gpt.rtf"

# Literature comparison spreadsheets
for f in "Confidence-Aware Decision Support for Radiologists.xlsx" \
         "Explainable Sequential Risk Prediction Across Slices.xlsx" \
         "Lung Cancer Detection.xlsx" \
         "Lung Cancer Prediction with Explainable Region Prioritization.xlsx" \
         "Multi-Level Interpretability.xlsx" \
         "Next Likely Cancer Development Region.xlsx" \
         "Research papers info.xlsx"; do
  mv_if "$f" "docs/literature-matrices/$f"
done

# Papers and summaries
mv_if "Journals"  "literature/journals"
mv_if "Summaries" "literature/summaries"

echo; echo "Done. Now run: git status"
