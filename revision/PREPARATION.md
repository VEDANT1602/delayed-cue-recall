# Preparation record

Auxiliary texture prediction can delay early cue recall in a supervised recurrent network

Vedant Thaker · 21 November 2026

## Original work

Vedant Thaker is the sole author. He designed the original delayed cue-recall study, wrote its manuscript, developed its original software, and requested the comparisons and checks reported in this revision.

The author reports that the research project began on 2 August 2026. The previous versions and the new revision represent different stages of the work.

## Revision activities

OpenAI Codex assisted with the original rerun, new-seed training, registered confirmation, optimizer and gradient-gating comparisons, and analysis of the UCI activity benchmark, as well as literature checks and preparation of the revised text. The author set the research question and requested the revision.

## Computational checks

The 720-row baseline replay matched all original choice and erasure counts. One prediction from an untrained auxiliary head differed by a single symbol in 16,384, as detailed in the previous replication record. The extension ran 648 sweep fits, 24 activity fits and a separately registered 48 confirmatory fits on CPU. Per-seed and per-update records document the grouping and the accuracy denominators. Reference software and supplementary scripts specify the dependency versions.

## Data and version records

The baseline repository contains the original training program and its standard-library verifier. The supplement contains the earlier seed extension and its locks, the new experiment code, per-seed scores, per-update alignments and the scored confirmation. Commands appear in README.md. The public confirmatory plan is at https://github.com/VEDANT1602/delayed-cue-recall/blob/ae1c428475388cee0045d20cdd02f8e175da67cb/revision/confirmatory-plan.md.

[Original pinned software](https://github.com/VEDANT1602/delayed-cue-recall/tree/686f4f7e50229f061b4efdc5cb5d3f4812a2fc48) · [Revision code and data](https://github.com/VEDANT1602/delayed-cue-recall/tree/main/revision/supplement)

The reported checks are computational. No independent personal rerun or line-by-line review of these revisions by the author is asserted here.
