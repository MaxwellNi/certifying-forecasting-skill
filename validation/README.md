# Reproduction checks

`unit_tests.json` records 304 distinct tests across all 11 distributed suites,
covering all 43 test files with no skips. The `unit_*.txt` transcripts retain
executed names and outcomes. Machine-specific directory and interpreter prefixes
are replaced with declared portable placeholders; local and published transcript
hashes are distinguished. The root [README](../README.md) lists every command.

`artifact_replay.json` records one complete aggregate invocation passing all 31
check groups. It replays stored arithmetic and designated primitive simulations,
including both separately frozen finite-law calibration protocols. Figures are
checked separately. `runtime_inventory.json` verifies unchanged Python sources,
nested result manifests and source overrides during that invocation.

The runs use installed Python 3.12.11 environments on the same host. They are
not new clean installations or independent external reproductions.
`installation.json` preserves its earlier installation receipt. The separate
source-to-model commands and retraining receipts remain with each study.

`sharp_calibration_integration.json` records the seven-law study integration;
`calibration_transcripts.json` identifies its plain-text execution records.
`figure_checks.json` and `display_fonts.json` describe current display checks.
Earlier attempts, correction receipts and exact source-design checks remain
available with their original scope; they do not imply additional current runs.

These checks verify specified calculations and transformations. Sampling
independence, historical feature availability and future decision gains require
their own evidence. An aggregate replay does not retrain all 41 original forecasts.
