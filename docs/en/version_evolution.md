# cann-device-lm Version Evolution

English | [简体中文](../zh/version_evolution.md)

## Release Cadence

| # | Release Date | Version | Type | Milestone |
|---|---|---|---|---|
| R0 | 2026-10-30 (Fri) | `0.1.0-beta.1` | Beta | **Initial release**: core inference pipeline working, API preview; first compatibility table |
| R1 | 2026-11-30 (Mon) | `0.2.0-beta.1` | Beta | First round of API adjustments |
| R2 | 2026-12-31 (Thu) * | `0.3.0-beta.1` | Beta | Operator expansion, performance baseline; *adjacent to New Year's Day, may be moved up to 12-30 |
| R3 | 2027-01-29 (Fri) | `0.4.0` | 0.x | **Drop beta suffix**, API begins to freeze (1-31 is Sunday) |
| R4 | 2027-02-26 (Fri) | `0.5.0` | 0.x | Multi-chip / multi-model support (2-28 is Sunday) |
| R5 | 2027-03-31 (Wed) | `0.6.0` | 0.x | **HarmonyOS March mid-version (7.1) adaptation release** |
| R6 | 2027-04-30 (Fri) | `0.7.0` | 0.x | 6.1 adaptation fallback slot (if 7.1 releases in late March, it lands in this version) |
| R7 | 2027-05-31 (Mon) | `0.8.0` | 0.x | Stability hardening |
| R8 | 2027-06-30 (Wed) | `0.9.0` | 0.x | Feature completeness closure |
| R9 | 2027-07-30 (Fri) | `0.10.0` | 0.x | **API Freeze**, entering 1.0 convergence (7-31 is Saturday) |
| R10 | 2027-08-31 (Tue) | `1.0.0-rc.1` | RC | First 1.0 candidate |
| R11 | 2027-09-30 (Thu)** | `1.0.0-rc.2` | RC | **HarmonyOS 8.0 (September major release) full adaptation + GA-level regression**;**falls on the last working day before National Day, release review must be completed before the holiday |
| R12 | 2027-10-29 (Fri) | **`1.0.0`** | **GA** | **First stable release (2027-10)**: full support for HarmonyOS 6.1 / 7.0; hard commitments take effect |
| R13 | 2027-11-30 (Tue) | `1.1.0` | GA | First feature release after GA |

## Compatibility Table

| CDLM Version | Type | Release Date | HarmonyOS | CANN | Chip | PyTorch | TorchNPU |
|---|---|---|---|---|---|---|---|
| 0.1.0-beta.1 | Beta | 2026-10-30 | 7.0 (2026-09 major release) | 9.3.0 | kirinx90 | 2.9.0 | 26.1.1 |
| 1.0.0 | GA | 2027-10-29 | 7.1 + 8.0 (full support) | `<TBD>` | kirinx90, kirin9030... | `<TBD>` | `<TBD>` |

> For detailed version compatibility (torch_npu package versions, torch/torch_npu/CANN triple-binding rules, P1/P2 environment isolation), see [Environment Dependencies](env_dependencies.md) §2.2.
