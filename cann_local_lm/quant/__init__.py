# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dopt GPTQ tooling + calibration data (top-level package, peer of converter).

This package holds the **offline GPTQ side**: dopt orchestration (subprocess
spawn + dopt_config generation), the vendored ``dopt_lm`` framework, and
calibration datasets. It does NOT contain the converter-side materialize/swap
logic — that lives at ``cann_local_lm/converter/export/quant_swap/`` co-located
with the export pipeline. Moved here from ``cann_local_lm/converter/quant/``
(2026-11-27); the CLI facade stays at ``cann_local_lm/converter/quant_entry.py``.

Submodules:
- ``auto_quant``: one-shot quantization (scene 2, spawn dopt subprocess)
- ``dopt_wrapper``: dopt subprocess body (model_define + weight_quant_pipline)
- ``download_dataset``: calibration jsonl download script
- ``dopt_lm/``: vendored GPTQ framework (independent CLI)
- ``calibrate_sets/``: calibration jsonl datasets
- ``config.yaml``: dopt GPTQ training config

Converter-side quant integration (materialize / QuantLinear / swap) is at
``cann_local_lm.converter.export.quant_swap``. See
docs/cann_lm_converter/quant/quant_design.md.
"""
