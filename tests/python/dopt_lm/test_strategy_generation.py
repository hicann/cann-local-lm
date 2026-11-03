import json

import torch

from cann_local_lm.converter.quant import dopt_wrapper
from cann_local_lm.converter import swap


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 4)
        self.embed_tokens = torch.nn.Embedding(8, 4)


def test_automatic_generators_use_canonical_int8_4_eco_strategy():
    model = ToyModel()

    strategies = dopt_wrapper.build_layer_strategy(model, "toy")

    assert strategies["proj"]["quant_strategy"] == "Quant_INT8-4_ECO"
    assert strategies["lm_head"]["quant_strategy"] == "Quant_INT8-4_ECO"
    assert strategies["embed_tokens"]["quant_strategy"] == "Quant_Embed_MinMax"


def test_converter_reads_int8_4_eco_defaults_and_group_override(tmp_path):
    config_path = tmp_path / "dopt_config.json"
    config_path.write_text(
        json.dumps(
            {
                "layer_strategy": {
                    "proj": {"quant_strategy": "Quant_INT8-4_ECO"},
                    "proj_override": {"quant_strategy": "Quant_INT8-4_ECO;64"},
                }
            }
        )
    )

    params = swap._read_quant_param_map(config_path)

    assert params["proj"] == {
        "group_size": 128,
        "bit": 4,
        "blocksize": 128,
        "per_channel": True,
        "weight_algo": "omni_group_min_max",
    }
    assert params["proj_override"]["group_size"] == 64
