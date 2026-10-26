import sys

from dopt_lm import opt_main


def test_parse_args_reads_all_cli_fields(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dopt",
            "--model-path",
            "model",
            "--dopt-config",
            "layers.json",
            "--optimize-config",
            "opt.yaml",
            "--output-dir",
            "output",
            "--hf_type_save",
        ],
    )
    args = opt_main.parse_args()
    assert vars(args) == {
        "model_path": "model",
        "dopt_config": "layers.json",
        "optimize_config": "opt.yaml",
        "output_dir": "output",
        "hf_type_save": True,
    }


def test_opt_main_wrappers_delegate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        opt_main, "generate_config", lambda *args: calls.append(("generate", args))
    )
    monkeypatch.setattr(
        opt_main,
        "weight_quant_pipline",
        lambda **kwargs: calls.append(("pipeline", kwargs)),
    )

    opt_main.generate_quant_config("model", "layers.json")
    opt_main.llm_quant_pipline(model_path="model", output_dir="output")

    assert calls == [
        ("generate", ("model", "layers.json")),
        ("pipeline", {"model_path": "model", "output_dir": "output"}),
    ]
