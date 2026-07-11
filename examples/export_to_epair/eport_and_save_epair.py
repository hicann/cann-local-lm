import torch
import torch.nn as nn
from torch.export import export
from simple_transformer import SimpleTransformer

# 加载模型
VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 8
D_FF = 256
N_LAYERS = 2
MAX_SEQ_LEN = 128
BATCH_SIZE = 1
SEQ_LEN = 8

def load_model(weights_path: str | Path, device: torch.device) -> SimpleTransformer:
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN).eval().to(device)
    path = Path(weights_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=str(device))
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model

def build_sample_inputs(device: torch.device) -> tuple[torch.Tensor]:
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    return (tokens,)

if weights_path is not None:
    model = load_model(weights_path, device)
else:
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN).eval().to(device)

# 导出为 ExportedProgram
device = "CPU"
sample_inputs = build_sample_inputs(device)
ep = export(model, example_inputs)

# 转换为 EpairContents
contents = export_to_epair(ep)

# 查看转换统计
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")
print(f"Backend: {contents.manifest.backend_name}")

# 保存为 epair 文件
epair_save(contents, "model.epair")
print("Saved to model.epair")