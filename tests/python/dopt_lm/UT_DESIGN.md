# dopt_lm 单元测试设计报告

## 1. 文档信息

- 被测对象：`cann_lm/converter/quant/dopt_lm`
- 测试归档目录：`tests/python/dopt_lm`
- 测试框架：`pytest`
- 基准环境：Python 3.11、PyTorch CPU
- 设计日期：2026-08-27

## 2. 测试目标与范围

本轮 UT 验证 `dopt_lm` 的配置解析、模块替换、伪量化、GPTQ 核心计算、数据分发和训练/PTQ 编排。测试必须满足以下原则：

1. 默认在 CPU 离线运行，不下载 Hugging Face 模型或数据集。
2. 数值算法使用小尺寸确定性 Tensor，验证数值、形状、dtype、状态和梯度。
3. Transformers、Datasets、Accelerate、NPU/CUDA 和文件模型通过 mock 隔离。
4. 不把真实大模型 GPTQ、长时 KD 训练和硬件算子验证混入 UT。
5. 生产代码导入路径按工具实际运行方式设置：`PYTHONPATH=cann_lm/converter/quant`，`DEVICE=cpu`。

范围内模块：

- `builder.py`、`common_tools/register.py`、`common_tools/log.py`
- `utils.py`、`do_opt.py`、`opt_main.py`
- `modules/qlinear.py`、`modules/qembed.py`、`modules/utils.py`
- `quantizer/strategy.py`、`qmanager.py`、`qutils.py`、`qeval.py`、`qmin_max.py`
- `opt_funcs/weight_quant_ptq/ptq_algo_hessian/{quant,ptq_algo,modelutils,datautils,ptq_loop}.py`
- `opt_funcs/eco_stage/train.py` 中可隔离的纯函数和 orchestration 分支

范围外项目：

- 真实 LLM/VLM 权重加载、真实数据集下载和全模型 `llama_sequential` 运行
- CUDA/NPU kernel、显存管理效果、分布式训练和性能指标
- CANN 导出及 converter 下游消费（属于集成/系统测试）

## 3. 代码分析结论

### 3.1 架构与调用关系

`opt_main`/`weight_quant_pipline` 是入口；`do_opt` 把 JSON 层策略展开并将原始 `Linear`/`Embedding` 替换成量化包装模块；`QClass` 从全局 `Quantizers` 注册表构造 weight/input/output quantizer；`qmin_max` 和 `qeval` 实现静态、动态、分组和可训练伪量化；GPTQ 路径由 `ptq_loop` 收集激活，`WeightQuantHessian` 累积 Hessian 并按组更新权重；最后 `train.py` 把 GPTQ scale/weight 适配回 CANN 轻量化工具模型并保存。

### 3.2 主要风险

| 风险域 | 代码风险 | UT 关注点 |
|---|---|---|
| 注册与插件 | 动态 import 会吞掉插件异常；注册名覆盖和缺失查找 | 注册、覆盖、异常、递归/单文件 import |
| 策略配置 | 默认配置被调用方修改、`;group_size` 解析、LoRA 嵌套合并 | 深拷贝、覆盖优先级、非法策略 |
| 模块替换 | 精确类型、类名和 `target_op` 三种映射；嵌套名字替换 | 映射优先级、float 跳过、未知类型 |
| 量化数值 | signed/unsigned clip、零 scale、广播、dtype 恢复 | 边界值、scale 下限、逐通道形状、STE 梯度 |
| 分组量化 | `cin % group_size`、二维权重约束 | 合法/非法 shape、输出 shape/dtype |
| 动态 A8 | 返回 `(normalized, scale)`，由 `QLinear` 反归一化 | 动态路径、bias 顺序、有限值 |
| GPTQ | Hessian 累积、dead column、Cholesky、group/act-order | 小矩阵可逆性、分组参数、权重写回、资源释放 |
| 数据加载 | `.pt` 与 JSON 分流、比例切片、随机性 | dispatch、切片、合并、固定 seed |
| 编排 | KD/PTQ 分支、CPU fp16 回退、tie embedding/lm_head | 参数透传、状态切换、保存分支、共享 quantizer |
| 设备 | 默认设备取环境变量，CPU 环境不可触发 CUDA/NPU | 测试启动前强制 `DEVICE=cpu`，设备 API mock |

## 4. 测试方法与判定准则

- 浮点结果使用 `torch.testing.assert_close`，一般容差 `rtol=1e-5, atol=1e-6`；低精度动态量化按量化步长放宽。
- 随机用例固定 `torch.manual_seed`/NumPy/Python seed。
- 每条测试独立使用 `tmp_path`、`monkeypatch` 和局部假对象，避免全局注册表、环境变量及 logger handler 污染。
- 异常分支同时检查异常类型和关键消息。
- 所有 UT 必须可重复执行；测试目录整体退出码为 0 才算通过。
- 覆盖率目标：纯逻辑/数值模块行覆盖率不低于 80%；`train.py`、`ptq_loop.py` 的大模型执行体以 orchestration 分支覆盖为主，不以全行覆盖率作为 UT 准入条件。

## 5. 用例设计

### 5.1 注册、工具与日志

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| REG-001 | `Register.register_func/get` | 默认名和指定名均可注册/获取，decorator 返回原函数 | P0 |
| REG-002 | `Register` | `in`、下标、keys、重复注册覆盖行为正确 | P1 |
| REG-003 | `Register.get/register_func` | 缺失 key、非 callable 注册抛出明确异常 | P0 |
| REG-004 | `Register.file_import` | 单文件、目录递归、skip 集合和导入失败告警路径 | P1 |
| BLD-001 | `init_builder` | 内置 modules/quantizer 被递归导入 | P0 |
| BLD-002 | `init_builder` | 相对插件路径拒绝；绝对插件目录只导入 `.py/.so` | P1 |
| UTL-001 | `replace_module_by_names` | 顶层/深层模块按完整名称替换，未命中保持不变 | P0 |
| LOG-001 | `FileTxtLogger` | recorder 路径、普通记录、force 记录和未知记录跳过 | P1 |
| LOG-002 | `set_record_mode/visualize_dist` | 按 `layer_name` 注册；两类 scale 项正确输出 | P2 |

### 5.2 策略解析与模型优化

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| STR-001 | `get_optimize_cfg` | float/预置策略返回正确配置且每次为深拷贝 | P0 |
| STR-002 | `get_optimize_cfg` | `strategy;group_size` 覆盖 weight group size | P0 |
| STR-003 | `get_optimize_cfg` | 未知策略、非法 group size 抛出异常 | P0 |
| OPT-001 | `get_target_op_cls` | 精确类型、类名、策略 fallback、未知类型映射 | P0 |
| OPT-002 | `generate_config_file` | 只生成支持层，默认策略为 float；已存在文件拒绝覆盖 | P0 |
| OPT-003 | `trim_dict` | 递归移除 `None`/空 dict，非 dict 原样返回 | P1 |
| OPT-004 | `parse_layer_config` | 默认值、标量/字典覆盖、未知扩展项、LoRA 嵌套策略合并 | P0 |
| OPT-005 | `paser_quant_config` | 空路径、未配置层、float 层和有效量化层分流 | P0 |
| OPT-006 | `get_optimized_op` | 显式 `target_op` 优先；未知源类型拒绝；float 返回原模块 | P0 |
| OPT-007 | `optimize_model` | 嵌套 Linear/Embedding 被替换，float 层保留，权重/层名保持 | P0 |
| OPT-008 | state helpers | calibrate/weight/input 状态只传给具备 quant_op 的模块；simulator 同步 | P1 |
| CLI-001 | `opt_main` | CLI 参数解析和两个 wrapper 函数完整透传 | P2 |

### 5.3 Quantizer 管理与量化包装模块

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| QMG-001 | `QClass.__init__` | weight/input/output 配置构造对应注册 quantizer，输入配置不被修改 | P0 |
| QMG-002 | `QClass` | enable 开关、set state、forward 调用次序、empty 判定 | P0 |
| QMG-003 | `QClass.set_calibrate_state` | 只向启用且支持该方法的 quantizer 传播 | P1 |
| MOD-001 | `get_quantizer_cfg` | None/部分/完整配置拆分正确 | P1 |
| MOD-002 | `QLinear` | 无量化和静态量化 forward 与参考 `F.linear+bias` 一致/符合伪量化 | P0 |
| MOD-003 | `QLinear` | dyn_s8 路径正确反归一化，shape/dtype/有限值正确 | P0 |
| MOD-004 | `QEmbedding` | per-channel 强制、tie 与 clone 两种权重语义、forward 参数透传 | P0 |
| MOD-005 | `QGemma4ScaledEmbedding` | tensor/scalar embed scale 两条构造分支及缩放 forward | P0 |

### 5.4 基础量化数值函数

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| QUT-001 | `get_clip_range` | signed/unsigned、keep_neg_max、2.5 bit 及非法组合 | P0 |
| QUT-002 | min/max helpers | 全局 top-k、逐通道 min/max/abs-max 与非法 top-k | P0 |
| QUT-003 | quant/antiquant/fakequant | rounding、clip、广播和逆变换结果准确 | P0 |
| QUT-004 | `RoundWithGrad`/grad helpers | forward rounding，backward 为 STE，grad clamp 正确 | P1 |
| QUT-005 | `fake_nf` | 选择最近 label 并保留输入梯度 | P1 |
| QEV-001 | int/fp offset Function | scale clamp、clip、dtype 恢复和 backward 返回约定 | P0 |
| QEV-002 | dynamic S8 | 非零/全零输入归一化、scale、范围、dtype 和 backward | P0 |
| QEV-003 | `DynEvalGridS8` | buffers、从旧 quantizer 初始化、forward 透传 | P1 |

### 5.5 MinMax/Omni/Group 量化器

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| MMX-001 | `MinMaxGrid` | per-channel/per-tensor scale shape 与初始 buffers | P0 |
| MMX-002 | `MinMaxGrid.forward` | calibrate on/off，signed 和 unsigned 0/1/2/3 数值路径 | P0 |
| MMX-003 | `MinMaxGrid` | 1-bit、fp16 标志、非法 unsigned mode 异常 | P1 |
| MMX-004 | `GroupMinMaxGridWeight` | 合法分组量化保持 shape/dtype；非逐通道/不可整除拒绝 | P0 |
| MMX-005 | `LSQAutograd` | clamp/no-clamp forward 以及 x/step 梯度 | P1 |
| MMX-006 | `OmniMinMaxGrid` | calibrate 累计极值、signed/unsigned、参数梯度和非校准路径 | P0 |
| MMX-007 | `OmniGroupMinMaxGridWeight` | 二维/整除约束与合法 forward | P0 |

### 5.6 GPTQ 核心

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| GPT-001 | `quant.quantize` | 普通整数格点和 trits 分支数值准确 | P0 |
| GPT-002 | `Quantizer.configure` | bit、sym/asym、trits 和 enable 状态正确 | P0 |
| GPT-003 | `Quantizer.find_params` | weight 及 2D/3D/4D activation 的 per-channel/global shape | P0 |
| GPT-004 | `Quantizer.find_params` | 全零输入、MSE search、ready/identity-before-ready | P1 |
| GPT-005 | `WeightQuantHessian.add_batch` | Linear 2D/3D 与 Conv2D 输入 Hessian shape、样本数、累计值 | P0 |
| GPT-006 | `WeightQuantHessian.fasterquant` | 小 Linear 分组量化写回有限权重并生成 group scale/zero | P0 |
| GPT-007 | `WeightQuantHessian.fasterquant` | dead column、actorder、static_groups 和多 block 分支 | P1 |
| GPT-008 | `WeightQuantHessian.free` | Hessian 释放并调用设备 cache helper | P2 |
| DEV-001 | `modelutils.find_layers` | 默认和自定义类型递归名称正确 | P0 |
| DEV-002 | device helpers | CPU no-op；CUDA/NPU dispatch 通过 mock 验证 | P2 |

### 5.7 数据、训练与 PTQ 编排

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| DAT-001 | `set_seed` | NumPy/Torch 随机序列可重复 | P1 |
| DAT-002 | `get_pt_data` | 多文件按比例切片、shuffle 后合并 input_embed | P0 |
| DAT-003 | `get_loaders` | `.pt`/JSON 两条分发路径参数透传 | P0 |
| TRN-001 | `check_is_trainable` | 命中/未命中/空 key 集合 | P1 |
| TRN-002 | `ce_loss` | 全 logits、top-k mask 和 reduction 分支与参考 KL 一致 | P0 |
| TRN-003 | `DoptTrainerCustomization.compute_loss` | MSE、KL、组合、默认 loss、return_outputs 和无 teacher | P0 |
| TRN-004 | trainer grad | `grad_norm` 有/无梯度结果正确 | P1 |
| TRN-005 | `seed_all` | CPU 随机性可重复，设备分支不误调用 | P1 |
| TRN-006 | `model_define` | 普通 CausalLM 原样返回；多模态外壳提取 language model/tied head | P1 |
| TRN-007 | `prepare_model` | 对象/路径、quant wrapper、checkpoint refresh、trainable key/dtype | P0 |
| TRN-008 | weight adapter | GPTQ scale/offset 写入、embedding 校准、tie lm_head 和权重量化 | P0 |
| TRN-009 | `weight_quant_pipline` | YAML 读取归档、CPU fp16 回退、base_model 校验、KD/PTQ 路由 | P0 |
| TRN-010 | `ptq_quant_pipline` | GPTQ 参数透传、模型状态装载、tie/untie、pth/HF 保存分支 | P0 |
| PTL-001 | `sort_sequent/_get_weight_config/QuantArgs` | 层名分组、配置健壮性和默认参数 | P0 |
| PTL-002 | `get_llama` | model/device/layers/seqlen 状态变换正确 | P1 |
| PTL-003 | `weight_quant_loop` | 配置解析、loader、sequential 调用和 state_dict 返回完整透传 | P0 |

## 6. 计划目录

```text
tests/python/dopt_lm/
├── UT_DESIGN.md
├── conftest.py
├── test_register_builder.py
├── test_strategy_do_opt.py
├── test_strategy_generation.py
├── test_qmanager_modules.py
├── test_quantizer_utils.py
├── test_min_max.py
├── test_gptq_core.py
├── test_data_train_orchestration.py
└── test_opt_main.py
```

## 7. 执行方式

```bash
pytest -q tests/python/dopt_lm
```

如环境安装了 coverage，可附加执行：

```bash
coverage run --source=cann_lm/converter/quant/dopt_lm -m pytest tests/python/dopt_lm
coverage report -m
```

## 8. 准入与遗留风险

准入条件：全部用例通过、无外网访问、CPU 环境可重复、无测试间状态泄漏。真实 NPU GPTQ、全模型精度和耗时需要在后续集成测试中使用实际模型、校准集和设备单独验证；UT 通过不等价于量化精度达标。

当前没有 skip 或 xfail 用例；所有已收集用例均作为普通回归用例执行。

## 9. 执行结果（2026-08-27）

### 9.1 测试结果

```text
119 passed, 18 warnings in 33.05s
pytest exit code: 0
```

- 收集用例总数：119
- 普通回归用例：119，全部通过
- skip/xfail：0
- `ruff check tests/python/dopt_lm`：通过
- `python -m compileall -q tests/python/dopt_lm`：通过

18 条 warning 来自当前 Python 环境安装的 `torch_npu` 自动设备探测/存储兼容提示；测试启动时已设置 `DEVICE=cpu`，没有执行 NPU UT。warning 不影响 pytest 退出码。

### 9.2 覆盖率结果

执行命令：

```bash
COVERAGE_FILE=/tmp/dopt_lm_ut_80.coverage DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 \
  python -m coverage run --source=cann_lm/converter/quant/dopt_lm \
  -m pytest -q tests/python/dopt_lm
COVERAGE_FILE=/tmp/dopt_lm_ut_80.coverage python -m coverage report -m
```

整体行覆盖率为 **81%（1962 statements，370 missed）**，达到不低于 80% 的准入目标。核心模块覆盖情况：

| 模块 | 行覆盖率 |
|---|---:|
| `builder.py` | 100% |
| `register.py` | 94% |
| `do_opt.py` | 99% |
| `qembed.py` | 96% |
| `qlinear.py` | 82% |
| `qeval.py` | 100% |
| `qmanager.py` | 96% |
| `qmin_max.py` | 93% |
| `qutils.py` | 91% |
| `strategy.py` | 100% |
| `train.py` | 82% |
| GPTQ `datautils.py` | 98% |
| GPTQ `quant.py` | 96% |
| GPTQ `ptq_algo.py` | 91% |

剩余未覆盖代码主要位于 `ptq_loop.py` 的完整 LLM layer-by-layer GPTQ 执行体（该文件 24%）。其配置解析、loader 分发和顶层 orchestration 已由 UT 覆盖；真实 attention mask、共享 KV、PLE、逐层 hook、lm_head/投影量化需要实际模型语义，留给集成测试，避免通过过度 mock 获得失真的覆盖率。

### 9.3 缺陷修复记录

| ID | 原现象 | 修复内容 | 回归结果 |
|---|---|---|---|
| DEF-001 | `WeightQuantHessian.fasterquant(static_groups=True)` 未收集 group scale/zero，最终 `torch.cat([])` | 从预计算的静态 quantizer groups 收集 scale/zero；act-order 只对动态分组参数执行重排；拒绝非正 groupsize | static on/off × act-order on/off 四种组合全部通过，并校验静态 scale/zero 数值；新增非法 groupsize 用例 |
| REF-001 | `get_pangupi` 无法体现函数实际读取 JSONL 校准数据的职责 | 正式重命名为 `get_json_data`，同步内部调用和 UT | 旧名称无残留；JSON 分发、长短 token chunk 用例通过 |
| CFG-001 | W4A8 主策略名 `Quant_act_weight_eco` 与配置语义不一致 | dopt_lm、wrapper、converter、UT 和文档统一改为 `Quant_INT8-4_ECO`，不保留旧名兼容入口 | 策略生成/解析 UT 通过；两个真实模型 ST 的 591 个 Linear 策略均使用新名称 |
