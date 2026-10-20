# cann-device-lm

本仓基于CANN在Kirin、昇腾等端、边芯片上，提供轻量、低延迟、低内存占用的LLM、VLM等大模型推理能力。
1.提供大模型转换工具
功能：支持将三方大模型转换为CANN LM大模型，包含量化、脚本优化、模型导出等步骤。
2.提供大模型推理引擎
功能：将LLM/VLM大模型tokenizer、embedding、prefill执行、decode执行、采样等步骤封装为统一的端到端工作流，提供端到端推理Pipeline。模型/算子编译、加载运行调用CANN的能力
技术栈：c++/python