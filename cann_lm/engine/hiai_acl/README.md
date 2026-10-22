# HIAI_ACL — ACL-like Shim for NDK

## 概述

`hiai_acl/` 是 NDK 的 ACL-like 适配层（shim），使 NDK 后端呈现与 real ACL 一致的 C API 表面。

NDK 后端实现（`ndk/`）调用本 shim 提供的 ACL 风格 C API（`aclInit`、`aclmdlLoadFromFile`、`aclmdlExecute` 等），shim 内部通过 `NDKProxy` 动态加载 `libhiai_foundation.so` + `libneural_network_core.so`，将 ACL API 调用转发到 NDK API。

两个后端实现代码几乎一致——差异仅在链接层：
- ACL: 链接 `libascendcl.so`（real runtime）
- NDK: 链接 `libhiai_ascendcl.so`（本 shim）

## 目录结构

```
hiai_acl/
├── src/
│   ├── acl_mdl.cpp          # ACL 模型 API（aclInit/LoadFromFile/Execute...）
│   ├── acl_rt.cpp           # ACL 运行时 API（SetDevice/CreateStream...）—— 虚拟 stream/event/context
│   ├── acl_manager.h        # ACLManager 单例（modelId→executor 映射）
│   ├── acl_manager.cpp
│   ├── acl_data.h           # aclDataBuffer / aclmdlDataset 结构定义
│   ├── acl_data.cpp
│   ├── acl_aipp.cpp         # AIPP API（保留，不纳入 LLM backend 接口）
│   └── ndk/                 # NDK 代理层（动态加载）
│       ├── ndk_proxy.h/cpp          # dlopen + dlsym
│       ├── dynamic_load_helper.h/cpp # .so handle + symbol cache
│       ├── hiai_ndk_nncore.h/cpp    # Compilation/Executor/Tensor ops
│       ├── hiai_ndk_executor.h/cpp   # Executor 辅助
│       ├── hiai_ndk_options.h/cpp    # Build options (shape/precision/tuning)
│       ├── hiai_ndk_tensor.h/cpp    # Tensor + AIPP + cache status
│       └── hiai_ndk_aipp.h/cpp      # AIPP 参数
├── utils/                   # 工具（日志/断言/字符串）
│   ├── log.h
│   ├── assert.h
│   ├── string_utils.h
│   ├── securestl.h
│   └── ir_option.h
├── CMakeLists.txt           # → libhiai_ascendcl.so
└── README.md                # 本文件
```

## 头文件说明

shim 不自维护 ACL 头副本；vendored ACL 头文件子集（`acl/acl_mdl.h` / `acl/acl_rt.h` / `acl/acl_base.h` + `acl/error_codes/`）位于 `../external_acl/`（与 `hiai_acl/` 平级）。签名与 real ACL 完全一致，未实现的函数返回 not supported。NDK 构建不需要 ACL SDK 已安装。

NDK 后端实现（`backend/ndk/`）编译时 `-I` 由 `hiai_ascendcl` PUBLIC 导出（指向 `../external_acl/`），`cann_backend` 经 `target_link_libraries(... PUBLIC hiai_ascendcl)` 继承。

## 来源

从 `cann-recipes-harmony-infer-master/harmony_acl/` 项目演进，补充以下 gap：
- `acl_rt.cpp`（新增）：虚拟 stream/event/context、aclrtSetDevice/GetDeviceCount
- `acl_mdl.cpp`（补充）：aclmdlExecuteAsync、aclmdlGetCurOutputDims、aclmdlGetInputDimsRange
- `acl_manager.cpp`（补充）：RunAsync、虚拟 stream 管理

## 设计文档

详见 `../backend/proposal.md` 4.6 HIAI_ACL 适配层设计。
