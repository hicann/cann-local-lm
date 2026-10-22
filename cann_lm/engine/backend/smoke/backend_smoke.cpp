/**
 * backend_smoke：backend 全链路调用示例 + host 无 NDK so 环境验证——
 *  1) 演示上层经 api/ 中性工厂的完整调用形态：
 *     CreateBuiltModel → RestoreFromFile → GetTensorDescs
 *     → CreateModelManager → Init/InitWeights → tensor 工厂 → Run → DeInit；
 *  2) 工厂函数可创建对象且可析构；
 *  3) dlopen 失败路径优雅降级（wrapper 返回失败码/nullptr，进程不崩溃）。
 *
 * 断言语义：对无效输入返回 SUCCESS 视为失败（计入退出码）；
 * 具体非 SUCCESS 错误码不作断言（host/设备失败码可不同）。
 * 张量创建在 host 无 so 时 NULL 为预期，仅打印不作退出码断言；
 * 有 so 环境的行为由集成测试覆盖。
 */
#include "model/built_model.h"
#include "model_manager/model_manager.h"
#include "tensor/nd_tensor_buffer.h"
#include "infra/securestl.h"
#include <cstdio>
#include <memory>
#include <vector>

using namespace hiai;

int main()
{
    int failures = 0;

    // ── 1. 模型侧：创建 + 恢复 + 描述查询 ──
    auto builtModel = CreateBuiltModel();
    printf("CreateBuiltModel: %s\n", builtModel ? "OK" : "NULL");
    failures += (builtModel == nullptr);

    if (builtModel) {
        builtModel->SetName("backend_smoke_example");
        Status st = builtModel->RestoreFromFile("/nonexistent/model.om");
        printf("RestoreFromFile(none): status=%d（期望非 SUCCESS，不崩溃）\n", static_cast<int>(st));
        failures += (st == SUCCESS);

        auto inDescs = builtModel->GetInputTensorDescs();
        auto outDescs = builtModel->GetOutputTensorDescs();
        printf("GetTensorDescs: in=%zu out=%zu（无模型期望 0）\n", inDescs.size(), outDescs.size());
    }

    // ── 2. 模型管理器侧：创建 + Init + 权重 ──
    auto modelManager = CreateModelManager();
    printf("CreateModelManager: %s\n", modelManager ? "OK" : "NULL");
    failures += (modelManager == nullptr);

    if (modelManager && builtModel) {
        ModelInitOptions options;
        Status st = modelManager->Init(options, builtModel, nullptr);
        printf("MM Init: status=%d（无模型期望非 SUCCESS，不崩溃）\n", static_cast<int>(st));
        failures += (st == SUCCESS);

        st = modelManager->InitWeights("/nonexistent/weights");
        printf("MM InitWeights: status=%d（期望非 SUCCESS）\n", static_cast<int>(st));
        printf("MM GetModelID: %u\n", modelManager->GetModelID());

        // ── 3. 张量工厂：普通张量 + KV cache 张量 ──
        NDTensorDesc desc;
        desc.dims = {1, 2, 4, 8};
        desc.dataType = DataType::FLOAT16;
        desc.format = Format::ND;
        auto input = CreateNDTensorBuffer(desc);
        printf("CreateNDTensorBuffer: %s（host 无 so 期望 NULL）\n", input ? "OK" : "NULL");

        auto kv = CreateKVCacheBuffer(desc, 4, 64);
        printf("CreateKVCacheBuffer: %s（host 无 so 期望 NULL）\n", kv ? "OK" : "NULL");
        if (kv) {
            Status rst = ResizeKVCacheBuffer(kv, 128);
            printf("ResizeKVCacheBuffer: status=%d\n", static_cast<int>(rst));
        }

        // ── 4. 前向：标准 Run 形态（inputs 按 GetInputTensorDescs 装配）──
        std::vector<std::shared_ptr<INDTensorBuffer>> inputs;
        std::vector<std::shared_ptr<INDTensorBuffer>> outputs;
        st = modelManager->Run(inputs, outputs);
        printf("MM Run: status=%d（无模型期望非 SUCCESS，不崩溃）\n", static_cast<int>(st));

        modelManager->DeInit();
    }

    printf("SMOKE %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
