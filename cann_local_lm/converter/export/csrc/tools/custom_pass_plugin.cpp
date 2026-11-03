/* custom_pass_plugin.cpp — QuantBatchMatmulV4 format 修正 CustomPass 插件。
 *
 * 独立 .so, 用旧 ABI (_GLIBCXX_USE_CXX11_ABI=0) 编译以匹配 libregister.so。
 * cann_lm_compile 在 aclgrphBuildInitialize 前 dlopen 此 .so,
 * PassReceiver 静态初始化将 pass 注册到 libregister.so 全局注册表。
 *
 * Stage: kAfterOriginGraphOptimize (SecondInferformat 之后)
 *
 * arch31 s8s4 tiling 要求:
 *   x1       (in 0): int8     ND         — SecondInferformat 传播为 FRACTAL_Z, 改回 ND
 *   x2       (in 1): int4     FRACTAL_Z  — 保持 FRACTAL_Z, 恢复 storage shape
 *   bias     (in 2): float16  ND         — optional, 改回 ND
 *   x1_scale (in 3): float32  ND         — 改回 ND
 *   x2_scale (in 4): uint64   ND         — 改回 ND
 *   y        (out 0): float16 ND         — 改回 ND
 */

#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "graph/attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/ascend_string.h"
#include "graph/tensor.h"
#include "graph/gnode.h"
#include "reg_register.h"
#include "register/register_custom_pass.h"
#include <iostream>
#include <string>

static ge::Status FixQbmV4FormatPass(ge::GraphPtr& graph, ge::CustomPassContext& ctx)
{
    if (graph == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto nodes = graph->GetAllNodes();
    int fixed = 0;
    for (auto& node : nodes) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS || std::string(type.GetString()) != "QuantBatchMatmulV4") {
            continue;
        }
        // Fix inputs: x1(0)=ND, x2(1)=FRACTAL_Z, bias(2)=ND, x1_scale(3)=ND, x2_scale(4)=ND
        for (int idx = 0; idx <= 4; ++idx) {
            ge::TensorDesc desc;
            if (node.GetInputDesc(idx, desc) != ge::GRAPH_SUCCESS) {
                continue; // optional input may not exist (e.g. bias)
            }
            auto origin_shape = desc.GetOriginShape();
            ge::Format fmt = (idx == 1) ? ge::FORMAT_FRACTAL_Z : ge::FORMAT_ND;
            desc.SetFormat(fmt);
            desc.SetOriginFormat(fmt);
            desc.SetShape(origin_shape);
            desc.SetOriginShape(origin_shape);
            if (node.UpdateInputDesc(idx, desc) == ge::GRAPH_SUCCESS) {
                fixed++;
            }
        }
        // Fix output: y(0)=ND
        ge::TensorDesc y_desc;
        if (node.GetOutputDesc(0, y_desc) == ge::GRAPH_SUCCESS) {
            auto origin_shape = y_desc.GetOriginShape();
            y_desc.SetFormat(ge::FORMAT_ND);
            y_desc.SetOriginFormat(ge::FORMAT_ND);
            y_desc.SetShape(origin_shape);
            y_desc.SetOriginShape(origin_shape);
            if (node.UpdateOutputDesc(0, y_desc) == ge::GRAPH_SUCCESS) {
                fixed++;
            }
        }
    }
    if (fixed > 0) {
        std::cout << "[CustomPass] Fixed QuantBatchMatmulV4: x1/bias/x1_scale/x2_scale/y to ND, "
                  << "x2 to FRACTAL_Z (" << fixed << " descs)" << std::endl;
    }
    (void)ctx;
    return ge::GRAPH_SUCCESS;
}

static ge::PassReceiver kFixQbmV4FormatPassReceiver
    __attribute__((unused)) = ge::PassRegistrationData("FixQbmV4FormatPass")
                                  .CustomPassFn(FixQbmV4FormatPass)
                                  .Stage(ge::CustomPassStage::kAfterOriginGraphOptimize);
