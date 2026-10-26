// 权重共享链路 executor 层（ModelExecutor::Load share_weights_model_id 透传 +
// ModelId() 暴露）UT——ut_backend_shareweights 的 executor 级延伸。
// 独立目标缘由：本文件经链接期符号覆盖 hiai::CreateModelManager / CreateBuiltModel
// / CreateNDTensorBuffer（cann_lm_engine 为 STATIC 库，ut 对象定义优先、库内
// 工厂 TU 不被抽取——各工厂 TU 仅定义单一符号，无连带抽出冲突）。不并入预存
// 损坏的 model_executor_ut（其旧 API 测试文件 host 编不过，见 engine/CMakeLists
// ut_linear_state_pool 处注释）；也不并入 ut_backend_shareweights——工厂覆盖为
// 二进制级符号替换，会劫持其 TC002 对真实 NDK impl 守卫的 host 验证。
// 覆盖四处契约（UT-TC-SW-004..007，编号接续 ut_backend_shareweights）：
//   UT-TC-SW-004：Load 第 5 参 share_weights_model_id>0 → mm Init 收到的
//     ModelInitOptions.shareWeightsModelId 原值透传（填入先于 Init——绑定在
//     compilation 构造 executor 前生效的 executor 侧前提）。
//   UT-TC-SW-005：ModelId() = Load 成功后 mm GetModelID() 返回值（pipeline 期
//     decode 执行器绑定的取值来源）；未加载 / Unload 后 = 0。
//   UT-TC-SW-006（回归锚点）：不传第 5 参（默认 0）→ Init 收到 0（不绑定，
//     与 UT-TC-SW-003 默认契约呼应——老 ROM 缺符号不影响默认加载路径）。
//   UT-TC-SW-007：Load 第 2 参 weight 原样透传 mm InitWeights（分离部署
//     decode 自身 weightDir 到达 mm 的链路证据，
//     docs/seperate_pd_weights.md §4.2——复用判定由底层自理，本层契约
//     = 路径按实参原样到达）。
// 局限：pipeline 接线（prefill ModelId → decode Load 第 5 参）与"绑定→executor
//   构造"先序需真实 .so / 设备验证，host 由本层透传契约 + 实现评审覆盖。
// 关联：model_executor.cpp（Load 绑定填入 / ModelId 存储）/ pipeline_executor.cpp
//   （双执行器接线）/ backend/api/types.h（shareWeightsModelId 契约注释）。
#include "core/component/model_executor/model_executor.h"
#include "model_manager/model_manager.h"
#include "model/built_model.h"
#include "tensor/nd_tensor_buffer.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// ── mock 基建（具名 namespace：外部链接类型用于工厂覆盖签名，避免内部链接
//    类型进外部函数签名的告警）──────────────────────────────────────────────
namespace sw_exec_mock {

// host 可用 INDTensorBuffer 最小实现（host 无 NDK so，真工厂返 nullptr）
class HostTensorBuffer final : public hiai::INDTensorBuffer {
public:
    explicit HostTensorBuffer(const hiai::NDTensorDesc& desc) : desc_(desc)
    {
        // 字节数 ≥ 4（GetData 非空约定；非正维按 1 估算——本 UT descs 无 0 维）
        std::size_t bytes = 4;
        for (int32_t d : desc_.dims) {
            bytes *= static_cast<std::size_t>(d > 0 ? d : 1);
        }
        data_.resize(bytes);
    }
    void* GetData() override { return data_.data(); }
    std::size_t GetSize() const override { return data_.size(); }
    const hiai::NDTensorDesc& GetTensorDesc() const override { return desc_; }
    void SetCacheStatus(uint8_t status) override { cache_status_ = status; }
    uint8_t GetCacheStatus() override { return cache_status_; }

private:
    hiai::NDTensorDesc desc_;
    std::vector<uint8_t> data_;
    uint8_t cache_status_ = 0;
};

// Mock IBuiltModel（当前 backend API）：buffer 态 Restore 恒过；非 KV/状态工位
// descs（避开池绑定路径——本 UT 不挂池，与 pipeline 构造期空依赖形态一致）
class MockBuiltModel final : public hiai::IBuiltModel {
public:
    hiai::Status RestoreFromBuffer(const std::shared_ptr<hiai::IBuffer>& buffer) override
    {
        return (buffer && buffer->GetData()) ? hiai::SUCCESS : hiai::FAILURE;
    }
    hiai::Status RestoreFromFile(const char* file) override
    {
        return (file != nullptr && *file != '\0') ? hiai::SUCCESS : hiai::FAILURE;
    }
    void SetName(const std::string&) override {}
    std::vector<hiai::NDTensorDesc> GetInputTensorDescs() const override
    {
        hiai::NDTensorDesc d;
        d.tensorName = "input_embed";
        d.dataType = hiai::DataType::FLOAT16;
        d.dims = {1, 1, 8};
        return {d};
    }
    std::vector<hiai::NDTensorDesc> GetOutputTensorDescs() const override
    {
        hiai::NDTensorDesc d;
        d.dataType = hiai::DataType::FLOAT32;
        d.dims = {1, 1};
        return {d};
    }
    void SetCustomData(const hiai::CustomModelData&) override {}
    const hiai::CustomModelData& GetCustomData() override { return custom_data_; }

private:
    hiai::CustomModelData custom_data_;
};

// Mock IModelManager（当前 backend API）。断言通道类级静态——Load 内经工厂
// 创建，测试无法预置实例态。
class MockModelManager final : public hiai::IModelManager {
public:
    static hiai::ModelInitOptions last_init_opts; // Init 收到的 options 快照
    static uint32_t model_id;                     // GetModelID 返回值（默认 1）
    static int init_count;
    static std::string last_init_weights_dir; // InitWeights 收到的 weightDir 快照

    hiai::Status Init(const hiai::ModelInitOptions& options, const std::shared_ptr<hiai::IBuiltModel>& builtModel,
                      const std::shared_ptr<hiai::IModelManagerListener>&) override
    {
        ++init_count;
        last_init_opts = options;
        built_ = builtModel;
        return hiai::SUCCESS;
    }
    hiai::Status InitWeights(const std::string& weightDir) override
    {
        last_init_weights_dir = weightDir; // UT-TC-SW-007 断言通道（类级静态，同上口径）
        return weightDir.empty() ? hiai::FAILURE : hiai::SUCCESS;
    }
    hiai::Status DeinitWeights() override { return hiai::SUCCESS; }
    hiai::Status GetWeightBuffer(const std::string&, std::shared_ptr<hiai::IBuffer>&) override
    {
        return hiai::UNSUPPORTED;
    }
    hiai::Status FlushWeight(const std::string&, size_t, size_t) override { return hiai::UNSUPPORTED; }
    hiai::Status UpdateWeight(const std::string&, size_t, size_t, const void*) override { return hiai::UNSUPPORTED; }
    uint32_t GetModelID() override { return model_id; }
    hiai::Status Run(const std::vector<std::shared_ptr<hiai::INDTensorBuffer>>&,
                     std::vector<std::shared_ptr<hiai::INDTensorBuffer>>&) override
    {
        return hiai::SUCCESS;
    }
    hiai::Status ReleaseMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    hiai::Status RecoverMemory(hiai::MemoryMode) override { return hiai::SUCCESS; }
    void DeInit() override { built_.reset(); }

private:
    std::shared_ptr<hiai::IBuiltModel> built_;
};

hiai::ModelInitOptions MockModelManager::last_init_opts;
uint32_t MockModelManager::model_id = 1;
int MockModelManager::init_count = 0;
std::string MockModelManager::last_init_weights_dir;

void ResetMocks()
{
    MockModelManager::last_init_opts = hiai::ModelInitOptions{};
    MockModelManager::model_id = 1;
    MockModelManager::init_count = 0;
    MockModelManager::last_init_weights_dir.clear();
}

// 权重路径（图与权重分离；空路径 = Load 跳过 InitWeights）
constexpr const char* kWeight = "/mock/weights";

// 合法 omc buffer 源（buffer 态优先，D-15）
cann_lm::engine::ArtifactSource MakeOmc()
{
    static char omc_buf[8];
    cann_lm::engine::ArtifactSource s;
    s.data = cann_lm::engine::Buffer{omc_buf, sizeof(omc_buf)};
    return s;
}

// 容量域合法配置（TC_106 前置）；hidden_size/vocab_align_size 留 0：
// desc 修正跳过（无已知名称命中），dims 保持 mock 原值
cann_lm::engine::ModelConfig MakeConfig()
{
    cann_lm::engine::ModelConfig c;
    c.kv_cache_max_len = 16;
    c.num_hidden_layers = 2;
    return c;
}

} // namespace sw_exec_mock

// ── 工厂覆盖（hiai 命名空间；链接期符号优先于库内实现）──────────────────────
namespace hiai {
std::shared_ptr<IModelManager> CreateModelManager() { return std::make_shared<sw_exec_mock::MockModelManager>(); }
std::shared_ptr<IBuiltModel> CreateBuiltModel() { return std::make_shared<sw_exec_mock::MockBuiltModel>(); }
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& tensorDesc)
{
    return std::make_shared<sw_exec_mock::HostTensorBuffer>(tensorDesc);
}
} // namespace hiai

namespace {

// UT-TC-SW-004: Load 透传——share_weights_model_id>0 → mm Init 收到原值
TEST(ShareWeightsExecutorTest, LoadPassesShareWeightsIdToInit_TC004)
{
    sw_exec_mock::ResetMocks();
    sw_exec_mock::MockModelManager::model_id = 0xC0FFEE; // 区分值（与透传值互异，防串扰误判）
    cann_lm::engine::ModelExecutor exec(nullptr, nullptr, nullptr);
    ASSERT_TRUE(exec.Load(sw_exec_mock::MakeOmc(), sw_exec_mock::kWeight, sw_exec_mock::MakeConfig(), /*seq_len=*/1,
                          /*share_weights_model_id=*/77u)
                    .ok());
    EXPECT_EQ(sw_exec_mock::MockModelManager::init_count, 1);
    EXPECT_EQ(sw_exec_mock::MockModelManager::last_init_opts.shareWeightsModelId, 77u);
}

// UT-TC-SW-005: ModelId() 暴露——Load 成功后 = GetModelID() 值；未加载/Unload = 0
TEST(ShareWeightsExecutorTest, ModelIdReflectsGetModelID_TC005)
{
    sw_exec_mock::ResetMocks();
    sw_exec_mock::MockModelManager::model_id = 0xC0FFEE;
    cann_lm::engine::ModelExecutor exec(nullptr, nullptr, nullptr);
    EXPECT_EQ(exec.ModelId(), 0u); // 未加载 = 0 = 不绑定语义
    ASSERT_TRUE(exec.Load(sw_exec_mock::MakeOmc(), sw_exec_mock::kWeight, sw_exec_mock::MakeConfig(), 1).ok());
    EXPECT_EQ(exec.ModelId(), 0xC0FFEEu);
    ASSERT_TRUE(exec.Unload().ok());
    EXPECT_EQ(exec.ModelId(), 0u);
}

// UT-TC-SW-006（回归锚点）: 默认 0 不绑定——不传第 5 参 → Init 收到 0
TEST(ShareWeightsExecutorTest, LoadDefaultZeroNoBind_TC006)
{
    sw_exec_mock::ResetMocks();
    cann_lm::engine::ModelExecutor exec(nullptr, nullptr, nullptr);
    ASSERT_TRUE(exec.Load(sw_exec_mock::MakeOmc(), sw_exec_mock::kWeight, sw_exec_mock::MakeConfig(), 1).ok());
    EXPECT_EQ(sw_exec_mock::MockModelManager::last_init_opts.shareWeightsModelId, 0u);
}

// UT-TC-SW-007: weight 实参透传——Load 第 2 参原样到达 mm InitWeights。
// 分离部署（docs/seperate_pd_weights.md §4.2）的 executor 级前提：decode
// 绑定 share_weights_model_id 的同时按自身 weightDir 发起 InitWeights——
// 路径按实参原样到达 mm 即本层契约完备（复用多少由底层同源规则决定）。
TEST(ShareWeightsExecutorTest, LoadPassesWeightDirToInitWeights_TC007)
{
    sw_exec_mock::ResetMocks();
    constexpr const char* kDecodeWeight = "/mock/weights_decode"; // 与 kWeight 互异
    cann_lm::engine::ModelExecutor exec(nullptr, nullptr, nullptr);
    ASSERT_TRUE(exec.Load(sw_exec_mock::MakeOmc(), kDecodeWeight, sw_exec_mock::MakeConfig(), /*seq_len=*/1,
                          /*share_weights_model_id=*/0xC0FFEEu)
                    .ok());
    EXPECT_EQ(sw_exec_mock::MockModelManager::last_init_weights_dir, kDecodeWeight);
}

} // namespace
