/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
 */
/*
 * backend 唯一的 type 头：合并自 error_types.h / util/base_types.h / nd_tensor_desc.h /
 * model_manager_types.h / hiai_execute_option_types.h /
 * common/types.h(模型文件头部分) / model_type_util.h / compatible/HiAiModelBuilderType.h。
 * 仅保留移植链路实际使用的内容；结构体字段布局与原定义一致。
 */
#ifndef FRAMEWORK_TYPES_H_
#define FRAMEWORK_TYPES_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hiai {
// ── 状态码（原 base/error_types.h）──
using Status = uint32_t;

constexpr Status SUCCESS = 0;
constexpr Status FAILURE = 1;
constexpr Status UNINITIALIZED = 2;
constexpr Status INVALID_PARAM = 3;
constexpr Status TIMEOUT = 4;
constexpr Status UNSUPPORTED = 5;
constexpr Status MEMORY_EXCEPTION = 6;
constexpr Status INVALID_API = 7;
constexpr Status INVALID_POINTER = 8;
constexpr Status CALC_EXCEPTION = 9;
constexpr Status FILE_NOT_EXIST = 10;
constexpr Status COMM_EXCEPTION = 11;
constexpr Status DATA_OVERFLOW = 12;
constexpr Status CONNECT_EXCEPTION = 19;

// ── 基础数据类型（原 util/base_types.h）──
enum class DataType { UINT8, FLOAT32, FLOAT16, INT32, INT8, INT16, BOOL, INT64, UINT32, DOUBLE, RESERVED };

enum class Format {
    NCHW = 0, /* NCHW */
    NHWC,     /* NHWC */
    ND,       /* Nd Tensor */
    RESERVED
};

// ── 共享内存句柄（原 util/base_types.h）──
// 保留原因：文件 2 已注释保留的 CreateNDTensorBuffer(NativeHandle) 重载的参数类型。
struct NativeHandle {
    int fd;
    int size;
    int offset;
};

// ── Tensor 描述（原 tensor/nd_tensor_desc.h）──
struct NDTensorDesc {
    std::vector<int32_t> dims;
    DataType dataType = DataType::FLOAT32;
    Format format = Format::NCHW;
    std::string tensorName = ""; // 该字段不为空字符串时，表示获取tensorName成功
};

// ── 模型管理选项（原 model_manager/model_manager_types.h）──
enum class PerfMode {
    UNSET = 0,
    LOW = 1,
    MIDDLE,
    HIGH,
    EXTREME_HIGH,
    HIGH_COMPUTE_UNIT = 103,
};

enum class ModelCacheStrategy {
    OFF = 0,
    ON = 1,
};

struct ModelInitOptions {
    PerfMode perfMode = PerfMode::MIDDLE;
    PerfMode bandMode = PerfMode::UNSET; /* only support [UNSET, HIGH] mode */
    ModelCacheStrategy modelCacheStrategy = ModelCacheStrategy::OFF;
    // 权重共享绑定：0=不绑定（默认，行为不变）；>0=源模型 GetModelID() 返回值。
    // 绑定后本模型按自身 weightDir InitWeights（与源模型 weightDir 可不同，
    // docs/seperate_pd_weights.md §2）；哪些权值复用源模型已加载份由底层同源
    // 规则决定（部分共享），上层无需感知复用细节。
    uint32_t shareWeightsModelId = 0;
};

enum class MemoryMode {
    ALL = 0,
    WEIGHT_ALL = 1,
    WEIGHT_RES = 2, // resident只恢复常驻内存
    FEATURE_MAP = 3,
};

// 保留原因：文件 4 已注释保留的 IModelManagerListener::OnRunDone 签名的参数类型。
// 未使用：引擎链路不用 hiai::Context（tensor_manager.h 与 engine 自有 Context 冲突）。
class Context {
public:
    std::string GetValue(const std::string& key) const;
    void SetValue(const std::string& key, const std::string& value);
    void RemoveValue(const std::string& key);
    const std::map<std::string, std::string>& GetContent() const;

private:
    std::map<std::string, std::string> paras_;
};

inline std::string Context::GetValue(const std::string& key) const
{
    auto it = paras_.find(key);
    return it != paras_.end() ? it->second : "";
}

inline void Context::SetValue(const std::string& key, const std::string& value) { paras_[key] = value; }

inline void Context::RemoveValue(const std::string& key) { paras_.erase(key); }

inline const std::map<std::string, std::string>& Context::GetContent() const { return paras_; }

// ── NDK 执行选项 C 类型（原 c/hcl/hiai_execute_option_types.h）──
extern "C" {
typedef enum {
    /** 不使用，默认采用中级性能模式 */
    HIAI_PERFMODE_UNSET = 0,
    /** 低级性能模式 */
    HIAI_PERFMODE_LOW,
    /** 中级性能模式 */
    HIAI_PERFMODE_NORMAL,
    /** 高级性能模式 */
    HIAI_PERFMODE_HIGH,
    /** 极高性能模式 */
    HIAI_PERFMODE_EXTREME,
    /** 介于低级和中级之间的性能模式，不保证跨平台兼容，生态不开放*/
    HIAI_PERFMODE_MID_LOW_EXTEND = 51,
    /** 仅计算单元高性能模式，相比HIGH模式，性能会有微小损失但功耗有优化提升 */
    HIAI_PERFMODE_HIGH_COMPUTE_UNIT = 103,
    /** 仅计算单元中等性能模式 */
    HIAI_PERFMODE_MIDDLE_BAND_NORMAL_COMPUTE_UNIT = 202,
} HIAI_PerfMode;

typedef enum {
    HIAI_MODELCACHESTRATEGY_OFF = 0,
    HIAI_MODELCACHESTRATEGY_ON = 1,
} HIAI_ModelCacheStrategy;
}

// ── 模型文件头（原 common/types.h 保留部分）──
/*
 * @brief 模型名字长度
 */
const int MODEL_NAME_LENGTH = 32;

/*
 * @brief 用户自定义信息长度
 */
const int USER_DEFINE_INFO_LENGTH = 32;

/*
 * @brief 模型文件签名长度
 */
const int MODEL_FILE_CHECKSUM_LENGTH = 64;

/*
 * @brief 模型使用的托管算子库信息长度
 */
const int CUSTOM_LIB_INFO_LENGTH = 16;
/*
 * @brief 模型文件头保留字段长度
 */
const int MODEL_FILE_RESERVED_LENGTH = 52;

/*
 * @brief 模型文件加密类型
 */
enum ModelEncryptType {
    UNENCRYPTED, // 不加密
    ENCRYPTED    // 加密
};

/*
 * @brief 校验签名
 */
enum ModelCheckType {
    CHECK,  // 校验签名
    UNCHECK // 不校验
};

/*
 * @brief 模型文件魔数
 */
const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49; /* 魔数DOMI */

/*
 * @brief 模型头长度（原为 extern 声明且仓内无定义，按注释固定值 256 收敛为定义）
 */
constexpr uint32_t MODEL_FILE_HEAD_LEN = 256;

/*
 * @brief 平台版本长度
 */
const int PLATFORM_VERSION_LEN = 20;

constexpr uint32_t MODEL_VERSION = 0x10000000; /* < 模型版本 1.0 */

/*
 * @brief 模型文件文件头定义
 */
struct ModelFileHeader {
    uint32_t magic = MODEL_FILE_MAGIC_NUM;              /* 魔数DOMI */
    uint32_t headsize = MODEL_FILE_HEAD_LEN;            /* 模型头长度，固定256 */
    uint32_t version = MODEL_VERSION;                   /* *< 版本1.0 */
    uint8_t checksum[MODEL_FILE_CHECKSUM_LENGTH] = {0}; /* 签名 */
    uint32_t length = 0;                                /* 密文长度, 非加密模型为明文长度 */
    uint8_t isEncrypt = ModelEncryptType::UNENCRYPTED;  /* 是否加密 0:不加密，1:加密 */
    uint8_t isChecksum = ModelCheckType::CHECK;         /* *< 是否检查checksum */
    uint8_t modeltype = 0; /* 0：IR模型 1：OM标准模型 2: OM Tiny模型 3:异构分段编译模型 */
    uint8_t genmode = 0;   /*  0：离线生成 1：在线生成 */
    uint8_t name[MODEL_NAME_LENGTH] = {0};                 /* 模型名字32长度 */
    uint32_t ops = 0;                                      /* 算力，单位Kops */
    uint8_t userdefineinfo[USER_DEFINE_INFO_LENGTH] = {0}; /* 用户自定义信息 32长度 */
    uint32_t omIrVersion = 0;
    uint8_t platformVersion[PLATFORM_VERSION_LEN] = {0};
    uint8_t platformType = {0};
    uint8_t heterTuningFlag = {0}; /* 模型是否做过在线异构调优 */
    uint32_t customDataLen = 0;
    uint32_t multiShapeGears = 0;
    uint8_t customLibInfo[CUSTOM_LIB_INFO_LENGTH] = {0}; /* 模型使用的托管算子库信息(hash值及时间戳) 16长度 */
    uint8_t reserved[MODEL_FILE_RESERVED_LENGTH] = {0}; /* 保留字段 52 长度 */
};

enum ModelType {
    IR_GRAPH_MODEL = 0,      /* 老模型 标准OM模型（异构前） */
    OM_STANDARD_MODEL,       /* 之前没使用地方 */
    IGS_MODEL,               /* 异构前的TINY模型 */
    STANDARD_IR_GRAPH_MODEL, /* 异构之后的标准IR模型 */
    HCS_PARTITION_MODEL,     /* 异构之后的分段D模型 */
    ISPNN_MODEL,             /* ISPNN模型 */
    SPECIAL_3RD_MODEL,
    IR_API_GRAPH_MODEL, /* IR API构造经序列化后的模型, 无模型头 */
    UTINY_MODEL,        /* Pha模型，枚举值修改后，需要修改sensorhub侧代码 */
    MULTI_SHAPE_MODEL,
    ENUM_SHAPE_MODEL,            /* 枚举Shape模型 */
    SINGLE_IP_MODEL,             /* 单IP模型（SCL简化格式） */
    MULTI_SHAPE_SINGLE_IP_MODEL, /* 枚举Shape模型（SCL简化格式） */
};

} // namespace hiai
#endif // FRAMEWORK_TYPES_H_
