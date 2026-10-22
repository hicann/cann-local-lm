/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef HIAI_API_BUILT_MODEL_H
#define HIAI_API_BUILT_MODEL_H

#include <string>
#include "tensor/nd_tensor_buffer.h"
#include "types.h"

namespace hiai {
struct CustomModelData {
    std::string type;
    std::string value;
};

class IBuiltModel {
public:
    virtual ~IBuiltModel() = default;

    virtual Status RestoreFromBuffer(const std::shared_ptr<IBuffer>& buffer) = 0;

    /**
     * @brief Restore model from file
     *
     * @param [in] file The filename of the model, including the path
     * The use path is legal and has read access, and the model name is reasonable without malicious characters.
     * Path + file name such as /data/local/tmp/name.om or current working directory + file name such as ./name.om
     * or just the filename without the path eg name.om
     * @return Return success or other value.
    */
    virtual Status RestoreFromFile(const char* file) = 0;

    virtual void SetName(const std::string& name) = 0;

    virtual std::vector<NDTensorDesc> GetInputTensorDescs() const = 0;
    virtual std::vector<NDTensorDesc> GetOutputTensorDescs() const = 0;

    virtual void SetCustomData(const CustomModelData& customModelData) = 0;
    virtual const CustomModelData& GetCustomData() = 0;
};

std::shared_ptr<IBuiltModel> CreateBuiltModel();
} // namespace hiai
#endif // HIAI_API_BUILT_MODEL_H