/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef FRAMEWORK_MODEL_MANAGER_GENERAL_MODEL_MANAGER_CORE_EXECUTOR_CONTAINER_H
#define FRAMEWORK_MODEL_MANAGER_GENERAL_MODEL_MANAGER_CORE_EXECUTOR_CONTAINER_H

#include <map>
#include <memory>
#include <mutex>

#include "neural_network_runtime/neural_network_runtime_type.h"

namespace hiai {

class ExecutorContainer {
public:
    static ExecutorContainer* GetInstance();

    void AddExecutor(uint64_t builtModelId, const std::shared_ptr<OH_NNExecutor>& executor);

    void RemoveExecutor(const std::shared_ptr<OH_NNExecutor>& executor);

    std::shared_ptr<OH_NNExecutor> GetExecutor(uint64_t builtModelId);

    ExecutorContainer() = default;

    ~ExecutorContainer();

    ExecutorContainer(const ExecutorContainer& container) = delete;

    ExecutorContainer& operator=(const ExecutorContainer& container) = delete;

private:
    std::mutex containerMutex_;

    std::map<uint64_t, std::shared_ptr<OH_NNExecutor>> executorMap_;
};
} // namespace hiai
#endif // FRAMEWORK_MODEL_MANAGER_GENERAL_MODEL_MANAGER_CORE_EXECUTOR_CONTAINER_H
