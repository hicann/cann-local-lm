/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "executor_container.h"

#include "infra/log.h"
#include "infra/securestl.h"
#include "infra/assertion.h"

namespace hiai {
namespace {
std::mutex intanceMutex;
std::shared_ptr<ExecutorContainer> ptrInstance;
} // namespace

ExecutorContainer* ExecutorContainer::GetInstance()
{
    if (ptrInstance == nullptr) {
        std::lock_guard<std::mutex> lock(intanceMutex);
        if (ptrInstance == nullptr) {
            ptrInstance = make_shared_nothrow<ExecutorContainer>();
            HIAI_EXPECT_NOT_NULL_R(ptrInstance, nullptr);
        }
    }

    return ptrInstance.get();
}

ExecutorContainer::~ExecutorContainer()
{
    std::lock_guard<std::mutex> lock(containerMutex_);
    executorMap_.clear();
}

void ExecutorContainer::AddExecutor(uint64_t builtModelId, const std::shared_ptr<OH_NNExecutor>& executor)
{
    if (executor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(containerMutex_);
    auto it = executorMap_.find(builtModelId);
    if (it != executorMap_.end()) {
        CANN_LLM_LOGW("builtModel has been added "); // builtmodel和executor是一对一的关系
        return;
    }
    executorMap_.insert(std::pair<uint64_t, std::shared_ptr<OH_NNExecutor>>(builtModelId, executor));
}

void ExecutorContainer::RemoveExecutor(const std::shared_ptr<OH_NNExecutor>& executor)
{
    if (executor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(containerMutex_);
    for (auto it = executorMap_.begin(); it != executorMap_.end(); ++it) {
        if (it->second == executor) {
            executorMap_.erase(it);
            return;
        }
    }
    CANN_LLM_LOGW("No matching executor found.");
}

std::shared_ptr<OH_NNExecutor> ExecutorContainer::GetExecutor(uint64_t builtModelId)
{
    std::lock_guard<std::mutex> lock(containerMutex_);
    auto it = executorMap_.find(builtModelId);
    if (it != executorMap_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace hiai
