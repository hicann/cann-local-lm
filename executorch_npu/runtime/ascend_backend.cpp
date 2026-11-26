/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <nlohmann/json.hpp>

#include "executorch_npu/manifest.h"
#include "executorch_npu/sha256.h"
#include "executorch_npu/state.h"

namespace executorch_npu {
namespace {

using Json = nlohmann::json;

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;

constexpr const char *kBackendName = "AscendBackend";
constexpr size_t kWeightAlignment = 32;

bool AlignUp(size_t value, size_t alignment, size_t *output) {
  if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return false;
  }
  *output = ((value + alignment - 1) / alignment) * alignment;
  return true;
}

bool CheckAcl(aclError status, const char *operation) {
  (void)operation;
  if (status == ACL_SUCCESS) {
    return true;
  }
  const char *recent_error = aclGetRecentErrMsg();
  (void)recent_error;
  ET_LOG(Error, "%s failed with ACL error %d: %s", operation, status,
         recent_error == nullptr ? "<no ACL detail>" : recent_error);
  return false;
}

bool EnsureAclInitialized() {
  static std::once_flag once;
  static aclError initialization_status = ACL_ERROR_INVALID_PARAM;
  std::call_once(once, []() {
    initialization_status = aclInit(nullptr);
    if (initialization_status == ACL_ERROR_REPEAT_INITIALIZE) {
      initialization_status = ACL_SUCCESS;
    }
  });
  return CheckAcl(initialization_status, "aclInit");
}

std::string CompileSpecValue(ArrayRef<CompileSpec> compile_specs, const char *expected_key) {
  for (const CompileSpec &spec : compile_specs) {
    if (spec.key != nullptr && std::strcmp(spec.key, expected_key) == 0 && spec.value.buffer != nullptr) {
      return std::string(static_cast<const char *>(spec.value.buffer), spec.value.nbytes);
    }
  }
  return {};
}

bool CompileOptionValue(ArrayRef<CompileSpec> compile_specs, const char *key, std::string *value) {
  const std::string direct = CompileSpecValue(compile_specs, key);
  if (!direct.empty()) {
    *value = direct;
    return true;
  }
  const std::string options = CompileSpecValue(compile_specs, "ascend_compile_options");
  if (options.empty()) {
    return false;
  }
  try {
    const Json parsed = Json::parse(options);
    const auto entry = parsed.find(key);
    if (entry == parsed.end()) {
      return false;
    }
    if (entry->is_string()) {
      *value = entry->get<std::string>();
      return true;
    }
    if (entry->is_number_integer()) {
      *value = std::to_string(entry->get<int64_t>());
      return true;
    }
  } catch (const Json::exception &) {
    return false;
  }
  return false;
}

bool ParseNonNegativeSize(const std::string &value, size_t *output) {
  if (value.empty() ||
      !std::all_of(value.begin(), value.end(), [](char character) { return character >= '0' && character <= '9'; })) {
    return false;
  }
  try {
    const unsigned long long parsed = std::stoull(value);
    if (parsed > std::numeric_limits<size_t>::max()) {
      return false;
    }
    *output = static_cast<size_t>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool AclSocIsCompatible(const Manifest &manifest) {
  const char *soc_name = aclrtGetSocName();
  return soc_name != nullptr && manifest.acl_soc_name == soc_name;
}

struct DeviceWeight {
  DeviceWeight(int device_id_value, void *data_value, size_t size_value)
      : device_id(device_id_value), data(data_value), size(size_value) {}

  ~DeviceWeight() {
    if (data != nullptr) {
      aclrtSetDevice(device_id);
      aclrtFree(data);
    }
  }

  DeviceWeight(const DeviceWeight &) = delete;
  DeviceWeight &operator=(const DeviceWeight &) = delete;

  int device_id;
  void *data;
  size_t size;
};

struct WeightCacheEntry {
  std::mutex mutex;
  std::condition_variable ready;
  bool loading = false;
  std::weak_ptr<DeviceWeight> value;
};

class SharedWeightCache {
 public:
  static SharedWeightCache &Instance() {
    static SharedWeightCache cache;
    return cache;
  }
  // 外置权重上传device，同一张NPU 同一个内容仅上传一次
  std::shared_ptr<DeviceWeight> Acquire(int device_id, const WeightGroup &group, const NamedDataMap &named_data_map,
                                        std::string *error) {
    // 为权重设置唯一key，并插入全局map中，用于共享
    const std::string cache_key =
        std::to_string(device_id) + ":" + group.key + ":" + group.sha256 + ":" + std::to_string(group.nbytes);
    std::shared_ptr<WeightCacheEntry> entry;
    {
      std::lock_guard<std::mutex> lock(entries_mutex_);
      auto &slot = entries_[cache_key];
      if (!slot) {
        slot = std::make_shared<WeightCacheEntry>();
      }
      entry = slot;
    }
    // 线程中已经成功加载权重-->所有使用者都可以直接使用
    // 线程中未加载权重-->判断是否有其他加载者-->有，成为等待者
    // 线程中未加载权重-->判断是否有其他加载者-->无其他加载者，则加载，加载后通知所有等待者直接使用
    for (;;) {
      std::unique_lock<std::mutex> lock(entry->mutex);
      if (auto existing = entry->value.lock()) {
        return existing;
      }
      if (entry->loading) {
        entry->ready.wait(lock, [&]() { return !entry->loading; });
        continue;
      }
      entry->loading = true;
      lock.unlock();

      std::shared_ptr<DeviceWeight> loaded = Load(device_id, group, named_data_map, error);

      lock.lock();
      if (loaded) {
        entry->value = loaded;
      }
      entry->loading = false;
      lock.unlock();
      entry->ready.notify_all();
      return loaded;
    }
  }

 private:
  std::shared_ptr<DeviceWeight> Load(int device_id, const WeightGroup &group, const NamedDataMap &named_data_map,
                                     std::string *error) {
    // 读取并校验权重
    Result<FreeableBuffer> data_result = named_data_map.get_data(group.key);
    if (!data_result.ok()) {
      *error = "PTD named data is missing: " + group.key;
      return nullptr;
    }
    FreeableBuffer host_data(std::move(data_result.get()));
    if (host_data.size() != group.nbytes) {
      *error = "PTD named data size mismatch: " + group.key;
      return nullptr;
    }
    std::string digest;
    if (!Sha256(host_data.data(), host_data.size(), &digest, error) || digest != group.sha256) {
      *error = "PTD named data SHA-256 mismatch: " + group.key;
      return nullptr;
    }
    // 指定device
    if (!CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice")) {
      *error = "failed to select Ascend device";
      return nullptr;
    }
    // 分配空间
    size_t allocation_size = 0;
    if (!AlignUp(group.nbytes, kWeightAlignment, &allocation_size)) {
      *error = "external weight allocation size overflows";
      return nullptr;
    }
    void *device_data = nullptr;
    if (!CheckAcl(aclrtMallocAlign32(&device_data, allocation_size, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMallocAlign32(weight)")) {
      *error = "failed to allocate external weight device memory";
      return nullptr;
    }
    if (reinterpret_cast<uintptr_t>(device_data) % kWeightAlignment != 0) {
      aclrtFree(device_data);
      *error = "external weight device address is misaligned";
      return nullptr;
    }
    // 分配的内存空间清零，然后填充权重数据
    if (!CheckAcl(aclrtMemset(device_data, allocation_size, 0, allocation_size), "aclrtMemset(weight padding)") ||
        !CheckAcl(
            aclrtMemcpy(device_data, allocation_size, host_data.data(), host_data.size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(weight H2D)")) {
      aclrtFree(device_data);
      *error = "failed to copy external weight to device";
      return nullptr;
    }
    return std::make_shared<DeviceWeight>(device_id, device_data, allocation_size);
  }

  std::mutex entries_mutex_;
  std::unordered_map<std::string, std::shared_ptr<WeightCacheEntry>> entries_;
};

struct StateTensorCopy {
  std::vector<uint8_t> data;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct StateDefinition {
  std::string dtype;
  std::vector<int64_t> shape;
  std::string reset_policy;
  std::string host_visibility;
  std::string reset_owner;
};

struct CachedState {
  explicit CachedState(int device_id_value) : device_id(device_id_value) {}

  ~CachedState() {
    aclrtSetDevice(device_id);
    if (current != nullptr) {
      aclrtFree(current);
    }
    if (next != nullptr) {
      aclrtFree(next);
    }
  }

  CachedState(const CachedState &) = delete;
  CachedState &operator=(const CachedState &) = delete;

  int device_id;
  void *current = nullptr;
  void *next = nullptr;
  size_t nbytes = 0;
  bool reset_done = false;
};

struct StateNamespace {
  explicit StateNamespace(int device_id_value) : device_id(device_id_value) {}

  int device_id;
  bool released = false;
  std::mutex execution_mutex;
  std::unordered_map<std::string, StateDefinition> definitions;
  std::unordered_map<std::string, StateTensorCopy> initializers;
  std::unordered_map<std::string, std::shared_ptr<CachedState>> states;
};

bool StateTensorIsValid(const StateTensorData &tensor) {
  if (tensor.data == nullptr || tensor.nbytes == 0 || tensor.dtype == nullptr || tensor.dtype[0] == '\0' ||
      (tensor.rank != 0 && tensor.shape == nullptr)) {
    return false;
  }
  if (tensor.rank == 0) {
    return true;
  }
  return std::all_of(tensor.shape, tensor.shape + tensor.rank, [](int64_t dimension) { return dimension >= 0; });
}

StateTensorCopy CopyStateTensor(const StateTensorData &tensor) {
  StateTensorCopy copy;
  const auto *bytes = static_cast<const uint8_t *>(tensor.data);
  copy.data.assign(bytes, bytes + tensor.nbytes);
  copy.dtype = tensor.dtype;
  if (tensor.rank != 0) {
    copy.shape.assign(tensor.shape, tensor.shape + tensor.rank);
  }
  return copy;
}

bool StateMetadataMatches(const StateDefinition &definition, const StateTensorCopy &tensor) {
  return definition.dtype == tensor.dtype && definition.shape == tensor.shape;
}

bool MakeZeroStateInitializer(const StateBinding &binding, StateTensorCopy *initializer) {
  size_t nbytes = 0;
  if (!StateZeroInitializerBytes(binding, &nbytes)) {
    return false;
  }
  initializer->dtype = binding.dtype;
  initializer->shape = binding.shape;
  initializer->data.assign(nbytes, 0);
  return true;
}

bool StateDefinitionMatches(const StateDefinition &left, const StateBinding &right) {
  return left.dtype == right.dtype && left.shape == right.shape && left.reset_policy == right.reset_policy &&
         left.host_visibility == right.host_visibility;
}

std::string StateOwnerIdentity(const Manifest &manifest) {
  return manifest.method + "#" + std::to_string(manifest.delegate_index);
}

class DeviceStateCache {
 public:
  static DeviceStateCache &Instance() {
    static DeviceStateCache cache;
    return cache;
  }

  StateStatus SetInitializer(int device_id, const std::string &state_namespace, const std::string &state_id,
                             const StateTensorData &initializer) {
    if (device_id < 0 || state_namespace.empty() || state_id.empty() || !StateTensorIsValid(initializer)) {
      return StateStatus::kInvalidArgument;
    }
    const StateTensorCopy copy = CopyStateTensor(initializer);
    for (;;) {
      std::shared_ptr<StateNamespace> entry = GetOrCreate(device_id, state_namespace);
      std::lock_guard<std::mutex> lock(entry->execution_mutex);
      if (entry->released) {
        continue;
      }
      if (entry->states.count(state_id) != 0 || entry->initializers.count(state_id) != 0) {
        return StateStatus::kInvalidState;
      }
      const auto definition = entry->definitions.find(state_id);
      if (definition != entry->definitions.end() && !StateMetadataMatches(definition->second, copy)) {
        return StateStatus::kInvalidArgument;
      }
      entry->initializers.emplace(state_id, copy);
      return StateStatus::kOk;
    }
  }

  StateStatus Reset(int device_id, const std::string &state_namespace, const std::string &state_id,
                    const StateTensorData &source) {
    if (device_id < 0 || state_namespace.empty() || state_id.empty() || !StateTensorIsValid(source)) {
      return StateStatus::kInvalidArgument;
    }
    const StateTensorCopy copy = CopyStateTensor(source);
    std::shared_ptr<StateNamespace> entry = Find(device_id, state_namespace);
    if (!entry) {
      return StateStatus::kNotFound;
    }
    std::lock_guard<std::mutex> lock(entry->execution_mutex);
    if (entry->released) {
      return StateStatus::kNotFound;
    }
    const auto definition = entry->definitions.find(state_id);
    if (definition == entry->definitions.end()) {
      return StateStatus::kNotFound;
    }
    if (definition->second.reset_policy != "explicit") {
      return StateStatus::kInvalidState;
    }
    if (!StateMetadataMatches(definition->second, copy)) {
      return StateStatus::kInvalidArgument;
    }
    std::string error;
    std::shared_ptr<CachedState> state = GetOrCreateStateLocked(*entry, state_id, copy, &error);
    if (!state || !CopyToStateBuffers(*state, copy.data.data(), copy.data.size(), &error)) {
      return StateStatus::kResourceError;
    }
    return StateStatus::kOk;
  }

  StateStatus Release(int device_id, const std::string &state_namespace) {
    if (device_id < 0 || state_namespace.empty()) {
      return StateStatus::kInvalidArgument;
    }
    const std::string key = NamespaceKey(device_id, state_namespace);
    std::shared_ptr<StateNamespace> entry;
    {
      std::lock_guard<std::mutex> lock(namespaces_mutex_);
      const auto iterator = namespaces_.find(key);
      if (iterator == namespaces_.end()) {
        return StateStatus::kNotFound;
      }
      entry = iterator->second;
    }
    std::unique_lock<std::mutex> execution_lock(entry->execution_mutex);
    {
      std::lock_guard<std::mutex> lock(namespaces_mutex_);
      const auto iterator = namespaces_.find(key);
      if (iterator == namespaces_.end() || iterator->second != entry) {
        return StateStatus::kNotFound;
      }
      entry->released = true;
      namespaces_.erase(iterator);
    }
    entry->states.clear();
    entry->initializers.clear();
    entry->definitions.clear();
    return StateStatus::kOk;
  }

  std::shared_ptr<StateNamespace> GetOrCreate(int device_id, const std::string &state_namespace) {
    const std::string key = NamespaceKey(device_id, state_namespace);
    std::lock_guard<std::mutex> lock(namespaces_mutex_);
    auto &entry = namespaces_[key];
    if (!entry) {
      entry = std::make_shared<StateNamespace>(device_id);
    }
    return entry;
  }

  bool RegisterBindingsLocked(StateNamespace &entry, const Manifest &manifest, std::string *error) {
    const std::string owner = StateOwnerIdentity(manifest);
    for (const StateBinding &binding : manifest.state_bindings) {
      auto [iterator, inserted] = entry.definitions.emplace(
          binding.state_id,
          StateDefinition{binding.dtype, binding.shape, binding.reset_policy, binding.host_visibility, {}});
      StateDefinition &definition = iterator->second;
      if (!inserted && !StateDefinitionMatches(definition, binding)) {
        *error = "state metadata differs across delegates: " + binding.state_id;
        return false;
      }
      if (binding.reset_owner) {
        if (!definition.reset_owner.empty() && definition.reset_owner != owner) {
          *error = "state has more than one reset owner: " + binding.state_id;
          return false;
        }
        definition.reset_owner = owner;
      }
      const auto initializer = entry.initializers.find(binding.state_id);
      if (initializer != entry.initializers.end() && !StateMetadataMatches(definition, initializer->second)) {
        *error = "state initializer metadata mismatch: " + binding.state_id;
        return false;
      }
    }
    return true;
  }

  bool RegisterBindings(int device_id, const std::string &state_namespace, const Manifest &manifest,
                        std::string *error) {
    for (;;) {
      std::shared_ptr<StateNamespace> entry = GetOrCreate(device_id, state_namespace);
      std::lock_guard<std::mutex> lock(entry->execution_mutex);
      if (entry->released) {
        continue;
      }
      return RegisterBindingsLocked(*entry, manifest, error);
    }
  }

  std::shared_ptr<CachedState> GetOrCreateStateLocked(StateNamespace &entry, const std::string &state_id,
                                                      const StateTensorCopy &initializer, std::string *error) {
    const auto existing = entry.states.find(state_id);
    if (existing != entry.states.end()) {
      if (existing->second->nbytes != initializer.data.size()) {
        *error = "state byte size changed: " + state_id;
        return nullptr;
      }
      return existing->second;
    }
    if (!CheckAcl(aclrtSetDevice(entry.device_id), "aclrtSetDevice")) {
      *error = "failed to select device for state";
      return nullptr;
    }
    auto state = std::make_shared<CachedState>(entry.device_id);
    state->nbytes = initializer.data.size();
    if (!CheckAcl(aclrtMallocAlign32(&state->current, state->nbytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMallocAlign32(state current)") ||
        !CheckAcl(aclrtMallocAlign32(&state->next, state->nbytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMallocAlign32(state next)") ||
        !CopyToStateBuffers(*state, initializer.data.data(), initializer.data.size(), error)) {
      return nullptr;
    }
    entry.states.emplace(state_id, state);
    return state;
  }

  bool CopyToStateBuffers(CachedState &state, const void *source, size_t nbytes, std::string *error) {
    if (source == nullptr || nbytes != state.nbytes || !CheckAcl(aclrtSetDevice(state.device_id), "aclrtSetDevice") ||
        !CheckAcl(aclrtMemcpy(state.current, state.nbytes, source, nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(state current H2D)") ||
        !CheckAcl(aclrtMemcpy(state.next, state.nbytes, source, nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(state next H2D)")) {
      *error = "failed to initialize state buffers";
      return false;
    }
    return true;
  }

 private:
  static std::string NamespaceKey(int device_id, const std::string &state_namespace) {
    return std::to_string(device_id) + "\n" + state_namespace;
  }

  std::shared_ptr<StateNamespace> Find(int device_id, const std::string &state_namespace) {
    std::lock_guard<std::mutex> lock(namespaces_mutex_);
    const auto iterator = namespaces_.find(NamespaceKey(device_id, state_namespace));
    return iterator == namespaces_.end() ? nullptr : iterator->second;
  }

  std::mutex namespaces_mutex_;
  std::unordered_map<std::string, std::shared_ptr<StateNamespace>> namespaces_;
};

struct PreparedState {
  const StateBinding *binding = nullptr;
  std::shared_ptr<CachedState> state;
};

class StateExecution {
 public:
  static std::unique_ptr<StateExecution> Begin(int device_id, const std::string &state_namespace,
                                               const Manifest &manifest, Span<EValue *> args, std::string *error) {
    DeviceStateCache &cache = DeviceStateCache::Instance();
    for (;;) {
      std::shared_ptr<StateNamespace> state_namespace_entry = cache.GetOrCreate(device_id, state_namespace);
      auto execution = std::unique_ptr<StateExecution>(new (std::nothrow) StateExecution(state_namespace_entry));
      if (!execution) {
        *error = "failed to allocate state execution lease";
        return nullptr;
      }
      if (state_namespace_entry->released) {
        continue;
      }
      if (!cache.RegisterBindingsLocked(*state_namespace_entry, manifest, error) ||
          !execution->Prepare(cache, manifest, args, error)) {
        return nullptr;
      }
      return execution;
    }
  }

  void *Input(size_t argument_index) const {
    const auto iterator = input_by_argument_.find(argument_index);
    return iterator == input_by_argument_.end() ? nullptr : iterator->second->state->current;
  }

  void *Output(size_t argument_index) const {
    const auto iterator = output_by_argument_.find(argument_index);
    return iterator == output_by_argument_.end() ? nullptr : iterator->second->state->next;
  }

  size_t Nbytes(size_t argument_index) const {
    auto iterator = input_by_argument_.find(argument_index);
    if (iterator != input_by_argument_.end()) {
      return iterator->second->state->nbytes;
    }
    iterator = output_by_argument_.find(argument_index);
    return iterator == output_by_argument_.end() ? 0 : iterator->second->state->nbytes;
  }

  bool IsSynchronizedOutput(size_t argument_index) const {
    const auto iterator = output_by_argument_.find(argument_index);
    return iterator != output_by_argument_.end() && iterator->second->binding->host_visibility == "synchronized";
  }

  void Commit() {
    for (PreparedState &prepared : states_) {
      if (HasOutput(*prepared.binding)) {
        std::swap(prepared.state->current, prepared.state->next);
      }
    }
  }

 private:
  static bool HasOutput(const StateBinding &binding) {
    return binding.next_output_arg != std::numeric_limits<size_t>::max();
  }

  explicit StateExecution(std::shared_ptr<StateNamespace> state_namespace)
      : state_namespace_(std::move(state_namespace)), lock_(state_namespace_->execution_mutex) {}

  bool Prepare(DeviceStateCache &cache, const Manifest &manifest, Span<EValue *> args, std::string *error) {
    const std::string current_owner = StateOwnerIdentity(manifest);
    states_.reserve(manifest.state_bindings.size());
    for (const StateBinding &binding : manifest.state_bindings) {
      const StateDefinition &definition = state_namespace_->definitions.at(binding.state_id);
      if (binding.reset_policy == "on_prefill" && definition.reset_owner.empty()) {
        *error = "on_prefill state has no registered reset owner: " + binding.state_id;
        return false;
      }
      const auto initializer = state_namespace_->initializers.find(binding.state_id);
      StateTensorCopy host_current;
      StateTensorCopy implicit_zero;

      const bool is_owner = binding.reset_owner && definition.reset_owner == current_owner;
      auto existing = state_namespace_->states.find(binding.state_id);
      if (existing == state_namespace_->states.end() && binding.reset_policy == "on_prefill" && !is_owner) {
        *error = "on_prefill state was consumed before its reset owner: " + binding.state_id;
        return false;
      }
      const StateTensorCopy *source = nullptr;
      if (initializer != state_namespace_->initializers.end()) {
        source = &initializer->second;
      } else if (binding.host_visibility == "synchronized") {
        if (binding.current_input_arg >= args.size() || !args[binding.current_input_arg]->isTensor()) {
          *error = "state current argument is not a tensor";
          return false;
        }
        const auto &current_tensor = args[binding.current_input_arg]->toTensor();
        const auto *current_data = static_cast<const uint8_t *>(current_tensor.const_data_ptr());
        if (current_data == nullptr || current_tensor.nbytes() == 0) {
          *error = "synchronized state current tensor has no storage";
          return false;
        }
        host_current.dtype = binding.dtype;
        host_current.shape = binding.shape;
        host_current.data.assign(current_data, current_data + current_tensor.nbytes());
        source = &host_current;
      } else if (binding.host_visibility == "device_only" && binding.reset_policy == "never" &&
                 MakeZeroStateInitializer(binding, &implicit_zero)) {
        source = &implicit_zero;
      }
      if (source == nullptr) {
        *error = "device_only state requires an initializer: " + binding.state_id;
        return false;
      }
      if (!StateMetadataMatches(definition, *source)) {
        *error = "state source metadata mismatch: " + binding.state_id;
        return false;
      }
      std::shared_ptr<CachedState> state =
          cache.GetOrCreateStateLocked(*state_namespace_, binding.state_id, *source, error);
      if (!state) {
        return false;
      }
      if (is_owner && !state->reset_done) {
        if (!cache.CopyToStateBuffers(*state, source->data.data(), source->data.size(), error)) {
          return false;
        }
        state->reset_done = true;
      }
      states_.push_back(PreparedState{&binding, std::move(state)});
      PreparedState *prepared = &states_.back();
      input_by_argument_.emplace(binding.current_input_arg, prepared);
      // Skip output registration if next_output_arg is -1 (state not updated)
      if (binding.next_output_arg != std::numeric_limits<size_t>::max()) {
        output_by_argument_.emplace(binding.next_output_arg, prepared);
      }
    }
    return true;
  }

  std::shared_ptr<StateNamespace> state_namespace_;
  std::unique_lock<std::mutex> lock_;
  std::vector<PreparedState> states_;
  std::unordered_map<size_t, PreparedState *> input_by_argument_;
  std::unordered_map<size_t, PreparedState *> output_by_argument_;
};

struct AscendDelegateHandle {
  AscendDelegateHandle(Manifest manifest_value, FreeableBuffer &&om_value,
                       std::vector<std::shared_ptr<DeviceWeight>> weights_value, int device_id_value,
                       std::string state_namespace_value)
      : manifest(std::move(manifest_value)),
        om(std::move(om_value)),
        weights(std::move(weights_value)),
        device_id(device_id_value),
        state_namespace(std::move(state_namespace_value)) {}

  Manifest manifest;
  FreeableBuffer om;
  std::vector<std::shared_ptr<DeviceWeight>> weights;
  std::unordered_map<std::string, std::shared_ptr<DeviceWeight>> weight_by_key;
  int device_id;
  std::string state_namespace;
  uint32_t model_id = 0;
  bool model_loaded = false;
  aclmdlDesc *model_desc = nullptr;
  std::mutex lifecycle_mutex;
  std::condition_variable no_executions;
  size_t in_flight = 0;
  bool closing = false;
};

class ExecutionPin {
 public:
  explicit ExecutionPin(AscendDelegateHandle *handle) : handle_(handle) {
    std::lock_guard<std::mutex> lock(handle_->lifecycle_mutex);
    if (!handle_->closing) {
      ++handle_->in_flight;
      acquired_ = true;
    }
  }

  ~ExecutionPin() {
    if (!acquired_) {
      return;
    }
    std::lock_guard<std::mutex> lock(handle_->lifecycle_mutex);
    --handle_->in_flight;
    if (handle_->in_flight == 0) {
      handle_->no_executions.notify_all();
    }
  }

  bool acquired() const {
    return acquired_;
  }

 private:
  AscendDelegateHandle *handle_;
  bool acquired_ = false;
};

const char *ScalarTypeName(ScalarType type) {
  switch (type) {
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Byte:
      return "uint8";
    case ScalarType::Char:
      return "int8";
    case ScalarType::Short:
      return "int16";
    case ScalarType::Int:
      return "int32";
    case ScalarType::Long:
      return "int64";
    case ScalarType::Half:
      return "float16";
    case ScalarType::BFloat16:
      return "bfloat16";
    case ScalarType::Float:
      return "float32";
    case ScalarType::Double:
      return "float64";
    default:
      return nullptr;
  }
}

aclDataType AclDataType(const std::string &type) {
  if (type == "bool") {
    return ACL_BOOL;
  }
  if (type == "uint8") {
    return ACL_UINT8;
  }
  if (type == "int8") {
    return ACL_INT8;
  }
  if (type == "int16") {
    return ACL_INT16;
  }
  if (type == "int32") {
    return ACL_INT32;
  }
  if (type == "int64") {
    return ACL_INT64;
  }
  if (type == "float16") {
    return ACL_FLOAT16;
  }
  if (type == "bfloat16") {
    return ACL_BF16;
  }
  if (type == "float32") {
    return ACL_FLOAT;
  }
  if (type == "float64") {
    return ACL_DOUBLE;
  }
  return ACL_DT_UNDEFINED;
}

bool IsCannGraphBoundaryName(Direction direction, size_t index, const char *name) {
  if (name == nullptr) {
    return false;
  }
  const std::string prefix =
      direction == Direction::kInput ? "arg_" + std::to_string(index) : "output_" + std::to_string(index) + "_";
  if (direction == Direction::kInput) {
    const std::string value(name);
    const std::string generic_prefix = "arg_";
    return prefix == value ||
           (value.compare(0, generic_prefix.size(), generic_prefix) == 0 && value.size() > generic_prefix.size() &&
            std::all_of(value.begin() + generic_prefix.size(), value.end(),
                        [](char character) { return character >= '0' && character <= '9'; }));
  }
  return std::string(name).compare(0, prefix.size(), prefix) == 0;
}

bool IsDynamicBinding(const IoBinding &binding) {
  return !binding.shape_range.empty();
}

bool ValidateTensor(const IoBinding &binding, const EValue &value, size_t expected_nbytes) {
  if (!value.isTensor()) {
    return false;
  }
  const auto &tensor = value.toTensor();
  const char *dtype = ScalarTypeName(tensor.scalar_type());
  if (dtype == nullptr || binding.dtype != dtype) {
    ET_LOG(Error, "Ascend tensor dtype mismatch: name=%s, expected=%s, actual=%s", binding.ascend_name.c_str(),
           binding.dtype.c_str(), dtype == nullptr ? "<unsupported>" : dtype);
    return false;
  }
  const bool dynamic = IsDynamicBinding(binding);
  if ((!dynamic && tensor.nbytes() != expected_nbytes) ||
      (dynamic && expected_nbytes != 0 && tensor.nbytes() > expected_nbytes)) {
    ET_LOG(Error,
           "Ascend tensor byte size mismatch: name=%s, model_capacity=%zu, "
           "actual=%zu",
           binding.ascend_name.c_str(), expected_nbytes, tensor.nbytes());
    return false;
  }
  if (tensor.sizes().size() != binding.shape.size()) {
    ET_LOG(Error, "Ascend tensor rank mismatch: name=%s", binding.ascend_name.c_str());
    return false;
  }
  for (size_t index = 0; index < binding.shape.size(); ++index) {
    const int64_t actual = static_cast<int64_t>(tensor.sizes()[index]);
    if (binding.shape[index] >= 0) {
      if (actual != binding.shape[index]) {
        ET_LOG(Error, "Ascend tensor static dimension mismatch: name=%s", binding.ascend_name.c_str());
        return false;
      }
      continue;
    }
    const DimensionRange &range = binding.shape_range[index];
    if (actual < range.minimum || actual > range.maximum) {
      ET_LOG(Error,
             "Ascend tensor dynamic dimension is out of range: name=%s, "
             "dimension=%zu, actual=%lld, range=[%lld,%lld]",
             binding.ascend_name.c_str(), index, static_cast<long long>(actual), static_cast<long long>(range.minimum),
             static_cast<long long>(range.maximum));
      return false;
    }
  }
  for (size_t index = 0; index < tensor.dim_order().size(); ++index) {
    if (tensor.dim_order()[index] != index) {
      return false;
    }
  }
  return true;
}

bool ValidateDescriptorIo(const AscendDelegateHandle &handle, std::string *error) {
  if (aclmdlGetNumInputs(handle.model_desc) != handle.manifest.inputs.size() ||
      aclmdlGetNumOutputs(handle.model_desc) != handle.manifest.outputs.size()) {
    *error = "ACL descriptor I/O count differs from manifest";
    return false;
  }
  auto append_error = [&](std::string message) {
    if (!error->empty()) {
      *error += "; ";
    }
    *error += std::move(message);
  };
  auto validate = [&](const IoBinding &binding, size_t index) {
    const bool input = binding.direction == Direction::kInput;
    const char *kind = input ? "input" : "output";
    const char *name = input ? aclmdlGetInputNameByIndex(handle.model_desc, index)
                             : aclmdlGetOutputNameByIndex(handle.model_desc, index);
    const aclDataType dtype =
        input ? aclmdlGetInputDataType(handle.model_desc, index) : aclmdlGetOutputDataType(handle.model_desc, index);
    aclmdlIODims dimensions{};
    const aclError dims_status = input ? aclmdlGetInputDims(handle.model_desc, index, &dimensions)
                                       : aclmdlGetOutputDims(handle.model_desc, index, &dimensions);
    const char *real_name = aclmdlGetTensorRealName(handle.model_desc, dimensions.name);
    const bool name_matches =
        name != nullptr && (binding.ascend_name == name || binding.ascend_name == dimensions.name ||
                            (real_name != nullptr && binding.ascend_name == real_name) ||
                            IsCannGraphBoundaryName(binding.direction, index, name));
    if (!name_matches || dtype != AclDataType(binding.dtype) || dims_status != ACL_SUCCESS) {
      append_error(std::string("ACL ") + kind + "[" + std::to_string(index) + "] metadata mismatch: expected name=" +
                   binding.ascend_name + ", dtype=" + std::to_string(AclDataType(binding.dtype)) +
                   "; actual name=" + (name == nullptr ? "<null>" : name) + ", resolved name=" +
                   (real_name == nullptr ? "<null>" : real_name) + ", dims name=" + dimensions.name +
                   ", dtype=" + std::to_string(dtype) + ", dims_status=" + std::to_string(dims_status));
      return false;
    }
    // GE/ACL may expose a logical rank-zero scalar as [0]. Keep the manifest
    // and state cache in the PyTorch shape domain and normalize only at this
    // compiled-model descriptor boundary.
    const bool acl_scalar = binding.shape.empty() && dimensions.dimCount == 1 && dimensions.dims[0] == 0;
    if (!acl_scalar && dimensions.dimCount != binding.shape.size()) {
      append_error(std::string("ACL ") + kind + " rank differs from manifest");
      return false;
    }
    for (size_t dimension = 0; !acl_scalar && dimension < binding.shape.size(); ++dimension) {
      const int64_t expected = binding.shape[dimension];
      const int64_t actual = dimensions.dims[dimension];
      if (actual != -1 && expected >= 0 && actual != expected) {
        append_error(std::string("ACL ") + kind + " dimensions differ from manifest");
        return false;
      }
    }
    if (input && IsDynamicBinding(binding)) {
      aclmdlIODimsRange dimensions_range{};
      const aclError range_status = aclmdlGetInputDimsRange(handle.model_desc, index, &dimensions_range);
      if (range_status != ACL_SUCCESS || dimensions_range.rangeCount != binding.shape_range.size()) {
        append_error("ACL input shape range differs from manifest");
        return false;
      }
      for (size_t dimension = 0; dimension < binding.shape_range.size(); ++dimension) {
        if (dimensions_range.range[dimension][0] != binding.shape_range[dimension].minimum ||
            dimensions_range.range[dimension][1] != binding.shape_range[dimension].maximum) {
          append_error("ACL input shape range differs from manifest");
          return false;
        }
      }
    }
    return true;
  };
  bool valid = true;
  for (size_t index = 0; index < handle.manifest.inputs.size(); ++index) {
    if (!validate(handle.manifest.inputs[index], index)) {
      valid = false;
    }
  }
  for (size_t index = 0; index < handle.manifest.outputs.size(); ++index) {
    if (!validate(handle.manifest.outputs[index], index)) {
      valid = false;
    }
  }
  return valid;
}

class InvocationBuffers {
 public:
  ~InvocationBuffers() {
    for (aclDataBuffer *buffer : data_buffers_) {
      if (buffer != nullptr) {
        aclDestroyDataBuffer(buffer);
      }
    }
    if (inputs_ != nullptr) {
      aclmdlDestroyDataset(inputs_);
    }
    if (outputs_ != nullptr) {
      aclmdlDestroyDataset(outputs_);
    }
    for (aclTensorDesc *descriptor : tensor_descriptors_) {
      if (descriptor != nullptr) {
        aclDestroyTensorDesc(descriptor);
      }
    }
    for (void *buffer : device_buffers_) {
      if (buffer != nullptr) {
        aclrtFree(buffer);
      }
    }
  }

  bool Initialize() {
    inputs_ = aclmdlCreateDataset();
    outputs_ = aclmdlCreateDataset();
    return inputs_ != nullptr && outputs_ != nullptr;
  }

  bool Add(bool input, void *device_data, size_t size, std::string *error) {
    aclDataBuffer *data_buffer = aclCreateDataBuffer(device_data, size);
    if (data_buffer == nullptr) {
      *error = "aclCreateDataBuffer failed";
      return false;
    }
    data_buffers_.push_back(data_buffer);
    const aclError status = aclmdlAddDatasetBuffer(input ? inputs_ : outputs_, data_buffer);
    if (status != ACL_SUCCESS) {
      *error = "aclmdlAddDatasetBuffer failed";
      return false;
    }
    return true;
  }

  bool SetTensorDesc(bool input, size_t index, aclDataType data_type, const Tensor &tensor, std::string *error) {
    if (tensor.sizes().size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      *error = "tensor rank is too large for aclCreateTensorDesc";
      return false;
    }
    std::vector<int64_t> dimensions;
    dimensions.reserve(tensor.sizes().size());
    for (SizesType dimension : tensor.sizes()) {
      dimensions.push_back(static_cast<int64_t>(dimension));
    }
    aclTensorDesc *descriptor =
        aclCreateTensorDesc(data_type, static_cast<int>(dimensions.size()), dimensions.data(), ACL_FORMAT_ND);
    if (descriptor == nullptr) {
      *error = "aclCreateTensorDesc failed";
      return false;
    }
    const aclError status = aclmdlSetDatasetTensorDesc(input ? inputs_ : outputs_, descriptor, index);
    if (status != ACL_SUCCESS) {
      aclDestroyTensorDesc(descriptor);
      *error = "aclmdlSetDatasetTensorDesc failed with ACL error " + std::to_string(status);
      return false;
    }
    tensor_descriptors_.push_back(descriptor);
    return true;
  }

  void TrackDeviceBuffer(void *buffer) {
    device_buffers_.push_back(buffer);
  }

  aclmdlDataset *inputs() const {
    return inputs_;
  }

  aclmdlDataset *outputs() const {
    return outputs_;
  }

 private:
  aclmdlDataset *inputs_ = nullptr;
  aclmdlDataset *outputs_ = nullptr;
  std::vector<aclDataBuffer *> data_buffers_;
  std::vector<aclTensorDesc *> tensor_descriptors_;
  std::vector<void *> device_buffers_;
};

class AscendBackend final : public BackendInterface {
 public:
  bool is_available() const override {
    uint32_t count = 0;
    return aclrtGetDeviceCount(&count) == ACL_SUCCESS && count > 0;
  }

  Result<DelegateHandle *> init(BackendInitContext &context, FreeableBuffer *processed,
                                ArrayRef<CompileSpec> compile_specs) const override {
    // 初始化acl
    if (processed == nullptr || !EnsureAclInitialized()) {
      return Error::DelegateInvalidCompatibility;
    }
    // 解析并校验manifest
    Manifest manifest;
    std::string error;
    if (!ParseManifest(processed->data(), processed->size(), &manifest, &error)) {
      ET_LOG(Error, "Ascend manifest rejected: %s", error.c_str());
      return Error::DelegateInvalidCompatibility;
    }
    const char *method_name = context.get_method_name();
    size_t compile_delegate_index = 0;
    std::string spec_method;
    std::string spec_delegate_index;
    CompileOptionValue(compile_specs, "method_name", &spec_method);
    CompileOptionValue(compile_specs, "delegate_index", &spec_delegate_index);
    if (method_name == nullptr || manifest.method != method_name || spec_method != manifest.method ||
        !ParseNonNegativeSize(spec_delegate_index, &compile_delegate_index) ||
        compile_delegate_index != manifest.delegate_index) {
      ET_LOG(Error, "Ascend delegate identity does not match runtime context");
      return Error::DelegateInvalidCompatibility;
    }
    // 指定deviceid和state_namespace
    int device_id = 0;
    Result<int> configured_device = context.get_runtime_spec<int>("device_id");
    if (configured_device.ok()) {
      device_id = configured_device.get();
    } else if (configured_device.error() != Error::NotFound) {
      return configured_device.error();
    }
    std::string state_namespace;
    if (!manifest.state_bindings.empty()) {
      if (!configured_device.ok()) {
        ET_LOG(Error, "Stateful Ascend delegates require device_id");
        return Error::InvalidArgument;
      }
      Result<const char *> configured_namespace = context.get_runtime_spec<const char *>("state_namespace");
      if (!configured_namespace.ok() || configured_namespace.get() == nullptr ||
          configured_namespace.get()[0] == '\0') {
        ET_LOG(Error,
               "Stateful Ascend delegates require a non-empty "
               "state_namespace");
        return Error::InvalidArgument;
      }
      state_namespace = configured_namespace.get();
      if (!DeviceStateCache::Instance().RegisterBindings(device_id, state_namespace, manifest, &error)) {
        ET_LOG(Error, "Ascend state contract rejected: %s", error.c_str());
        return Error::InvalidProgram;
      }
    }
    // 校验硬件兼容性
    if (!CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice") || !AclSocIsCompatible(manifest)) {
      ET_LOG(Error, "Ascend device is incompatible with compiled artifact");
      return Error::DelegateInvalidCompatibility;
    }
    // 加载和校验manifest
    const NamedDataMap *named_data_map = context.get_named_data_map();
    if (named_data_map == nullptr) {
      return Error::NotFound;
    }
    Result<FreeableBuffer> om_result = named_data_map->get_data(manifest.model.named_data_key);
    if (!om_result.ok()) {
      ET_LOG(Error, "OM named data is missing");
      return om_result.error();
    }
    FreeableBuffer om(std::move(om_result.get()));
    std::string digest;
    if (om.size() != manifest.model.nbytes || !Sha256(om.data(), om.size(), &digest, &error) ||
        digest != manifest.model.sha256) {
      ET_LOG(Error, "OM named data identity mismatch");
      return Error::InvalidProgram;
    }
    // 读取权重
    std::vector<std::shared_ptr<DeviceWeight>> weights;
    weights.reserve(manifest.weight_groups.size());
    for (const WeightGroup &group : manifest.weight_groups) {
      auto weight = SharedWeightCache::Instance().Acquire(  // 权重加载到设备
          device_id, group, *named_data_map, &error);
      if (!weight) {
        ET_LOG(Error, "External weight load failed: %s", error.c_str());
        return Error::InvalidProgram;
      }
      weights.push_back(std::move(weight));
    }
    // 创建handle，持有manifest、om、weights、device_id、state_namespace等信息，后续执行使用
    auto handle = std::unique_ptr<AscendDelegateHandle>(new (std::nothrow) AscendDelegateHandle(
        std::move(manifest), std::move(om), std::move(weights), device_id, std::move(state_namespace)));
    if (!handle) {
      return Error::MemoryAllocationFailed;
    }
    for (size_t index = 0; index < handle->manifest.weight_groups.size(); ++index) {
      handle->weight_by_key.emplace(handle->manifest.weight_groups[index].key, handle->weights[index]);
    }
    // acl模型加载配置
    aclmdlConfigHandle *config = aclmdlCreateConfigHandle();
    if (config == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    // 指定模型来源：内存
    // 指定OM数据地址
    // 指定OM数据大小
    const size_t load_type = ACL_MDL_LOAD_FROM_MEM;
    const void *om_address = handle->om.data();
    const size_t om_size = handle->om.size();
    bool configured = CheckAcl(aclmdlSetConfigOpt(config, ACL_MDL_LOAD_TYPE_SIZET, &load_type, sizeof(load_type)),
                               "aclmdlSetConfigOpt(load type)") &&
                      CheckAcl(aclmdlSetConfigOpt(config, ACL_MDL_MEM_ADDR_PTR, &om_address, sizeof(om_address)),
                               "aclmdlSetConfigOpt(model address)") &&
                      CheckAcl(aclmdlSetConfigOpt(config, ACL_MDL_MEM_SIZET, &om_size, sizeof(om_size)),
                               "aclmdlSetConfigOpt(model size)");
    for (const FileConstantBinding &binding : handle->manifest.file_constant_bindings) {
      if (!configured) {
        break;
      }
      // 按照权重组的key找到加载了的权重，再根据offset找到精确的常量权重地址
      const auto group = handle->weight_by_key.find(binding.weight_group_key);
      if (group == handle->weight_by_key.end()) {
        configured = false;
        break;
      }
      void *address = static_cast<uint8_t *>(group->second->data) + binding.offset;
      size_t binding_span = 0;
      if (!AlignUp(binding.nbytes, 32, &binding_span) || binding.offset > group->second->size ||
          binding_span > group->second->size - binding.offset) {
        ET_LOG(Error, "External weight binding span is invalid: %s", binding.file_name.c_str());
        configured = false;
        break;
      }
      configured = CheckAcl(  // 注册外置权重地址
          aclmdlSetExternalWeightAddress(config, binding.file_name.c_str(), address, binding_span),
          "aclmdlSetExternalWeightAddress");
    }
    if (configured) {  // 加载模型
      configured = CheckAcl(aclmdlLoadWithConfig(config, &handle->model_id), "aclmdlLoadWithConfig");
      handle->model_loaded = configured;
    }
    aclmdlDestroyConfigHandle(config);
    if (!configured) {
      return Error::InvalidProgram;
    }
    // 创建模型描述对象，用于查询模型输入输出的数量、名称、dtype、shape等
    handle->model_desc = aclmdlCreateDesc();
    if (handle->model_desc == nullptr ||
        !CheckAcl(aclmdlGetDesc(handle->model_desc, handle->model_id), "aclmdlGetDesc") ||
        !ValidateDescriptorIo(*handle, &error)) {
      ET_LOG(Error, "ACL model descriptor rejected: %s", error.c_str());
      if (handle->model_desc != nullptr) {
        aclmdlDestroyDesc(handle->model_desc);
        handle->model_desc = nullptr;
      }
      aclmdlUnload(handle->model_id);
      handle->model_loaded = false;
      return Error::InvalidProgram;
    }
    return handle.release();
  }

  Error execute(BackendExecutionContext &, DelegateHandle *raw_handle, Span<EValue *> args) const override {
    // 保护handle运行期间不被清理
    if (raw_handle == nullptr) {
      return Error::InvalidArgument;
    }
    auto *handle = static_cast<AscendDelegateHandle *>(raw_handle);
    ExecutionPin pin(handle);
    if (!pin.acquired()) {
      return Error::InvalidState;
    }
    if (!CheckAcl(aclrtSetDevice(handle->device_id), "aclrtSetDevice")) {  // 指定执行设备
      return Error::Internal;
    }
    auto requires_evalue = [&](const IoBinding &binding) {
      if (binding.semantic_role == "data") {
        return true;
      }
      for (const StateBinding &state : handle->manifest.state_bindings) {
        const bool matches =
            (binding.semantic_role == "state_current" && state.current_input_arg == binding.et_arg_index) ||
            (binding.semantic_role == "state_next" && state.next_output_arg == binding.et_arg_index);
        if (matches) {
          return state.host_visibility == "synchronized";
        }
      }
      return false;
    };
    // ExecuTorch allocates one argument slot per manifest binding regardless of
    // host visibility. Filtering by requires_evalue here would under-count a
    // device_only state occupying the trailing slots, which is exactly where
    // mutated module buffers land, and reject a valid invocation.
    if (args.size() != ExpectedArgumentCount(handle->manifest)) {
      return Error::InvalidArgument;
    }
    // 校验素有输入 输出tensor
    for (const IoBinding &binding : handle->manifest.inputs) {
      const size_t index = &binding - handle->manifest.inputs.data();
      const size_t nbytes = aclmdlGetInputSizeByIndex(handle->model_desc, index);
      if (requires_evalue(binding) &&
          (binding.et_arg_index >= args.size() || !ValidateTensor(binding, *args[binding.et_arg_index], nbytes))) {
        return Error::InvalidArgument;
      }
    }
    for (const IoBinding &binding : handle->manifest.outputs) {
      const size_t index = &binding - handle->manifest.outputs.data();
      const size_t nbytes = aclmdlGetOutputSizeByIndex(handle->model_desc, index);
      if (requires_evalue(binding) &&
          (binding.et_arg_index >= args.size() || !ValidateTensor(binding, *args[binding.et_arg_index], nbytes))) {
        return Error::InvalidArgument;
      }
    }
    // 准备有状态执行
    std::string error;
    std::unique_ptr<StateExecution> state_execution;
    if (!handle->manifest.state_bindings.empty()) {
      state_execution =
          StateExecution::Begin(handle->device_id, handle->state_namespace, handle->manifest, args, &error);
      if (!state_execution) {
        ET_LOG(Error, "Ascend state preparation failed: %s", error.c_str());
        return Error::InvalidState;
      }
    }
    // 创建acl 输入/输出容器invocation
    InvocationBuffers invocation;
    if (!invocation.Initialize()) {
      return Error::MemoryAllocationFailed;
    }
    for (size_t index = 0; index < handle->manifest.inputs.size(); ++index) {
      const IoBinding &binding = handle->manifest.inputs[index];
      if (binding.semantic_role == "state_current") {
        size_t nbytes = aclmdlGetInputSizeByIndex(handle->model_desc, index);
        if (nbytes == 0) {
          nbytes = state_execution->Nbytes(binding.et_arg_index);
        }
        void *state_input = state_execution->Input(binding.et_arg_index);
        if (state_input == nullptr || !invocation.Add(true, state_input, nbytes, &error)) {
          return Error::InvalidState;
        }
        continue;
      }
      const auto &tensor = args[binding.et_arg_index]->toTensor();
      const size_t data_nbytes = tensor.nbytes();
      size_t nbytes = data_nbytes;
      const size_t model_input_size = aclmdlGetInputSizeByIndex(handle->model_desc, index);
      if (IsDynamicBinding(binding) && model_input_size > nbytes) {
        nbytes = model_input_size;
      }
      void *device_buffer = nullptr;
      if (!CheckAcl(aclrtMallocAlign32(&device_buffer, nbytes, ACL_MEM_MALLOC_HUGE_FIRST),
                    "aclrtMallocAlign32(input)")) {
        return Error::MemoryAllocationFailed;
      }
      invocation.TrackDeviceBuffer(device_buffer);
      if (!CheckAcl(
              aclrtMemcpy(device_buffer, data_nbytes, tensor.const_data_ptr(), data_nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy(input H2D)") ||
          !invocation.Add(true, device_buffer, nbytes, &error) ||
          (IsDynamicBinding(binding) &&
           !invocation.SetTensorDesc(true, index, AclDataType(binding.dtype), tensor, &error))) {
        ET_LOG(Error, "Ascend dynamic input preparation failed: %s", error.c_str());
        return Error::Internal;
      }
    }

    std::vector<void *> output_buffers;
    std::vector<size_t> output_capacities;
    output_buffers.reserve(handle->manifest.outputs.size());
    output_capacities.reserve(handle->manifest.outputs.size());
    for (size_t index = 0; index < handle->manifest.outputs.size(); ++index) {
      const IoBinding &binding = handle->manifest.outputs[index];
      if (binding.semantic_role == "state_next") {
        size_t nbytes = aclmdlGetOutputSizeByIndex(handle->model_desc, index);
        if (nbytes == 0) {
          nbytes = state_execution->Nbytes(binding.et_arg_index);
        }
        void *state_output = state_execution->Output(binding.et_arg_index);
        if (state_output == nullptr || !invocation.Add(false, state_output, nbytes, &error)) {
          return Error::InvalidState;
        }
        output_buffers.push_back(state_output);
        output_capacities.push_back(nbytes);
        continue;
      }
      const auto &tensor = args[binding.et_arg_index]->toTensor();
      size_t nbytes = tensor.nbytes();
      const bool dynamic_output = IsDynamicBinding(binding);
      const size_t model_output_size = aclmdlGetOutputSizeByIndex(handle->model_desc, index);
      if (dynamic_output && model_output_size == 0) {
        if (!invocation.Add(false, nullptr, 0, &error)) {
          return Error::Internal;
        }
        output_buffers.push_back(nullptr);
        output_capacities.push_back(0);
        continue;
      }
      if (dynamic_output) {
        nbytes = model_output_size;
      }
      void *device_buffer = nullptr;
      if (!CheckAcl(aclrtMallocAlign32(&device_buffer, nbytes, ACL_MEM_MALLOC_HUGE_FIRST),
                    "aclrtMallocAlign32(output)")) {
        return Error::MemoryAllocationFailed;
      }
      invocation.TrackDeviceBuffer(device_buffer);
      output_buffers.push_back(device_buffer);
      output_capacities.push_back(nbytes);
      if (!invocation.Add(false, device_buffer, nbytes, &error)) {
        return Error::Internal;
      }
    }
    // 执行OM
    if (!CheckAcl(aclmdlExecute(handle->model_id, invocation.inputs(), invocation.outputs()), "aclmdlExecute")) {
      return Error::Internal;
    }
    // 回传输出到host
    for (size_t index = 0; index < handle->manifest.outputs.size(); ++index) {
      const IoBinding &binding = handle->manifest.outputs[index];
      if (binding.semantic_role == "state_next" && !state_execution->IsSynchronizedOutput(binding.et_arg_index)) {
        continue;
      }
      auto &tensor = args[binding.et_arg_index]->toTensor();
      size_t nbytes = tensor.nbytes();
      aclTensorDesc *output_descriptor = nullptr;
      if (IsDynamicBinding(binding)) {
        output_descriptor = aclmdlGetDatasetTensorDesc(invocation.outputs(), index);
        if (output_descriptor == nullptr) {
          ET_LOG(Error, "Ascend dynamic output descriptor is null");
          return Error::Internal;
        }
      }
      if (output_descriptor != nullptr) {
        const size_t rank = aclGetTensorDescNumDims(output_descriptor);
        if (rank == ACL_UNKNOWN_RANK || rank != binding.shape.size() ||
            rank > static_cast<size_t>(std::numeric_limits<int>::max())) {
          ET_LOG(Error, "Ascend dynamic output has invalid rank");
          return Error::Internal;
        }
        std::vector<SizesType> dimensions;
        dimensions.reserve(rank);
        for (size_t dimension_index = 0; dimension_index < rank; ++dimension_index) {
          int64_t dimension = 0;
          if (!CheckAcl(aclGetTensorDescDimV2(output_descriptor, dimension_index, &dimension),
                        "aclGetTensorDescDimV2") ||
              dimension < 0 || dimension > std::numeric_limits<SizesType>::max()) {
            ET_LOG(Error, "Ascend dynamic output has invalid dimension");
            return Error::Internal;
          }
          const int64_t manifest_dimension = binding.shape[dimension_index];
          const DimensionRange &range = binding.shape_range[dimension_index];
          if ((manifest_dimension >= 0 && dimension != manifest_dimension) ||
              (manifest_dimension == -1 && (dimension < range.minimum || dimension > range.maximum))) {
            ET_LOG(Error, "Ascend dynamic output dimension is out of range");
            return Error::Internal;
          }
          dimensions.push_back(static_cast<SizesType>(dimension));
        }
        const Error resize_status =
            executorch::runtime::resize_tensor(tensor, ArrayRef<SizesType>(dimensions.data(), dimensions.size()));
        if (resize_status != Error::Ok) {
          ET_LOG(Error, "Ascend dynamic output resize failed: 0x%x", static_cast<unsigned int>(resize_status));
          return resize_status;
        }
        nbytes = aclGetTensorDescSize(output_descriptor);
        if (output_buffers[index] == nullptr) {
          aclDataBuffer *output_data_buffer = aclmdlGetDatasetBuffer(invocation.outputs(), index);
          void *system_output = output_data_buffer == nullptr ? nullptr : aclGetDataBufferAddr(output_data_buffer);
          if (system_output == nullptr) {
            ET_LOG(Error, "Ascend dynamic output buffer is null");
            return Error::Internal;
          }
          output_buffers[index] = system_output;
          output_capacities[index] = nbytes;
          invocation.TrackDeviceBuffer(system_output);
        }
      }
      if (nbytes == 0 || nbytes != tensor.nbytes() || nbytes > output_capacities[index]) {
        ET_LOG(Error,
               "Ascend dynamic output byte size is invalid: actual=%zu, "
               "tensor=%zu, capacity=%zu",
               nbytes, tensor.nbytes(), output_capacities[index]);
        return Error::Internal;
      }
      if (!CheckAcl(
              aclrtMemcpy(tensor.mutable_data_ptr(), nbytes, output_buffers[index], nbytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy(output D2H)")) {
        return Error::Internal;
      }
    }
    if (state_execution) {
      state_execution->Commit();
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle *raw_handle) const override {
    if (raw_handle == nullptr) {
      return;
    }
    auto *handle = static_cast<AscendDelegateHandle *>(raw_handle);
    {
      std::unique_lock<std::mutex> lock(handle->lifecycle_mutex);
      handle->closing = true;
      handle->no_executions.wait(lock, [&]() { return handle->in_flight == 0; });
    }
    aclrtSetDevice(handle->device_id);
    if (handle->model_desc != nullptr) {
      aclmdlDestroyDesc(handle->model_desc);
      handle->model_desc = nullptr;
    }
    if (handle->model_loaded) {
      aclmdlUnload(handle->model_id);
      handle->model_loaded = false;
    }
    delete handle;
  }
};

AscendBackend backend_instance;
Backend backend{kBackendName, &backend_instance};
[[maybe_unused]] const Error registration_status = executorch::runtime::register_backend(backend);

}  // namespace

StateStatus SetStateInitializer(int device_id, const char *state_namespace, const char *state_id,
                                const StateTensorData &initializer) {
  if (state_namespace == nullptr || state_id == nullptr) {
    return StateStatus::kInvalidArgument;
  }
  return DeviceStateCache::Instance().SetInitializer(device_id, state_namespace, state_id, initializer);
}

StateStatus ResetState(int device_id, const char *state_namespace, const char *state_id,
                       const StateTensorData &source) {
  if (state_namespace == nullptr || state_id == nullptr) {
    return StateStatus::kInvalidArgument;
  }
  return DeviceStateCache::Instance().Reset(device_id, state_namespace, state_id, source);
}

StateStatus ReleaseStateNamespace(int device_id, const char *state_namespace) {
  if (state_namespace == nullptr) {
    return StateStatus::kInvalidArgument;
  }
  return DeviceStateCache::Instance().Release(device_id, state_namespace);
}

}  // namespace executorch_npu
