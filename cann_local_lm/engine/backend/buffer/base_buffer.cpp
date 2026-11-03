/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#include "base_buffer.h"
#include <new>
#include "securec.h"
#include "infra/log.h"
// inc
#include "infra/securestl.h"
#include "infra/assertion.h"

namespace hiai {
BaseBuffer::BaseBuffer() {}

BaseBuffer::BaseBuffer(const BaseBuffer& other)
{
    if (!other.isOwner_) {
        isOwner_ = other.isOwner_;
        data_ = other.data_;
        size_ = other.size_;
    } else {
        isOwner_ = false;
        data_ = nullptr;
        size_ = 0;
        CopyFrom(other.data_, other.size_);
    }
}

BaseBuffer::BaseBuffer(uint8_t* data, size_t size, bool shouldOwn) : isOwner_(shouldOwn), data_(data), size_(size) {}

BaseBuffer::BaseBuffer(uint8_t* data, size_t size, std::function<void(uint8_t*)> freeFunc)
    : data_(data), size_(size), freeFunc_(freeFunc)
{
    isOwner_ = true;
}

BaseBuffer::~BaseBuffer() { Clear(); }

bool BaseBuffer::Init(size_t size, uint8_t defaultVal)
{
    if (size == 0) {
        return false;
    }

    Clear();

    data_ = new (std::nothrow) uint8_t[size];
    if (data_ == nullptr) {
        CANN_LLM_LOGE("new failed.");
        return false;
    }
    if (memset_s(data_, size, defaultVal, size) != EOK) {
        CANN_LLM_LOGE("buffer memset failed.");
        delete[] data_;
        data_ = nullptr;
        return false;
    }

    size_ = size;
    isOwner_ = true;
    return true;
}

bool BaseBuffer::CopyFrom(const uint8_t* data, size_t size)
{
    if (data == nullptr || size == 0) {
        return false;
    }

    Clear();

    data_ = new (std::nothrow) uint8_t[size];
    if (data_ == nullptr) {
        CANN_LLM_LOGE("new failed.");
        return false;
    }
    if (memcpy_s(data_, size, data, size) != EOK) {
        CANN_LLM_LOGE("buffer memset failed.");
        delete[] data_;
        data_ = nullptr;
        return false;
    }

    size_ = size;
    isOwner_ = true;
    return true;
}

void BaseBuffer::SetData(uint8_t* data, size_t size, bool shouldOwn)
{
    Clear();

    data_ = data;
    size_ = size;
    isOwner_ = shouldOwn;
}

const uint8_t* BaseBuffer::GetData() const { return data_; }

void* BaseBuffer::GetData() { return data_; }

uint8_t* BaseBuffer::MutableData() { return data_; }

size_t BaseBuffer::GetSize() const { return size_; }

void BaseBuffer::Clear()
{
    if (!isOwner_ || data_ == nullptr) {
        return;
    }

    if (freeFunc_ != nullptr) {
        freeFunc_(data_);
    } else {
        delete[] data_;
    }

    data_ = nullptr;
    size_ = 0;
}

// 原实现位于 support/buffer/local_buffer.cpp（LocalBuffer 纯适配器已并入 BaseBuffer）
std::shared_ptr<IBuffer> CreateLocalBuffer(size_t size)
{
    HIAI_EXPECT_TRUE_R(size > 0, nullptr);

    std::shared_ptr<BaseBuffer> buffer = make_shared_nothrow<BaseBuffer>();
    HIAI_EXPECT_NOT_NULL_R(buffer, nullptr);

    if (!buffer->Init(size)) {
        return nullptr;
    }
    return buffer;
}

std::shared_ptr<IBuffer> CreateLocalBuffer(void* data, size_t size, bool shouldOwn)
{
    HIAI_EXPECT_TRUE_R(data != nullptr && size > 0, nullptr);

    std::shared_ptr<BaseBuffer> buffer = make_shared_nothrow<BaseBuffer>();
    HIAI_EXPECT_NOT_NULL_R(buffer, nullptr);

    buffer->SetData(static_cast<uint8_t*>(data), size, shouldOwn);
    return buffer;
}
} // namespace hiai
