/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: file util impl
 *
 */

#ifndef _FMK_FILE_UTIL_IMPL_H
#define _FMK_FILE_UTIL_IMPL_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include "types.h"
#include "base_buffer.h"

namespace hiai {
class FileUtilImpl {
public:
    static Status LoadFileToBuffer(const std::string& fileName, std::shared_ptr<BaseBuffer>& buffer);
    static Status LoadFileToBuffer(const std::string& fileName, std::shared_ptr<BaseBuffer>& buffer, const uint64_t size);
    static Status LoadFileAsUint8(const std::string& fileName, uint8_t*& data, uint64_t& size);
    static Status LoadFileAsUint8ByOffsetAndSize(
        const std::string& fileName, uint8_t*& data, const uint64_t offset, const uint64_t size);
};
}
#endif