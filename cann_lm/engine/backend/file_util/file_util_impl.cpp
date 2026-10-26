/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025 All rights reserved.
 * Description: img file util
 */

#include "file_util_impl.h"
#include <climits>
#include <fstream>
#include <vector>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>

#include "infra/assertion.h"
#include "infra/log.h"

static int64_t MAX_SUPPORTED_FILE_SIZE = 0x1C0000000ULL; // 7GB

namespace hiai {
namespace {
Status LoadToBufferBySize(FILE* fp, uint64_t size, std::shared_ptr<BaseBuffer>& buffer)
{
    HIAI_EXPECT_NOT_NULL(buffer);
    uint8_t* data = new (std::nothrow) uint8_t[size];
    HIAI_EXPECT_NOT_NULL(data);
    buffer->SetData(data, size, true);
    uint64_t readSize = fread(buffer->MutableData(), sizeof(uint8_t), size, fp);
    HIAI_EXPECT_TRUE(readSize == size);
    return hiai::SUCCESS;
}

long GetFileSize(FILE* fp)
{
    if (fp == nullptr) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        CANN_LLM_LOGE("fseek SEEK_END error.");
        return -1;
    }

    long fileSize = ftell(fp);
    if (fileSize <= 0) {
        CANN_LLM_LOGE("ftell failed.");
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        CANN_LLM_LOGW("fseek SEEK_SET error.");
    }
    return fileSize;
}

FILE* OpenFile(const std::string& fileName, const std::string& mode)
{
    HIAI_EXPECT_TRUE_R(!fileName.empty(), nullptr);
    HIAI_EXPECT_TRUE_R(fileName.size() < PATH_MAX, nullptr);
    char pathTemp[PATH_MAX + 1] = {0x00};
    if (realpath(fileName.c_str(), pathTemp) == nullptr) {
        CANN_LLM_LOGE("fileName: %s is invalid.", fileName.c_str());
        return nullptr;
    }
    return fopen(pathTemp, mode.c_str());
}
} // namespace

Status FileUtilImpl::LoadFileToBuffer(const std::string& fileName, std::shared_ptr<BaseBuffer>& buffer)
{
    HIAI_EXPECT_TRUE(!fileName.empty());
    HIAI_EXPECT_NOT_NULL(buffer);
    std::unique_ptr<FILE, std::function<void(FILE*)>> fp(OpenFile(fileName, "rb"), [](FILE* p) { fclose(p); });
    HIAI_EXPECT_NOT_NULL(fp);
    long fileSize = GetFileSize(fp.get());
    HIAI_EXPECT_TRUE(fileSize > 0);
    HIAI_EXPECT_TRUE(fileSize <= MAX_SUPPORTED_FILE_SIZE);
    HIAI_EXPECT_EXEC(LoadToBufferBySize(fp.get(), static_cast<uint64_t>(fileSize), buffer));
    return hiai::SUCCESS;
}

Status FileUtilImpl::LoadFileToBuffer(const std::string& fileName, std::shared_ptr<BaseBuffer>& buffer,
                                      const uint64_t size)
{
    HIAI_EXPECT_TRUE(!fileName.empty());
    HIAI_EXPECT_TRUE(size > 0);
    HIAI_EXPECT_NOT_NULL(buffer);
    std::unique_ptr<FILE, std::function<void(FILE*)>> fp(OpenFile(fileName, "rb"), [](FILE* p) { fclose(p); });
    HIAI_EXPECT_NOT_NULL(fp);
    long fileSize = GetFileSize(fp.get());
    HIAI_EXPECT_TRUE(size <= static_cast<uint64_t>(fileSize));
    HIAI_EXPECT_EXEC(LoadToBufferBySize(fp.get(), size, buffer));
    return hiai::SUCCESS;
}

Status FileUtilImpl::LoadFileAsUint8(const std::string& fileName, uint8_t*& data, uint64_t& size)
{
    HIAI_EXPECT_TRUE(!fileName.empty());
    std::unique_ptr<FILE, std::function<void(FILE*)>> fp(OpenFile(fileName, "rb"), [](FILE* p) { fclose(p); });
    HIAI_EXPECT_NOT_NULL(fp);
    HIAI_EXPECT_NOT_NULL(fp.get());
    long fileSize = GetFileSize(fp.get());
    HIAI_EXPECT_TRUE(fileSize > 0 && fileSize <= MAX_SUPPORTED_FILE_SIZE);
    size = static_cast<uint64_t>(fileSize);

    // 若传入空指针，在函数内部分配资源
    if (data == nullptr) {
        data = new (std::nothrow) uint8_t[size];
        HIAI_EXPECT_NOT_NULL(data);
        uint64_t readSize = fread(data, sizeof(uint8_t), size, fp.get());
        if (readSize != size) {
            delete[] data;
            data = nullptr;
            return hiai::FAILURE;
        }
    } else {
        // 若传入的不是空指针，外部需要保证指针分配的长度不会出现内存越界
        uint64_t readSize = fread(data, sizeof(uint8_t), size, fp.get());
        HIAI_EXPECT_TRUE(readSize == size);
    }
    return hiai::SUCCESS;
}

Status FileUtilImpl::LoadFileAsUint8ByOffsetAndSize(const std::string& fileName, uint8_t*& data, const uint64_t offset,
                                                    const uint64_t size)
{
    HIAI_EXPECT_TRUE(!fileName.empty());
    HIAI_EXPECT_TRUE(size > 0);
    std::unique_ptr<FILE, std::function<void(FILE*)>> fp(OpenFile(fileName, "rb"), [](FILE* p) { fclose(p); });
    HIAI_EXPECT_NOT_NULL(fp);
    HIAI_EXPECT_NOT_NULL(fp.get());
    long fileSize = GetFileSize(fp.get());
    HIAI_EXPECT_TRUE(fileSize > 0);
    HIAI_EXPECT_TRUE(size + offset <= static_cast<uint64_t>(fileSize));
    HIAI_EXPECT_TRUE(fseek(fp.get(), offset, SEEK_SET) == 0);

    // 若传入空指针，在函数内部分配资源
    if (data == nullptr) {
        data = new (std::nothrow) uint8_t[size];
        HIAI_EXPECT_NOT_NULL(data);
        uint64_t readSize = fread(data, sizeof(uint8_t), size, fp.get());
        if (readSize != size) {
            delete[] data;
            data = nullptr;
            return hiai::FAILURE;
        }
    } else {
        // 若传入的不是空指针，外部需要保证指针分配的长度不会出现内存越界
        uint64_t readSize = fread(data, sizeof(uint8_t), size, fp.get());
        HIAI_EXPECT_TRUE(readSize == size);
    }
    return hiai::SUCCESS;
}
} // namespace hiai
