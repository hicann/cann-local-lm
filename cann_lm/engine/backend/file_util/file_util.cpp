/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#include "file_util.h"
#include <cerrno>
#include <cstdlib>
#include "infra/assertion.h"
#include "infra/securestl.h"
#include "file_util_impl.h"

using namespace std;

namespace hiai {

// 当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
Status FileUtil::RealpathUtil(const std::string& path, std::string& resolvedPath)
{
    char resolvedInputPath[PATH_MAX] = {0x00};
    if (realpath(path.c_str(), resolvedInputPath) == nullptr) {
        CANN_LLM_LOGE("get file realpath error %d", errno);
        return hiai::FAILURE;
    }

    resolvedPath = std::string(resolvedInputPath);
    return hiai::SUCCESS;
}

long FileUtil::GetFileSize(const std::string &filePath)
{
    long fileSize = 0;

    FILE* fp = OpenFile(filePath, "rb");
    fileSize = GetFileSize(fp);
    CloseFile(fp);

    return fileSize;
}

std::shared_ptr<BaseBuffer> FileUtil::LoadToBuffer(const std::string &filePath)
{
    std::shared_ptr<BaseBuffer> buffer = make_shared_nothrow<BaseBuffer>();
    HIAI_EXPECT_NOT_NULL_R(buffer, nullptr);
    HIAI_EXPECT_EXEC_R(FileUtilImpl::LoadFileToBuffer(filePath, buffer), nullptr);
    return buffer;
}

std::shared_ptr<BaseBuffer> FileUtil::LoadToBuffer(const std::string &filePath, uint64_t size)
{
    std::shared_ptr<BaseBuffer> buffer = make_shared_nothrow<BaseBuffer>();
    HIAI_EXPECT_NOT_NULL_R(buffer, nullptr);
    HIAI_EXPECT_EXEC_R(FileUtilImpl::LoadFileToBuffer(filePath, buffer, size), nullptr);
    return buffer;
}

// 当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
Status FileUtil::LoadAsUint8(const std::string filePath, uint64_t& size, uint8_t*& buffer)
{
    HIAI_EXPECT_EXEC(FileUtilImpl::LoadFileAsUint8(filePath, buffer, size));
    return hiai::SUCCESS;
}

Status FileUtil::LoadAsUint8ByOffsetAndSize(
    const std::string &filePath, const uint64_t offset, const uint64_t size, uint8_t*& data)
{
    HIAI_EXPECT_EXEC(FileUtilImpl::LoadFileAsUint8ByOffsetAndSize(filePath, data, offset, size));
    return hiai::SUCCESS;
}
} // namespace hiai
