/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_UTIL_FILE_UTIL_H
#define FRAMEWORK_UTIL_FILE_UTIL_H
#include <cstdio>
#include <memory>
#include <string>
#include <climits>

#include "base_buffer.h"
#include "types.h"
#include "infra/log.h"
#include "infra/assertion.h"
namespace hiai {
// 以下常量当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
const int32_t DOMI_MAX_PATH_LEN = 256;
class FileUtil {
public:
    // 当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
    static Status RealpathUtil(const std::string& path, std::string& resolvedPath);

    static long GetFileSize(const std::string &filePath);

    // 实链通路：built_model_ndk_impl.cpp(图文件加载) / customdata_util.cpp(自定义数据)
    static std::shared_ptr<BaseBuffer> LoadToBuffer(const std::string &filePath);
    static std::shared_ptr<BaseBuffer> LoadToBuffer(const std::string &filePath, uint64_t size);

    // 当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
    static Status LoadAsUint8(const std::string filePath, uint64_t& size, uint8_t*& buffer);

    static Status LoadAsUint8ByOffsetAndSize(
        const std::string &filePath, const uint64_t offset, const uint64_t size, uint8_t*& data);

    static Status CheckFilePath(std::string& resolvedPathStr)
    {
        if (resolvedPathStr.empty()) {
            CANN_LLM_LOGE("resolvedPath is empty!");
            return FAILURE;
        }

        std::string dirName = "";
        std::string outputFileName = "";

        size_t lastSplitIndex = resolvedPathStr.find_last_of("/\\");
        if (lastSplitIndex == std::string::npos) {
            dirName = "./";
            outputFileName = resolvedPathStr;
        } else {
            dirName = resolvedPathStr.substr(0, lastSplitIndex + 1);
            outputFileName = resolvedPathStr.substr(lastSplitIndex + 1);
        }

        if (outputFileName.empty()) {
            CANN_LLM_LOGE("outputFileName is empty!");
            return FAILURE;
        }

        char resolvedOutputPath[PATH_MAX + 1] = {0x00};
        if (realpath(dirName.c_str(), resolvedOutputPath) == nullptr) {
            CANN_LLM_LOGE("invalid output file path: %s", dirName.c_str());
            return FAILURE;
        }
        resolvedPathStr = std::string(resolvedOutputPath) + "/" + outputFileName;
        return SUCCESS;
    }
    // 以下两成员当前引擎零调用（CANNKit 上游移植 API surface，保留待复用）
    static Status WriteBufferToFile(const void* data, uint32_t size, const char* file)
    {
        HIAI_EXPECT_NOT_NULL(file);
        HIAI_EXPECT_NOT_NULL(data);
        std::string resolvedPathStr = file;
        HIAI_EXPECT_EXEC(CheckFilePath(resolvedPathStr));
        FILE* fp = fopen(resolvedPathStr.c_str(), "ab");
        if (fp == nullptr) {
            CANN_LLM_LOGE("open model fail, because not found file path");
            return FAILURE;
        }

        uint32_t writeSize = static_cast<uint32_t>(fwrite(data, 1, size, fp));
        if (writeSize != size) {
            fclose(fp);
            CANN_LLM_LOGE("WriteBufferToFile ERROR: writeSize(%u) != size(%u)", writeSize, size);
            return FAILURE;
        }

        fclose(fp);
        return SUCCESS;
    }
    static Status CreateEmptyFile(const char* file)
    {
        HIAI_EXPECT_NOT_NULL(file);
        std::string resolvedPathStr = file;
        HIAI_EXPECT_EXEC(CheckFilePath(resolvedPathStr));
        FILE* fp = fopen(resolvedPathStr.c_str(), "wb");
        if (fp == nullptr) {
            CANN_LLM_LOGE("open model fail, because not found file path");
            return FAILURE;
        }

        fclose(fp);
        return SUCCESS;
    }

private:
    static FILE* OpenFile(const std::string filePath, const std::string mode)
    {
        if (filePath.empty()) {
            CANN_LLM_LOGE("filePath is null.");
            return nullptr;
        }
 
        HIAI_EXPECT_TRUE_R(filePath.size() < PATH_MAX, nullptr);
        char pathTemp[PATH_MAX + 1] = {0x00};
        if (realpath(filePath.c_str(), pathTemp) == nullptr) {
            CANN_LLM_LOGE("filePath: %s is invalid.", filePath.c_str());
            return nullptr;
        }

        return fopen(pathTemp, mode.c_str());
    }
    static void CloseFile(FILE* fp)
    {
        if (fp == nullptr) {
            return;
        }
        fclose(fp);
    }
    static long GetFileSize(FILE* fp)
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
};
} // namespace hiai

#endif
