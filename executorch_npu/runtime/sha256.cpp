/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "executorch_npu/sha256.h"

#include <array>
#include <iomanip>
#include <memory>
#include <sstream>

#include <openssl/evp.h>

namespace executorch_npu {

bool Sha256(const void *data, size_t size, std::string *digest, std::string *error) {
  if (data == nullptr || digest == nullptr) {
    if (error != nullptr) {
      *error = "SHA-256 input and output must be non-null";
    }
    return false;
  }
  using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  Context context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
  unsigned int output_size = 0;
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), data, size) != 1 ||
      EVP_DigestFinal_ex(context.get(), output.data(), &output_size) != 1 || output_size != 32) {
    if (error != nullptr) {
      *error = "OpenSSL failed to compute SHA-256";
    }
    return false;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (size_t index = 0; index < output_size; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(output[index]);
  }
  *digest = stream.str();
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace executorch_npu
