/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <acl/acl_rt.h>

#include "cann_stub/cann_stub_control.h"

// ---------------------------------------------------------------------------
// GTest fixture
// ---------------------------------------------------------------------------

class CannStubTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cann_stub::Reset();
  }
  void TearDown() override {
    EXPECT_EQ(cann_stub::LiveStreamCount(), 0u) << "Stream leak detected";
    EXPECT_EQ(cann_stub::LiveAllocCount(), 0u) << "Memory leak detected";
  }
};

// ---------------------------------------------------------------------------
// aclInit / aclFinalize
// ---------------------------------------------------------------------------

TEST_F(CannStubTest, InitSetsInitializedFlag) {
  EXPECT_FALSE(cann_stub::IsInitialized());
  EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
  EXPECT_TRUE(cann_stub::IsInitialized());
}

TEST_F(CannStubTest, FinalizeClearsInitializedFlag) {
  ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
  EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
  EXPECT_FALSE(cann_stub::IsInitialized());
}

TEST_F(CannStubTest, InitErrorInjection) {
  cann_stub::SetNextError(1);
  EXPECT_EQ(aclInit(nullptr), 1);
  EXPECT_FALSE(cann_stub::IsInitialized());
}

TEST_F(CannStubTest, FinalizeErrorInjection) {
  ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
  cann_stub::SetNextError(2);
  EXPECT_EQ(aclFinalize(), 2);
  // State should be unchanged on error
  EXPECT_TRUE(cann_stub::IsInitialized());
}

// ---------------------------------------------------------------------------
// aclrtSetDevice / aclrtResetDevice
// ---------------------------------------------------------------------------

TEST_F(CannStubTest, SetDeviceRecordsDeviceId) {
  EXPECT_EQ(cann_stub::CurrentDevice(), -1);
  EXPECT_EQ(aclrtSetDevice(3), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::CurrentDevice(), 3);
}

TEST_F(CannStubTest, ResetDeviceClearsDeviceId) {
  ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
  EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::CurrentDevice(), -1);
}

TEST_F(CannStubTest, SetDeviceErrorInjection) {
  cann_stub::SetNextError(5);
  EXPECT_EQ(aclrtSetDevice(1), 5);
  EXPECT_EQ(cann_stub::CurrentDevice(), -1);
}

// ---------------------------------------------------------------------------
// aclrtCreateStream / aclrtDestroyStream / aclrtSynchronizeStream
// ---------------------------------------------------------------------------

TEST_F(CannStubTest, CreateStreamReturnsNonNullHandle) {
  aclrtStream stream = nullptr;
  EXPECT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
  EXPECT_NE(stream, nullptr);
  EXPECT_EQ(cann_stub::LiveStreamCount(), 1u);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

TEST_F(CannStubTest, StreamLifecycle) {
  aclrtStream s1 = nullptr;
  aclrtStream s2 = nullptr;
  ASSERT_EQ(aclrtCreateStream(&s1), ACL_SUCCESS);
  ASSERT_EQ(aclrtCreateStream(&s2), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveStreamCount(), 2u);

  EXPECT_EQ(aclrtDestroyStream(s1), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveStreamCount(), 1u);
  EXPECT_EQ(aclrtDestroyStream(s2), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveStreamCount(), 0u);
}

TEST_F(CannStubTest, SynchronizeStreamSucceeds) {
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
  EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

TEST_F(CannStubTest, CreateStreamErrorInjection) {
  cann_stub::SetNextError(7);
  aclrtStream stream = nullptr;
  EXPECT_EQ(aclrtCreateStream(&stream), 7);
  EXPECT_EQ(cann_stub::LiveStreamCount(), 0u);
}

TEST_F(CannStubTest, DestroyStreamErrorInjection) {
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);

  cann_stub::SetNextError(8);
  EXPECT_EQ(aclrtDestroyStream(stream), 8);
  // Stream still tracked; destroy without error to avoid leak in TearDown
  EXPECT_EQ(cann_stub::LiveStreamCount(), 1u);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

// ---------------------------------------------------------------------------
// aclrtMalloc / aclrtFree
// ---------------------------------------------------------------------------

TEST_F(CannStubTest, MallocReturnsNonNullPointer) {
  void *ptr = nullptr;
  EXPECT_EQ(aclrtMalloc(&ptr, 64, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(cann_stub::LiveAllocCount(), 1u);
  ASSERT_EQ(aclrtFree(ptr), ACL_SUCCESS);
}

TEST_F(CannStubTest, MallocTracksSize) {
  void *ptr = nullptr;
  ASSERT_EQ(aclrtMalloc(&ptr, 128, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
  auto sizes = cann_stub::LiveAllocSizes();
  ASSERT_EQ(sizes.size(), 1u);
  EXPECT_EQ(sizes[0], 128u);
  ASSERT_EQ(aclrtFree(ptr), ACL_SUCCESS);
}

TEST_F(CannStubTest, AllocLifecycle) {
  void *p1 = nullptr;
  void *p2 = nullptr;
  ASSERT_EQ(aclrtMalloc(&p1, 32, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
  ASSERT_EQ(aclrtMalloc(&p2, 256, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveAllocCount(), 2u);

  EXPECT_EQ(aclrtFree(p1), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveAllocCount(), 1u);
  EXPECT_EQ(aclrtFree(p2), ACL_SUCCESS);
  EXPECT_EQ(cann_stub::LiveAllocCount(), 0u);
}

TEST_F(CannStubTest, MallocMemoryIsReadWrite) {
  void *ptr = nullptr;
  ASSERT_EQ(aclrtMalloc(&ptr, sizeof(int), ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
  // Verify the pointer is actually usable (ASAN catches overflows here)
  *static_cast<int *>(ptr) = 42;
  EXPECT_EQ(*static_cast<int *>(ptr), 42);
  ASSERT_EQ(aclrtFree(ptr), ACL_SUCCESS);
}

TEST_F(CannStubTest, MallocErrorInjection) {
  cann_stub::SetNextError(9);
  void *ptr = nullptr;
  EXPECT_EQ(aclrtMalloc(&ptr, 64, ACL_MEM_MALLOC_NORMAL_ONLY), 9);
  EXPECT_EQ(cann_stub::LiveAllocCount(), 0u);
}

TEST_F(CannStubTest, FreeErrorInjection) {
  void *ptr = nullptr;
  ASSERT_EQ(aclrtMalloc(&ptr, 64, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);

  cann_stub::SetNextError(10);
  EXPECT_EQ(aclrtFree(ptr), 10);
  // Allocation still tracked; free without error to avoid leak in TearDown
  EXPECT_EQ(cann_stub::LiveAllocCount(), 1u);
  ASSERT_EQ(aclrtFree(ptr), ACL_SUCCESS);
}

// ---------------------------------------------------------------------------
// Error injection is one-shot
// ---------------------------------------------------------------------------

TEST_F(CannStubTest, ErrorInjectionConsumedAfterOneCall) {
  cann_stub::SetNextError(99);
  EXPECT_EQ(aclInit(nullptr), 99);
  // Second call must succeed (injection was consumed)
  EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
  ASSERT_EQ(aclFinalize(), ACL_SUCCESS);
}
