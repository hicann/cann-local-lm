#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common/base64.h"

TEST(Base64Test, DecodesPlainString)
{
    std::string decoded;
    ASSERT_TRUE(td::internal::DecodeBase64ToString("SGVsbG8gV29ybGQ=", decoded));
    EXPECT_EQ(decoded, "Hello World");
}

TEST(Base64Test, DecodesBinaryBytes)
{
    std::vector<uint8_t> decoded;
    ASSERT_TRUE(td::internal::DecodeBase64("QlNI", decoded));
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 'B');
    EXPECT_EQ(decoded[1], 'S');
    EXPECT_EQ(decoded[2], 'H');
}

TEST(Base64Test, RejectsInvalidPayload)
{
    std::string decoded;
    EXPECT_FALSE(td::internal::DecodeBase64ToString("not_base64***", decoded));
}

TEST(Base64Test, HandlesEmptyString)
{
    std::vector<uint8_t> decoded;
    ASSERT_TRUE(td::internal::DecodeBase64("", decoded));
    EXPECT_TRUE(decoded.empty());
}

TEST(Base64Test, HandlesPadding)
{
    std::vector<uint8_t> decoded;
    ASSERT_TRUE(td::internal::DecodeBase64("QQ==", decoded));
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0], 'A');
}

TEST(Base64Test, RejectsInvalidLength)
{
    std::vector<uint8_t> decoded;
    EXPECT_FALSE(td::internal::DecodeBase64("abc", decoded));
}
