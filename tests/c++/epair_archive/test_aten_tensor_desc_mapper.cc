#include <gtest/gtest.h>

#include <string>

#include "backend/aten_tensor_desc_mapper.h"
#include "common/logging/logging.h"
#include "common/pt2_type_constants.h"

namespace td {

class AtenTensorDescMapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    result_ = ValidationResult{true, {}};
  }

  AtenTensorDescMapper mapper_;
  ValidationResult result_;
};

TEST_F(AtenTensorDescMapperTest, MapPt2DtypeFloat) {
  ge::DataType dtype = mapper_.MapPt2Dtype(pt2::kDtypeFloat, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_FLOAT);
}

TEST_F(AtenTensorDescMapperTest, MapPt2DtypeHalf) {
  ge::DataType dtype = mapper_.MapPt2Dtype(pt2::kDtypeHalf, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_FLOAT16);
}

TEST_F(AtenTensorDescMapperTest, MapPt2DtypeInt64) {
  ge::DataType dtype = mapper_.MapPt2Dtype(pt2::kDtypeLong, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_INT64);
}

TEST_F(AtenTensorDescMapperTest, MapPt2DtypeBool) {
  ge::DataType dtype = mapper_.MapPt2Dtype(pt2::kDtypeBool, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_BOOL);
}

TEST_F(AtenTensorDescMapperTest, MapPt2DtypeUnsupported) {
  ge::DataType dtype = mapper_.MapPt2Dtype(999, result_, "test");
  EXPECT_FALSE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_UNDEFINED);
}

TEST_F(AtenTensorDescMapperTest, MapPt2LayoutStrided) {
  ge::Format fmt = mapper_.MapPt2Layout(pt2::kLayoutStrided, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(fmt, ge::FORMAT_ND);
}

TEST_F(AtenTensorDescMapperTest, MapPt2LayoutUnsupported) {
  mapper_.MapPt2Layout(0, result_, "test");
  EXPECT_FALSE(result_.ok);
}

TEST_F(AtenTensorDescMapperTest, MapGeDtypeStringFloat) {
  ge::DataType dtype = mapper_.MapGeDtypeString("DT_FLOAT", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_FLOAT);
}

TEST_F(AtenTensorDescMapperTest, MapGeDtypeStringInt64) {
  ge::DataType dtype = mapper_.MapGeDtypeString("DT_INT64", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_INT64);
}

TEST_F(AtenTensorDescMapperTest, MapGeDtypeStringBool) {
  ge::DataType dtype = mapper_.MapGeDtypeString("DT_BOOL", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_BOOL);
}

TEST_F(AtenTensorDescMapperTest, MapGeDtypeStringUnsupported) {
  ge::DataType dtype = mapper_.MapGeDtypeString("DT_COMPLEX64", result_, "test");
  EXPECT_FALSE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_UNDEFINED);
}

TEST_F(AtenTensorDescMapperTest, MapGeDtypeStringEmpty) {
  ge::DataType dtype = mapper_.MapGeDtypeString("", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(dtype, ge::DT_UNDEFINED);
}

TEST_F(AtenTensorDescMapperTest, MapGeLayoutStringND) {
  ge::Format fmt = mapper_.MapGeLayoutString("ND", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(fmt, ge::FORMAT_ND);
}

TEST_F(AtenTensorDescMapperTest, MapGeLayoutStringEmpty) {
  ge::Format fmt = mapper_.MapGeLayoutString("", result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(fmt, ge::FORMAT_ND);
}

TEST_F(AtenTensorDescMapperTest, MapGeLayoutStringUnsupported) {
  mapper_.MapGeLayoutString("NCHW", result_, "test");
  EXPECT_FALSE(result_.ok);
}

TEST_F(AtenTensorDescMapperTest, MapPt2TensorMetaBasic) {
  ParsedTensorMeta meta;
  meta.dtype = pt2::kDtypeFloat;
  meta.sizes = {2, 3};
  meta.layout = pt2::kLayoutStrided;

  GeTensorDescInfo info = mapper_.MapPt2TensorMeta(meta, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.dtype, ge::DT_FLOAT);
  ASSERT_EQ(info.shape.size(), 2u);
  EXPECT_EQ(info.shape[0], 2);
  EXPECT_EQ(info.shape[1], 3);
  EXPECT_EQ(info.format, ge::FORMAT_ND);
  EXPECT_TRUE(info.has_shape);
  EXPECT_TRUE(info.has_dtype);
  EXPECT_TRUE(info.has_format);
}

TEST_F(AtenTensorDescMapperTest, MapAscendIrDescUsesDesc) {
  ParsedAscendIrOpDesc desc;
  desc.dtype = "DT_FLOAT";
  desc.shape = {4, 4};
  desc.layout = "ND";
  desc.raw_json = OrderedJson{{"shape", OrderedJson{{"dim", {"4", "4"}}}},
                              {"dtype", "DT_FLOAT"}, {"layout", "ND"}};

  GeTensorDescInfo info = mapper_.MapAscendIrDesc(&desc, nullptr, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.dtype, ge::DT_FLOAT);
  ASSERT_EQ(info.shape.size(), 2u);
  EXPECT_EQ(info.shape[0], 4);
  EXPECT_EQ(info.shape[1], 4);
}

TEST_F(AtenTensorDescMapperTest, MapAscendIrDescFallsBackToMeta) {
  ParsedAscendIrOpDesc desc;
  desc.dtype = "";
  desc.layout = "";
  desc.shape = {};

  ParsedTensorMeta meta;
  meta.dtype = pt2::kDtypeFloat;
  meta.sizes = {1, 2, 3};
  meta.layout = pt2::kLayoutStrided;

  GeTensorDescInfo info = mapper_.MapAscendIrDesc(&desc, &meta, result_, "test");
  EXPECT_TRUE(result_.ok);
  EXPECT_EQ(info.dtype, ge::DT_FLOAT);
  ASSERT_EQ(info.shape.size(), 3u);
  EXPECT_EQ(info.shape[0], 1);
  EXPECT_EQ(info.shape[1], 2);
  EXPECT_EQ(info.shape[2], 3);
}

}  // namespace td
