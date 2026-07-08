#pragma once

#include <cstddef>
#include <cstdint>

namespace td {

namespace pt2 {
constexpr int64_t kDtypeByte = 1;
constexpr int64_t kDtypeChar = 2;
constexpr int64_t kDtypeShort = 3;
constexpr int64_t kDtypeInt = 4;
constexpr int64_t kDtypeLong = 5;
constexpr int64_t kDtypeHalf = 6;
constexpr int64_t kDtypeFloat = 7;
constexpr int64_t kDtypeDouble = 8;
constexpr int64_t kDtypeBool = 12;
constexpr int64_t kDtypeBFloat16 = 13;
constexpr int64_t kDtypeUInt16 = 28;

constexpr int64_t kLayoutStrided = 7;
}  // namespace pt2

enum class NodeInputKind {
  Tensor,
  TensorList,
  Operator,
  None,
  ScalarInt,
  ScalarFloat,
  ScalarBool,
  ScalarListInt,
  ScalarListFloat,
  ScalarListBool,
  ScalarString,
  Enum,
  SymInt,
  SymFloat,
  SymBool,
  SymListInt,
  SymListFloat,
  SymListBool
};

inline const char *NodeInputKindToString(NodeInputKind kind) {
  switch (kind) {
    case NodeInputKind::Tensor:
      return "Tensor";
    case NodeInputKind::TensorList:
      return "TensorList";
    case NodeInputKind::Operator:
      return "Operator";
    case NodeInputKind::None:
      return "None";
    case NodeInputKind::ScalarInt:
      return "ScalarInt";
    case NodeInputKind::ScalarFloat:
      return "ScalarFloat";
    case NodeInputKind::ScalarBool:
      return "ScalarBool";
    case NodeInputKind::ScalarListInt:
      return "ScalarListInt";
    case NodeInputKind::ScalarListFloat:
      return "ScalarListFloat";
    case NodeInputKind::ScalarListBool:
      return "ScalarListBool";
    case NodeInputKind::ScalarString:
      return "ScalarString";
    case NodeInputKind::Enum:
      return "Enum";
    case NodeInputKind::SymInt:
      return "SymInt";
    case NodeInputKind::SymFloat:
      return "SymFloat";
    case NodeInputKind::SymBool:
      return "SymBool";
    case NodeInputKind::SymListInt:
      return "SymListInt";
    case NodeInputKind::SymListFloat:
      return "SymListFloat";
    case NodeInputKind::SymListBool:
      return "SymListBool";
  }
  return "Unknown";
}

inline bool IsScalarKind(NodeInputKind kind) {
  return kind == NodeInputKind::ScalarInt || kind == NodeInputKind::ScalarFloat ||
         kind == NodeInputKind::ScalarBool || kind == NodeInputKind::ScalarListInt ||
         kind == NodeInputKind::ScalarListFloat || kind == NodeInputKind::ScalarListBool ||
         kind == NodeInputKind::ScalarString || kind == NodeInputKind::Enum;
}

inline bool IsSymKind(NodeInputKind kind) {
  return kind == NodeInputKind::SymInt || kind == NodeInputKind::SymFloat || kind == NodeInputKind::SymBool ||
         kind == NodeInputKind::SymListInt || kind == NodeInputKind::SymListFloat ||
         kind == NodeInputKind::SymListBool;
}

}  // namespace td
