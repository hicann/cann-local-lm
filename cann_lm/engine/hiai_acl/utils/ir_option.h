/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/utils/ir_option.h
 * @brief Build option string constants — inline (header-only).
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/utils/ir_option.h
 */
#ifndef UTILS_IR_OPTION_H
#define UTILS_IR_OPTION_H

namespace ge {
namespace ir_option {
static const char* const INPUT_SHAPE = "input_shape";
static const char* const EXEC_DEVICE_CONFIG_PATH = "ge.exec.exec_device_config_path";
static const char* const EXEC_DISABLE_REUSED_MEMORY = "ge.exec.disableReuseMemory";
static const char* const PRECISION_MODE_V2 = "ge.exec.precision_mode_v2";
} // namespace ir_option
} // namespace ge

#endif // UTILS_IR_OPTION_H
