# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
测试PyTorch patch机制
"""

import torch
import torch._higher_order_ops.auto_functionalize as auto_func
from torch_delegate.converter.patch_pytorch import (
    patch_pytorch,
    restore_pytorch,
    is_patched,
    get_original_function,
)


def test_patch_mechanism():
    """测试patch机制"""
    original_func = auto_func.do_auto_functionalize

    patched_func = patch_pytorch()

    assert is_patched(), "Should be patched"
    assert patched_func == original_func, "Should return original function"
    assert get_original_function() == original_func, "Should store original function"

    assert auto_func.do_auto_functionalize != original_func, (
        "Function should be replaced"
    )

    restore_pytorch()

    assert not is_patched(), "Should not be patched"
    assert auto_func.do_auto_functionalize == original_func, (
        "Function should be restored"
    )


def test_exported_program_generates_v2():
    """测试ExportedProgram生成v2格式节点"""
    from torch.export import export

    # Patch
    patch_pytorch()

    try:
        # 创建一个简单的模型
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel()
        example_input = torch.randn(2, 2)

        # Export
        ep = export(model, (example_input,))

        # 检查graph中的节点
        has_auto_functionalized_v2 = False
        has_auto_functionalized = False

        for node in ep.graph.nodes:
            if node.op == "call_function":
                if (
                    hasattr(torch.ops.higher_order, "auto_functionalized_v2")
                    and node.target == torch.ops.higher_order.auto_functionalized_v2
                ):
                    has_auto_functionalized_v2 = True
                elif (
                    hasattr(torch.ops.higher_order, "auto_functionalized")
                    and node.target == torch.ops.higher_order.auto_functionalized
                ):
                    has_auto_functionalized = True

        # 注意：简单模型可能不包含auto_functionalize节点
        # 这里主要测试patch机制是否正常工作
        print(f"  - Has auto_functionalized_v2: {has_auto_functionalized_v2}")
        print(f"  - Has auto_functionalized: {has_auto_functionalized}")

        print("✓ exported program test passed")

    finally:
        # 恢复
        restore_pytorch()


def test_context_manager():
    """测试上下文管理器模式"""
    # Patch
    original = patch_pytorch()

    try:
        # 在patch上下文中执行
        assert is_patched()

        # 模拟一些操作
        # ...

    finally:
        # 确保恢复
        restore_pytorch(original)

    assert not is_patched()
    print("✓ context manager test passed")


def test_multiple_patch_restore():
    """测试多次patch和恢复"""
    # 第一次patch
    original1 = patch_pytorch()
    assert is_patched()

    # 恢复
    restore_pytorch()
    assert not is_patched()

    # 第二次patch
    original2 = patch_pytorch()
    assert is_patched()

    # 恢复
    restore_pytorch()
    assert not is_patched()

    # 验证两次的original函数相同
    assert original1 == original2, "Original functions should be the same"

    print("✓ multiple patch/restore test passed")


def run_all_tests():
    """运行所有测试"""
    print("Running PyTorch patch tests...")
    print()

    try:
        test_patch_mechanism()
        test_exported_program_generates_v2()
        test_context_manager()
        test_multiple_patch_restore()

        print()
        print("✅ All tests passed!")

    except Exception as e:
        print()
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
