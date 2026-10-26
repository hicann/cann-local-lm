# 回退指南

如果需要回退本次编译工程优化，可以按照以下步骤操作：

## 回退头文件移动

如果需要将对外头文件恢复到原始位置：

1. **恢复原始头文件位置**
   ```bash
   # 恢复backend目录头文件
   cp csrc/inc/external/torch_delegate/delegate_session.h csrc/backend/
   cp csrc/inc/external/torch_delegate/epair_archive.h csrc/backend/
   cp csrc/inc/external/torch_delegate/status.h csrc/backend/

   # 恢复common目录头文件
   cp csrc/inc/external/torch_delegate/common/epair_archive_reader.h csrc/common/

   # 删除external目录
   rm -rf csrc/inc/external/
   ```

2. **恢复include路径**
   ```bash
   # 恢复backend目录include路径
   sed -i 's|"torch_delegate/epair_archive.h"|"backend/epair_archive.h"|g; s|"torch_delegate/status.h"|"backend/status.h"|g; s|"torch_delegate/delegate_session.h"|"backend/delegate_session.h"|g' csrc/backend/*.cc csrc/backend/*.h

   # 恢复common目录include路径
   sed -i 's|"torch_delegate/common/epair_archive_reader.h"|"common/epair_archive_reader.h"|g' csrc/common/*.cc csrc/common/*.h

   # 恢复其他目录include路径
   sed -i 's|"torch_delegate/epair_archive.h"|"backend/epair_archive.h"|g; s|"torch_delegate/status.h"|"backend/status.h"|g; s|"torch_delegate/delegate_session.h"|"backend/delegate_session.h"|g; s|"torch_delegate/common/epair_archive_reader.h"|"common/epair_archive_reader.h"|g' csrc/compatible/*.cc csrc/compatible/*.h csrc/fallback_codegen/*.cc csrc/fallback_codegen/*.h csrc/hetero_codegen/*.cc csrc/hetero_codegen/*.h csrc/partitioner/*.cc csrc/partitioner/*.h tests/cpp/*.cc
   ```

3. **恢复头文件中的include路径**
   ```bash
   # 恢复delegate_session.h
   sed -i 's|"torch_delegate/epair_archive.h"|"backend/epair_archive.h"|g; s|"torch_delegate/status.h"|"backend/status.h"|g' csrc/backend/delegate_session.h

   # 恢复epair_archive.h
   sed -i 's|"torch_delegate/common/epair_archive_reader.h"|"common/epair_archive_reader.h"|g' csrc/backend/epair_archive.h
   ```

## 回退CMakeLists.txt修改

如果需要恢复原始CMakeLists.txt：

1. **恢复CMakeLists.txt**
   ```bash
   git checkout CMakeLists.txt
   ```

2. **恢复csrc/CMakeLists.txt**
   ```bash
   git checkout csrc/CMakeLists.txt
   ```

3. **恢复tests/CMakeLists.txt**
   ```bash
   git checkout tests/CMakeLists.txt
   ```

## 回退构建脚本

如果需要删除新增的构建脚本：

```bash
rm build.sh build_test.sh
rm cmake/FindTorchDelegate.cmake
rm cmake/TorchDelegatePackage.cmake
rm requirements.txt
```

## 回退文档更新

如果需要恢复原始docs/build.md：

```bash
git checkout docs/build.md
```

## 回退build_third_party.sh修改

如果需要恢复原始build_third_party.sh：

```bash
git checkout build_third_party.sh
git checkout cmake/third_party/build/CMakeLists.txt
git checkout cmake/third_party/modules/Findlibtorch_cpu.cmake
```

## 完整回退

如果需要完整回退所有修改：

```bash
# 回退所有git跟踪的文件
git checkout -- .

# 删除新增的文件
rm -rf csrc/inc/external/
rm build.sh build_test.sh
rm cmake/FindTorchDelegate.cmake
rm cmake/TorchDelegatePackage.cmake
rm requirements.txt
```

## 注意事项

- 回退前请确保备份当前工作
- 回退后需要重新编译验证
- 如果有未提交的修改，请先stash或commit
