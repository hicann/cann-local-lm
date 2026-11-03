# hiai_ndk：NDK dlopen 转发层（纯 NDK，零 ACL）
add_library(hiai_ndk STATIC
    src/hiai_ndk_nncore.cpp
    src/hiai_ndk_options.cpp
    src/hiai_ndk_executor.cpp
    src/hiai_ndk_tensor.cpp
    src/hiai_ndk_align.cpp
    src/ndk_proxy.cpp
)
target_include_directories(hiai_ndk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# hiai_ndk_align.cpp 源自 framework 的裸文件名 include（"hiai_ndk_align.h"），
# 零改动拷贝到 src/ 后相对查找失效，补镜像头所在目录供其解析。
target_include_directories(hiai_ndk SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/model_manager/general_model_manager/ndk/hiai_ndk
)
target_include_directories(hiai_ndk SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../backend/api
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_cann
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(hiai_ndk PUBLIC cann_backend_support ${CMAKE_DL_LIBS})
# force-include ohos_compat.h（与 hiai_ascendcl 同机制）：剥离 CANNKit 头的
# Clang `__availability__(ohos, introduced=X.Y.Z)` 属性，GCC 词法层才能解析。
target_compile_options(hiai_ndk PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/../hiai_acl/src/ohos_compat.h)
