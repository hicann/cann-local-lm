/**
 * cann_llm hiai_ndk: NDK so 候选搜索路径。
 */
#ifndef CANN_LLM_HIAI_NDK_PATHS_H
#define CANN_LLM_HIAI_NDK_PATHS_H
#include <string>
#include <vector>
namespace hiai_ndk {
inline std::vector<std::string> GetNdkSearchPaths()
{
    return {
        "/system/lib64/ndk",
        "/system/lib/ndk",
        "/usr/lib64",
        "/usr/lib",
        "/system/lib",
        "", // 空 = 裸库名，交给系统加载器
    };
}
inline std::string JoinPath(const std::string& dir, const std::string& libName)
{
    return dir.empty() ? libName : dir + "/" + libName;
}
} // namespace hiai_ndk
#endif
