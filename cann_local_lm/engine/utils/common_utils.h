#ifndef CANN_LM_ENGINE_UTILS_COMMON_UTILS_H
#define CANN_LM_ENGINE_UTILS_COMMON_UTILS_H

#include <sstream>
#include <string>
#include <vector>

namespace cann_local_lm::engine {

// 将 vector 内容转为逗号分隔的字符串，用于日志打印
template <typename T>
std::string CvtVectorToStr(std::vector<T> vec)
{
    std::stringstream ss;
    for (auto& item : vec) {
        ss << item << ",";
    }
    return ss.str();
}

} // namespace cann_local_lm::engine

#endif // CANN_LM_ENGINE_UTILS_COMMON_UTILS_H
