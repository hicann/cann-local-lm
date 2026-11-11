# CXX11 ABI兼容性

## 核心概念

CXX11 ABI是GCC 5.1 （2015年） 引入的**破坏性ABI变更**，目的是实现C++11标准要求的新特性，但代价是与旧版本GCC编译的库**二进制不兼容**。

### ABI变更的本质

ABI (Application Binary Interface) 定义了：
- 数据类型的内存布局（`sizeof`，成员偏移）
- 函数调用约定 （参数传递、返回值）
- 符号命名规则
- 异常处理机制

CXX11 ABI变更主要影响 `std::string` 和 `std::list` 的内部实现，导致**内存布局完全不同**。

---

## 受影响的API分类

### 分类原则

ABI兼容性问题分为三类：

| 分类 | 说明 | 示例 |
|------|------|------|
| **直接受影响** | 类型本身的内存布局发生变化 | `std::string`, `std::list` |
| **间接受影响** | 类型包含直接受影响的类型作为元素/成员 | `std::vector<std::string>` |
| **不受影响** | 类型本身及其元素都不涉及直接受影响的类型 | `std::vector<int>` |

---

### 直接受影响的类型

#### `std::string` 及衍生类型 （最关键）

```cpp
// 旧 ABI: Copy-On-Write (COW)
// sizeof(std::string) = sizeof(char*) = 4 或 8 bytes
// 内部结构: 指针 → _Rep { size, capacity, refcount, data[] }

// CXX11 ABI: Small String Optimization (SSO)
// sizeof(std::string) = 通常 32 bytes (取决于实现)
// 内部结构: union {
//   char _M_local_buf[16];  // 短字符串直接存储
//   struct { char* _M_ptr; size_t _M_length; size_t _M_capacity; };  // 长字符串
// }
```

**直接受影响的string类型：**

| 类型 | 说明 |
|------|------|
| `std::string` | `basic_string<char>` |
| `std::wstring` | `basic_string<wchar_t>` |
| `std::u16string` | `basic_string<char16_t>` (C++11) |
| `std::u32string` | `basic_string<char32_t>` (C++11) |
| `std::pmr::string` | `basic_string<char, polymorphic_allocator>` (C++17) |
| `std::pmr::wstring` | `basic_string<wchar_t, polymorphic_allocator>` (C++17) |
| `std::pmr::u16string` | C++17 |
| `std::pmr::u32string` | C++17 |

**所有 `basic_string` 的成员函数都受影响：**

| 类别 | API |
|------|-----|
| **构造/析构** | `basic_string()`, `basic_string(const CharT*)`, `basic_string(const basic_string&)`, `basic_string(basic_string&&)`, `basic_string(size_type, CharT)`, `basic_string(InputIt, InputIt)`, `~basic_string()` |
| **赋值** | `operator=(const basic_string&)`, `operator=(basic_string&&)`, `operator=(const CharT*)`, `operator=(CharT)`, `assign()`, `assign_range()` (C++23) |
| **元素访问** | `at()`, `operator[]`, `front()`, `back()`, `data()`, `c_str()`, `operator basic_string_view()` (C++17) |
| **迭代器** | `begin()`, `cbegin()`, `end()`, `cend()`, `rbegin()`, `crbegin()`, `rend()`, `crend()` |
| **容量** | `empty()`, `size()`, `length()`, `max_size()`, `reserve()`, `capacity()`, `shrink_to_fit()`, `clear()` |
| **修改** | `insert()`, `erase()`, `push_back()`, `pop_back()`, `append()`, `append_range()` (C++23), `operator+=()`, `replace()`, `replace_with_range()` (C++23), `copy()`, `resize()`, `resize_and_overwrite()` (C++23), `swap()` |
| **查找** | `find()`, `rfind()`, `find_first_of()`, `find_first_not_of()`, `find_last_of()`, `find_last_not_of()`, `contains()` (C++23) |
| **比较** | `compare()`, `starts_with()` (C++20), `ends_with()` (C++20) |
| **子串** | `substr()` |
| **运算符** | `operator+`, `operator==`, `operator!=`, `operator<`, `operator<=`, `operator>`, `operator>=`, `operator<<`, `operator>>` |

#### `std::list` 及衍生类型

```cpp
// 旧 ABI: size() 是 O(n) 操作，遍历整个链表计数
// CXX11 ABI: size() 是 O(1) 操作，维护 size 成员变量

// 内存布局差异:
// 旧 ABI: _List_node { _M_data; _M_next; _M_prev; }
// 新 ABI: _List_node { _M_data; _M_next; _M_prev; } + _List_impl { _M_node; size_t _M_size; }
```

**直接受影响的list类型：**

| 类型 | 说明 |
|------|------|
| `std::list<T>` | 所有 `list<T>` 的ABI都变化 |
| `std::forward_list<T>` | **不受影响**（无 `size()` 方法，无ABI变化） |
| `std::pmr::list<T>` | C++17 polymorphic allocator版本 |

**受影响的 `std::list` 成员函数：**

| API | 旧ABI | CXX11 ABI |
|-----|--------|-----------|
| `size()` | O(n) 遍历 | O(1) 直接返回成员 |
| `max_size()` | 不变 | 不变 |
| `empty()` | 不变 | 不变 |
| `insert()` | 无需更新size | 需要更新size |
| `insert_range()` (C++23) | 无需更新size | 需要更新size |
| `emplace()` | 无需更新size | 需要更新size |
| `push_back()` | 无需更新size | 需要更新size |
| `emplace_back()` | 无需更新size | 需要更新size |
| `push_front()` | 无需更新size | 需要更新size |
| `emplace_front()` | 无需更新size | 需要更新size |
| `pop_back()` | 无需更新size | 需要更新size |
| `pop_front()` | 无需更新size | 需要更新size |
| `erase()` | 无需更新size | 需要更新size |
| `clear()` | 无需更新size | 需要更新size |
| `resize()` | O(n) | O(n) 但需维护size |
| `swap()` | 无需更新size | 需要交换size |
| `splice()` | 无需更新size | **复杂度变化**：从O(1) 变为O(N)（需计算被splice的元素数） |
| `merge()` | 无需更新size | 需要更新size |
| `remove()` | 无需更新size | 需要更新size |
| `remove_if()` | 无需更新size | 需要更新size |
| `unique()` | 无需更新size | 需要更新size |
| `sort()` | 不变 | 不变 |
| `reverse()` | 不变 | 不变 |
| `assign()` | 无需更新size | 需要更新size |
| `assign_range()` (C++23) | 无需更新size | 需要更新size |

---

### 间接受影响的类型

#### 容器类 (元素类型包含string/list)

**以下容器本身ABI不变，但如果元素类型是 `std::string` 或 `std::list`，则受影响：**

| 容器 | 受影响的实例化 | 不受影响的实例化 |
|------|----------------|------------------|
| `std::vector<T>` | `vector<string>`, `vector<list<int>>` | `vector<int>`, `vector<double>` |
| `std::deque<T>` | `deque<string>` | `deque<int>` |
| `std::array<T, N>` | `array<string, N>` | `array<int, N>` |
| `std::set<T>` | `set<string>` | `set<int>` |
| `std::multiset<T>` | `multiset<string>` | `multiset<int>` |
| `std::map<K, V>` | `map<string, T>`, `map<K, string>` | `map<int, int>` |
| `std::multimap<K, V>` | `multimap<string, T>` | `multimap<int, int>` |
| `std::unordered_set<T>` | `unordered_set<string>` | `unordered_set<int>` |
| `std::unordered_multiset<T>` | `unordered_multiset<string>` | `unordered_multiset<int>` |
| `std::unordered_map<K, V>` | `unordered_map<string, T>`, `unordered_map<K, string>` | `unordered_map<int, int>` |
| `std::unordered_multimap<K, V>` | `unordered_multimap<string, T>` | `unordered_multimap<int, int>` |
| `std::stack<T>` | `stack<string>` | `stack<int>` |
| `std::queue<T>` | `queue<string>` | `queue<int>` |
| `std::priority_queue<T>` | `priority_queue<string>` | `priority_queue<int>` |
| `std::flat_set<T>` (C++23) | `flat_set<string>` | `flat_set<int>` |
| `std::flat_map<K, V>` (C++23) | `flat_map<string, T>` | `flat_map<int, int>` |

**关键点：**
- `std::unordered_map` **本身不受ABI影响**
- 但 `std::unordered_map<std::string, int>` **受影响**，因为key类型是 `std::string`
- `std::unordered_map<int, std::string>` **也受影响**，因为value类型是 `std::string`

#### pair和tuple

| 类型 | 受影响的实例化 |
|------|----------------|
| `std::pair<T1, T2>` | `pair<string, T>`, `pair<T, string>` |
| `std::tuple<Ts...>` | `tuple<string, ...>`，任意元素为string |
| `std::pair<std::list<T>, U>` | pair包含list |

#### 智能指针

| 类型 | 受影响的实例化 |
|------|----------------|
| `std::shared_ptr<T>` | `shared_ptr<string>` |
| `std::unique_ptr<T>` | `unique_ptr<string>` |
| `std::weak_ptr<T>` | `weak_ptr<string>` |

#### 函数对象

| 类型 | 受影响的实例化 |
|------|----------------|
| `std::function<F>` | `function<void(string)>`, `function<string()>` |
| `std::move_only_function<F>` (C++23) | 同上 |

#### C++17/20新类型

| 类型 | 受影响的实例化 |
|------|----------------|
| `std::optional<T>` | `optional<string>` |
| `std::variant<Ts...>` | `variant<string, ...>` |
| `std::any` | **总是受影响**（内部可能存储任意类型，包括string） |
| `std::expected<T, E>` (C++23) | `expected<string, E>`, `expected<T, string>` |

#### C++17 filesystem

| 类型 | 说明 |
|------|------|
| `std::filesystem::path` | **直接受影响**（内部使用string-like存储） |
| `std::filesystem::directory_entry` | 受影响（包含path） |
| `std::filesystem::path::iterator` | 受影响 |
| `std::filesystem::space_info` | 不受影响（仅size_t成员） |
| `std::filesystem::file_status` | 不受影响 |
| `std::filesystem::file_type` | 不受影响 （枚举） |
| `std::filesystem::perms` | 不受影响 （枚举） |

#### C++11 regex

| 类型 | 说明 |
|------|------|
| `std::regex` | **直接受影响** (内部存储string pattern) |
| `std::wregex` | **直接受影响** |
| `std::regex_error` | **直接受影响** (继承exception，有string message) |
| `std::sub_match<BidirIt>` | 受影响（当匹配string时） |
| `std::match_results<BidirIt>` | 受影响 |

---

### 不受影响的类型

#### 基础类型

| 类型 | 说明 |
|------|------|
| `int`，`float`，`double`，`char`，`bool` 等 | 基础类型ABI不变 |
| `std::complex<T>` | 不受影响 |
| `std::array<T, N>` | 当T不是string/list时不受影响 |

#### 容器 (元素不含string/list)

| 容器 | 说明 |
|------|------|
| `std::vector<int>` | 不受影响 |
| `std::vector<double>` | 不受影响 |
| `std::vector<std::vector<int>>` | 不受影响 |
| `std::deque<int>` | 不受影响 |
| `std::set<int>` | 不受影响 |
| `std::map<int, int>` | 不受影响 |
| `std::unordered_map<int, int>` | **不受影响** |
| `std::unordered_set<int>` | 不受影响 |
| `std::forward_list<T>` | **不受影响** (无size() 方法) |

#### 其他不受影响的类型

| 类型 | 说明 |
|------|------|
| `std::bitset<N>` | 不受影响 |
| `std::valarray<T>` | 当T不是string时不受影响 |
| `std::span<T>` (C++20) | 不受影响 （仅视图） |
| `std::string_view` | **不受影响** （仅视图，无所有权） |
| `std::wstring_view` | 不受影响 |
| `std::chrono::duration` | 不受影响 |
| `std::chrono::time_point` | 不受影响 |
| `std::ratio` | 不受影响 |

---

### 异常类型 （全部受影响）

**所有标准异常类型都受影响**，因为它们内部存储 `std::string` 作为错误消息：

| 异常类型 | 说明 |
|----------|------|
| `std::exception` | 基类，`what()` 返回const char*，但派生类内部有string |
| `std::logic_error` | 直接存储string |
| `std::runtime_error` | 直接存储string |
| `std::invalid_argument` | 继承logic_error |
| `std::domain_error` | 继承logic_error |
| `std::length_error` | 继承logic_error |
| `std::out_of_range` | 继承logic_error |
| `std::range_error` | 继承runtime_error |
| `std::overflow_error` | 继承runtime_error |
| `std::underflow_error` | 继承runtime_error |
| `std::ios_base::failure` | 继承runtime_error |
| `std::system_error` | 继承runtime_error，存储string + error_code |
| `std::future_error` | 继承logic_error |
| `std::bad_exception` | 不受影响（无string成员） |
| `std::bad_alloc` | 不受影响（无string成员） |
| `std::bad_cast` | 不受影响（无string成员） |
| `std::bad_typeid` | 不受影响（无string成员） |
| `std::bad_function_call` | 不受影响（无string成员） |
| `std::bad_weak_ptr` | 不受影响（无string成员） |
| `std::bad_array_new_length` | 不受影响（无string成员） |
| `std::filesystem::filesystem_error` (C++17) | 受影响（存储path） |
| `std::regex_error` (C++11) | 受影响 |
| `std::bad_optional_access` (C++17) | 不受影响（无string成员） |
| `std::bad_variant_access` (C++17) | 不受影响（无string成员） |
| `std::bad_any_cast` (C++17) | 不受影响（无string成员） |

---

### 受影响的函数签名

**任何函数参数或返回值包含 `std::string` 或 `std::list` 的都受影响：**

```cpp
// 参数包含 string
void print(const std::string& s);
void process(std::string s);
void handle(std::vector<std::string> names);

// 返回值包含 string
std::string format(const char* fmt, ...);
std::string get_name();
std::optional<std::string> find_name(int id);
std::map<std::string, int> get_mapping();

// 参数包含 list
void process_list(std::list<int>& lst);
std::list<int> generate_list();

// 运算符
std::string operator+(const std::string&, const std::string&);
std::ostream& operator<<(std::ostream&, const std::string&);
std::istream& operator>>(std::istream&, std::string&);
bool operator==(const std::string&, const std::string&);
bool operator<(const std::string&, const std::string&);

// 模板函数 (实例化时受影响)
template<typename T>
void foo(T t);  // 当 T = std::string 时受影响

template<typename T>
std::vector<T> make_vector();  // 当 T = std::string 时受影响
```

---

## 符号层面的差异

### Name Mangling变化

```cpp
// 旧 ABI (CXX11_ABI=0):
// std::string 被编码为 "Ss" (short name)
_ZNKSs4sizeEv  // std::string::size() const

// CXX11 ABI (CXX11_ABI=1):
// std::string 被编码为 "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE"
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
```

**验证方法：**

```bash
# 检查库使用的 ABI
nm -D libfoo.so | grep basic_string | head -3

# 输出含 "__cxx11" → CXX11 ABI (=1)
# 输出不含 "__cxx11" → 旧 ABI (=0)
```

---

## 兼容性问题场景

### 链接不同ABI的库

```text
项目代码 (ABI=1) ──链接──→ libA.so (ABI=0)
                           ↓
                    libA 返回 std::string
                           ↓
                    项目代码读取 → 内存布局不匹配 → SIGSEGV
```

### 动态加载冲突

```text
程序 (ABI=1) ──dlopen──→ libB.so (ABI=0)
                    ↓
            libB 导出 std::string 符号
                    ↓
            程序的 std::string 调用链接到 libB 的实现
                    ↓
            两份不同实现的 std::string 混用 → 数据损坏
```

### Python wheel libtorch问题 (torch-delegate实际案例)

```text
Python venv/libtorch.so (ABI=X)
    ↓ 导出 std::filesystem::remove_all
系统 libstdc++.so (ABI=Y)
    ↓ 也导出 std::filesystem::remove_all
    ↓
运行时链接器选择错误实现 → SIGSEGV
```

---

## 解决方案

### 统一ABI设置

```cmake
# CMakeLists.txt
if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
    set(_GLIBCXX_USE_CXX11_ABI 1)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
```

### 检查第三方库ABI

```bash
# 检查 libtorch
nm -D libtorch.so | grep -E "basic_string|__cxx11" | head -5

# 检查系统 libstdc++
nm -D /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep __cxx11 | head -5
```

### 使用独立libtorch分发包

torch-delegate使用独立libtorch而非Python wheel版，原因是：
- 独立版本不导出STL符号（避免与系统libstdc++冲突）
- 明确标注ABI版本（下载页面有 `_cxx11_abi` 或 `_pre_cxx11_abi` 后缀）

---

## 完整受影响API清单

### 直接受影响的类型

| 类别 | 类型 |
|------|------|
| **string系列** | `std::string`, `std::wstring`, `std::u16string`, `std::u32string`, `std::pmr::string`, `std::pmr::wstring`, `std::pmr::u16string`, `std::pmr::u32string` |
| **list系列** | `std::list<T>`, `std::pmr::list<T>` |
| **regex** | `std::regex`, `std::wregex`, `std::regex_error`, `std::sub_match`, `std::match_results` |
| **filesystem** | `std::filesystem::path`, `std::filesystem::directory_entry` |

### 间接受影响的类型 (元素包含string/list)

| 类别 | 类型 |
|------|------|
| **序列容器** | `vector<string>`, `deque<string>`, `array<string, N>` |
| **关联容器** | `set<string>`, `multiset<string>`, `map<string, T>`, `map<K, string>`, `multimap<string, T>` |
| **无序容器** | `unordered_set<string>`, `unordered_multiset<string>`, `unordered_map<string, T>`, `unordered_map<K, string>`, `unordered_multimap<string, T>` |
| **容器适配器** | `stack<string>`, `queue<string>`, `priority_queue<string>` |
| **C++23 flat容器** | `flat_set<string>`, `flat_map<string, T>` |
| **pair/tuple** | `pair<string, T>`, `tuple<string, ...>` |
| **智能指针** | `shared_ptr<string>`, `unique_ptr<string>`, `weak_ptr<string>` |
| **函数对象** | `function<void(string)>`, `function<string()>` |
| **C++17/20类型** | `optional<string>`, `variant<string, ...>`, `any` |
| **C++23类型** | `expected<string, E>`, `expected<T, string>` |

### 受影响的异常类型

| 类别 | 类型 |
|------|------|
| **logic_error系列** | `logic_error`, `invalid_argument`, `domain_error`, `length_error`, `out_of_range` |
| **runtime_error系列** | `runtime_error`, `range_error`, `overflow_error`, `underflow_error`, `ios_base::failure`, `system_error`, `future_error` |
| **C++11/17异常** | `regex_error`, `filesystem_error` |

### 不受影响的类型

| 类别 | 类型 |
|------|------|
| **基础类型** | `int`, `float`, `double`, `char`, `bool`, `complex<T>` |
| **容器 (元素不含string/list)** | `vector<int>`, `map<int, int>`, `unordered_map<int, int>` |
| **forward_list** | `forward_list<T>` (无size() 方法，ABI不变) |
| **string_view** | `string_view`，`wstring_view` （仅视图，无所有权） |
| **span** | `span<T>`（C++20，仅视图） |
| **bitset** | `bitset<N>` |
| **chrono** | `duration`, `time_point`, `ratio` |
| **无string成员的异常** | `exception`, `bad_alloc`, `bad_cast`, `bad_typeid`, `bad_function_call`, `bad_weak_ptr`, `bad_array_new_length`, `bad_optional_access`, `bad_variant_access`, `bad_any_cast` |

### 关键结论

1. **`std::unordered_map` 本身不受ABI影响**
2. **`std::unordered_map<std::string, int>` 受影响** (key是string)
3. **`std::unordered_map<int, std::string>` 受影响** (value是string)
4. **`std::unordered_map<int, int>` 不受影响**
5. **`std::forward_list` 不受影响** (无size() 方法)
6. **`std::string_view` 不受影响** （仅视图，无所有权）

---

## 最佳实践

1. **项目内统一ABI**：所有编译单元使用相同的 `_GLIBCXX_USE_CXX11_ABI` 值
2. **检查第三方库**：确保所有依赖库使用相同ABI
3. **避免STL符号导出**：使用 `-fvisibility=hidden` 或独立libtorch
4. **文档化ABI选择**：在README/AGENTS.md中明确记录项目使用的ABI

---

## GCC版本历史与ABI变化

### `_GLIBCXX_USE_CXX11_ABI` 控制的ABI变化 (GCC 5.1+)

| GCC版本 | ABI状态 | 说明 |
|----------|----------|------|
| GCC < 5.1 | 旧ABI | 默认使用C++98 ABI |
| GCC 5.1+ | 新ABI | 默认 `_GLIBCXX_USE_CXX11_ABI=1`，可通过宏切换回旧ABI |

**主要变化：**
- `std::string`： COW → SSO （内存布局完全不同）
- `std::list`： 添加 `_M_size` 成员，`size()` 从O(n) → O(1)
- 符号mangle不同：新ABI使用 `__cxx11` 前缀

### GCC早期版本的ABI问题 （已修复）

以下问题在GCC早期版本中存在，但在现代版本中已修复：

| 问题 | GCC版本范围 | 状态 |
|------|--------------|------|
| `std::list` 添加 `_M_size` 成员 | 4.7.0 - 4.7.1 | **已修复** （4.7.2移除） |
| `std::pair` move constructor non-trivial | 4.7.0 - 4.7.1 | **已修复** (4.7.2) |
| `complex::real()`/`imag()` 返回类型变化 | 4.4 - 4.7 | **已修复** (4.8.0 mangle不同) |
| `map::erase()` 等返回类型从 `void` → `iterator` | 4.5 - 4.8.1 | **已修复** (4.8.2 mangle不同) |
| `vector::data()` 返回类型从 `pointer` → `_Tp*` | 4.6+ | **兼容**（大多数allocator的 `pointer` 就是 `_Tp*`） |
| `_Rb_tree_node<T>` alignment问题 | 4.9.0 - 5.1.0 | **已修复** (5.2) |

### 重要结论

**自GCC 5.1起，C++11 ABI已稳定并与C++98兼容**（通过 `_GLIBCXX_USE_CXX11_ABI` 宏控制）。

如果使用GCC 4.x版本，链接C++98和C++11对象可能存在风险。建议：
- 重新编译所有代码使用GCC >= 5
- 或确保所有代码使用相同语言版本

---

## 其他ABI相关变化

以下变化不影响 `_GLIBCXX_USE_CXX11_ABI` 宏控制的ABI，但在特定场景下可能影响兼容性：

### `std::complex` 成员函数返回类型

```cpp
// GCC 4.4 - 4.7
_Tp& real();  // 非 const 版本返回引用
_Tp& imag();

// GCC 4.8+
_Tp real();   // 非 const 版本返回值
_Tp imag();
```

**影响**：如果导出这些函数签名，则不兼容。仅传递/返回 `std::complex` 对象则安全。

### 关联容器 `erase()` 返回类型

```cpp
// C++98 / GCC < 4.8.2
void erase(iterator);

// C++11 / GCC >= 4.8.2
iterator erase(const_iterator);  // 返回删除后的迭代器
```

**影响**：`map`，`set`，`multimap`，`multiset` 的 `erase()` 方法。仅导出签名时不兼容。

### `std::vector::data()` 返回类型

```cpp
// 旧版本
pointer data();  // pointer 是 allocator 定义的类型

// 新版本 (GCC 4.6+)
_Tp* data();     // 直接返回原始指针
```

**影响**：仅当allocator的 `pointer` 不是 `_Tp*` 时才有问题（罕见）。

### `istreambuf_iterator::reference` 类型

```cpp
// GCC < 4.7
typedef _CharT& reference;

// GCC >= 4.7
typedef _CharT reference;  // C++11 标准
```

**影响**：极低概率影响实际代码。

---

## 参考链接

### 官方文档

| 文档 | 链接 | 说明 |
|------|------|------|
| Itanium C++ ABI | https://itanium-cxx-abi.github.io/cxx-abi/ | C++ ABI规范的权威定义 |
| GCC Cxx11AbiCompatibility | https://gcc.gnu.org/wiki/Cxx11AbiCompatibility | GCC官方ABI兼容性说明 |
| libstdc++ ABI Policy | https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html | libstdc++ ABI政策 |

### 相关资源

| 资源 | 链接 | 说明 |
|------|------|------|
| GCC Dual ABI | https://gcc.gnu.org/gcc-5/changes.html#libstdc++ | GCC 5双ABI说明 |
| C++ Standard Library ABI | https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html | ABI检查工具和方法 |

---

## 附录 符号检查方法

### 检查库使用的ABI

```bash
# 方法 1: 检查 basic_string 符号
nm -D libfoo.so | grep basic_string | head -5

# 输出含 "__cxx11" → CXX11 ABI (=1)
# 输出不含 "__cxx11" → 旧 ABI (=0)

# 方法 2: 检查 string 符号
nm -D libfoo.so | grep -E "Ss|__cxx11" | head -5

# Ss 是旧 ABI 的 string 短名称
# __cxx11 是新 ABI 的标识符
```

### 检查导出的符号列表

```bash
# 获取所有导出符号
nm -D libfoo.so | cut -f1 -d' ' | sort -u > symbols.txt

# 解析 C++ 符号
grep '^_Z' symbols.txt | c++filt | sort > demangled.txt

# 检查是否涉及 std::string 或 std::list
grep -E "std::string|std::list|basic_string" demangled.txt
```

### 检查libstdc++ ABI版本

```bash
# 检查系统 libstdc++ 支持的 ABI
nm -D /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep __cxx11 | head -5

# 检查 GLIBCXX 版本
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5
```
