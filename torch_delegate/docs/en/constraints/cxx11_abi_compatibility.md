# CXX11 ABI Compatibility

English | [简体中文](../../zh/constraints/cxx11_abi_compatibility.md)

## Core Concepts

The CXX11 ABI is the **breaking ABI change** introduced by GCC 5.1 (2015) to implement the new features required by the C++11 standard. The cost is **binary incompatibility** with libraries built by older GCC versions.

### The Nature of the ABI Change

ABI (Application Binary Interface) defines:

- The memory layout of data types (`sizeof`, member offsets)
- Function calling conventions (argument passing, return values)
- Symbol naming rules
- Exception handling mechanisms

The CXX11 ABI change mainly affects the internal implementation of `std::string` and `std::list`, resulting in **completely different memory layouts**.

---

## Affected API Categories

### Classification Principle

ABI compatibility issues fall into three categories:

| Category | Description | Example |
|------|------|------|
| **Directly affected** | The memory layout of the type itself changes | `std::string`, `std::list` |
| **Indirectly affected** | The type contains directly affected types as elements/members | `std::vector<std::string>` |
| **Not affected** | Neither the type nor its elements involve directly affected types | `std::vector<int>` |

---

### Directly Affected Types

#### `std::string` and Derived Types (most critical)

```cpp
// Old ABI: Copy-On-Write (COW)
// sizeof(std::string) = sizeof(char*) = 4 or 8 bytes
// Internal structure: pointer → _Rep { size, capacity, refcount, data[] }

// CXX11 ABI: Small String Optimization (SSO)
// sizeof(std::string) = typically 32 bytes (implementation-dependent)
// Internal structure: union {
//   char _M_local_buf[16];  // short strings stored inline
//   struct { char* _M_ptr; size_t _M_length; size_t _M_capacity; };  // long strings
// }
```

**Directly affected string types:**

| Type | Description |
|------|------|
| `std::string` | `basic_string<char>` |
| `std::wstring` | `basic_string<wchar_t>` |
| `std::u16string` | `basic_string<char16_t>` (C++11) |
| `std::u32string` | `basic_string<char32_t>` (C++11) |
| `std::pmr::string` | `basic_string<char, polymorphic_allocator>` (C++17) |
| `std::pmr::wstring` | `basic_string<wchar_t, polymorphic_allocator>` (C++17) |
| `std::pmr::u16string` | C++17 |
| `std::pmr::u32string` | C++17 |

**All `basic_string` member functions are affected:**

| Category | API |
|------|-----|
| **Construction/destruction** | `basic_string()`, `basic_string(const CharT*)`, `basic_string(const basic_string&)`, `basic_string(basic_string&&)`, `basic_string(size_type, CharT)`, `basic_string(InputIt, InputIt)`, `~basic_string()` |
| **Assignment** | `operator=(const basic_string&)`, `operator=(basic_string&&)`, `operator=(const CharT*)`, `operator=(CharT)`, `assign()`, `assign_range()` (C++23) |
| **Element access** | `at()`, `operator[]`, `front()`, `back()`, `data()`, `c_str()`, `operator basic_string_view()` (C++17) |
| **Iterators** | `begin()`, `cbegin()`, `end()`, `cend()`, `rbegin()`, `crbegin()`, `rend()`, `crend()` |
| **Capacity** | `empty()`, `size()`, `length()`, `max_size()`, `reserve()`, `capacity()`, `shrink_to_fit()`, `clear()` |
| **Modifiers** | `insert()`, `erase()`, `push_back()`, `pop_back()`, `append()`, `append_range()` (C++23), `operator+=()`, `replace()`, `replace_with_range()` (C++23), `copy()`, `resize()`, `resize_and_overwrite()` (C++23), `swap()` |
| **Search** | `find()`, `rfind()`, `find_first_of()`, `find_first_not_of()`, `find_last_of()`, `find_last_not_of()`, `contains()` (C++23) |
| **Comparison** | `compare()`, `starts_with()` (C++20), `ends_with()` (C++20) |
| **Substrings** | `substr()` |
| **Operators** | `operator+`, `operator==`, `operator!=`, `operator<`, `operator<=`, `operator>`, `operator>=`, `operator<<`, `operator>>` |

#### `std::list` and Derived Types

```cpp
// Old ABI: size() is O(n), traversing the whole list to count elements
// CXX11 ABI: size() is O(1), maintaining a size member variable

// Memory layout difference:
// Old ABI: _List_node { _M_data; _M_next; _M_prev; }
// New ABI: _List_node { _M_data; _M_next; _M_prev; } + _List_impl { _M_node; size_t _M_size; }
```

**Directly affected list types:**

| Type | Description |
|------|------|
| `std::list<T>` | The ABI of every `list<T>` changes |
| `std::forward_list<T>` | **Not affected** (no `size()` method, no ABI change) |
| `std::pmr::list<T>` | C++17 polymorphic allocator version |

**Affected `std::list` member functions:**

| API | Old ABI | CXX11 ABI |
|-----|--------|-----------|
| `size()` | O(n) traversal | O(1) member return |
| `max_size()` | unchanged | unchanged |
| `empty()` | unchanged | unchanged |
| `insert()` | no size update | size update required |
| `insert_range()` (C++23) | no size update | size update required |
| `emplace()` | no size update | size update required |
| `push_back()` | no size update | size update required |
| `emplace_back()` | no size update | size update required |
| `push_front()` | no size update | size update required |
| `emplace_front()` | no size update | size update required |
| `pop_back()` | no size update | size update required |
| `pop_front()` | no size update | size update required |
| `erase()` | no size update | size update required |
| `clear()` | no size update | size update required |
| `resize()` | O(n) | O(n), but must maintain size |
| `swap()` | no size update | sizes must be swapped |
| `splice()` | no size update | **complexity change**: from O(1) to O(N) (the number of spliced elements must be counted) |
| `merge()` | no size update | size update required |
| `remove()` | no size update | size update required |
| `remove_if()` | no size update | size update required |
| `unique()` | no size update | size update required |
| `sort()` | unchanged | unchanged |
| `reverse()` | unchanged | unchanged |
| `assign()` | no size update | size update required |
| `assign_range()` (C++23) | no size update | size update required |

---

### Indirectly Affected Types

#### Containers (element type contains string/list)

**The following containers are unchanged themselves, but are affected when the element type is `std::string` or `std::list`:**

| Container | Affected instantiations | Unaffected instantiations |
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

**Key points:**

- `std::unordered_map` **is not ABI-affected in itself**
- but `std::unordered_map<std::string, int>` **is affected**, because the key type is `std::string`
- `std::unordered_map<int, std::string>` **is also affected**, because the value type is `std::string`

#### pair and tuple

| Type | Affected instantiations |
|------|----------------|
| `std::pair<T1, T2>` | `pair<string, T>`, `pair<T, string>` |
| `std::tuple<Ts...>` | `tuple<string, ...>`, any element that is a string |
| `std::pair<std::list<T>, U>` | pair contains list |

#### Smart pointers

| Type | Affected instantiations |
|------|----------------|
| `std::shared_ptr<T>` | `shared_ptr<string>` |
| `std::unique_ptr<T>` | `unique_ptr<string>` |
| `std::weak_ptr<T>` | `weak_ptr<string>` |

#### Function objects

| Type | Affected instantiations |
|------|----------------|
| `std::function<F>` | `function<void(string)>`, `function<string()>` |
| `std::move_only_function<F>` (C++23) | same as above |

#### New types in C++17/20

| Type | Affected instantiations |
|------|----------------|
| `std::optional<T>` | `optional<string>` |
| `std::variant<Ts...>` | `variant<string, ...>` |
| `std::any` | **always affected** (may internally store any type, including string) |
| `std::expected<T, E>` (C++23) | `expected<string, E>`, `expected<T, string>` |

#### C++17 filesystem

| Type | Description |
|------|------|
| `std::filesystem::path` | **Directly affected** (uses string-like storage internally) |
| `std::filesystem::directory_entry` | Affected (contains a path) |
| `std::filesystem::path::iterator` | Affected |
| `std::filesystem::space_info` | Not affected (only size_t members) |
| `std::filesystem::file_status` | Not affected |
| `std::filesystem::file_type` | Not affected (enum) |
| `std::filesystem::perms` | Not affected (enum) |

#### C++11 regex

| Type | Description |
|------|------|
| `std::regex` | **Directly affected** (stores a string pattern internally) |
| `std::wregex` | **Directly affected** |
| `std::regex_error` | **Directly affected** (inherits from exception, has a string message) |
| `std::sub_match<BidirIt>` | Affected (when matching strings) |
| `std::match_results<BidirIt>` | Affected |

---

### Unaffected Types

#### Fundamental types

| Type | Description |
|------|------|
| `int`, `float`, `double`, `char`, `bool`, etc. | Fundamental type ABIs are unchanged |
| `std::complex<T>` | Not affected |
| `std::array<T, N>` | Not affected when T is not string/list |

#### Containers (elements contain no string/list)

| Container | Description |
|------|------|
| `std::vector<int>` | Not affected |
| `std::vector<double>` | Not affected |
| `std::vector<std::vector<int>>` | Not affected |
| `std::deque<int>` | Not affected |
| `std::set<int>` | Not affected |
| `std::map<int, int>` | Not affected |
| `std::unordered_map<int, int>` | **Not affected** |
| `std::unordered_set<int>` | Not affected |
| `std::forward_list<T>` | **Not affected** (no `size()` method) |

#### Other unaffected types

| Type | Description |
|------|------|
| `std::bitset<N>` | Not affected |
| `std::valarray<T>` | Not affected when T is not string |
| `std::span<T>` (C++20) | Not affected (view only) |
| `std::string_view` | **Not affected** (view only, no ownership) |
| `std::wstring_view` | Not affected |
| `std::chrono::duration` | Not affected |
| `std::chrono::time_point` | Not affected |
| `std::ratio` | Not affected |

---

### Exception Types (all affected)

**All standard exception types are affected**, because they store a `std::string` internally as the error message:

| Exception Type | Description |
|----------|------|
| `std::exception` | Base class; `what()` returns const char*, but derived classes store a string internally |
| `std::logic_error` | Stores a string directly |
| `std::runtime_error` | Stores a string directly |
| `std::invalid_argument` | Inherits from logic_error |
| `std::domain_error` | Inherits from logic_error |
| `std::length_error` | Inherits from logic_error |
| `std::out_of_range` | Inherits from logic_error |
| `std::range_error` | Inherits from runtime_error |
| `std::overflow_error` | Inherits from runtime_error |
| `std::underflow_error` | Inherits from runtime_error |
| `std::ios_base::failure` | Inherits from runtime_error |
| `std::system_error` | Inherits from runtime_error, stores a string + error_code |
| `std::future_error` | Inherits from logic_error |
| `std::bad_exception` | Not affected (no string member) |
| `std::bad_alloc` | Not affected (no string member) |
| `std::bad_cast` | Not affected (no string member) |
| `std::bad_typeid` | Not affected (no string member) |
| `std::bad_function_call` | Not affected (no string member) |
| `std::bad_weak_ptr` | Not affected (no string member) |
| `std::bad_array_new_length` | Not affected (no string member) |
| `std::filesystem::filesystem_error` (C++17) | Affected (stores a path) |
| `std::regex_error` (C++11) | Affected |
| `std::bad_optional_access` (C++17) | Not affected (no string member) |
| `std::bad_variant_access` (C++17) | Not affected (no string member) |
| `std::bad_any_cast` (C++17) | Not affected (no string member) |

---

### Affected Function Signatures

**Any function whose parameters or return value contain `std::string` or `std::list` is affected:**

```cpp
// parameters contain string
void print(const std::string& s);
void process(std::string s);
void handle(std::vector<std::string> names);

// return values contain string
std::string format(const char* fmt, ...);
std::string get_name();
std::optional<std::string> find_name(int id);
std::map<std::string, int> get_mapping();

// parameters contain list
void process_list(std::list<int>& lst);
std::list<int> generate_list();

// operators
std::string operator+(const std::string&, const std::string&);
std::ostream& operator<<(std::ostream&, const std::string&);
std::istream& operator>>(std::istream&, std::string&);
bool operator==(const std::string&, const std::string&);
bool operator<(const std::string&, const std::string&);

// template functions (affected on instantiation)
template<typename T>
void foo(T t);  // affected when T = std::string

template<typename T>
std::vector<T> make_vector();  // affected when T = std::string
```

---

## Symbol-Level Differences

### Name Mangling Changes

```cpp
// Old ABI (CXX11_ABI=0):
// std::string is encoded as "Ss" (short name)
_ZNKSs4sizeEv  // std::string::size() const

// CXX11 ABI (CXX11_ABI=1):
// std::string is encoded as "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE"
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
```

**How to verify:**

```bash
# Check which ABI a library uses
nm -D libfoo.so | grep basic_string | head -3

# Output contains "__cxx11" → CXX11 ABI (=1)
# Output does not contain "__cxx11" → old ABI (=0)
```

---

## Compatibility Problem Scenarios

### Linking libraries with different ABIs

```text
project code (ABI=1) ──link──→ libA.so (ABI=0)
                           ↓
                    libA returns std::string
                           ↓
                     project code reads it → memory layout mismatch → SIGSEGV
```

### Dynamic loading conflict

```text
program (ABI=1) ──dlopen──→ libB.so (ABI=0)
                    ↓
            libB exports std::string symbols
                    ↓
            the program's std::string calls bind to libB's implementation
                    ↓
            two different std::string implementations mixed → data corruption
```

### Python wheel libtorch problem (real TorchDelegate case)

```text
Python venv/libtorch.so (ABI=X)
    ↓ exports std::filesystem::remove_all
system libstdc++.so (ABI=Y)
    ↓ also exports std::filesystem::remove_all
    ↓
runtime linker picks the wrong implementation → SIGSEGV
```

---

## Solutions

### Unify the ABI setting

```cmake
# CMakeLists.txt
if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
    set(_GLIBCXX_USE_CXX11_ABI 1)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
```

### Check third-party library ABIs

```bash
# Check libtorch
nm -D libtorch.so | grep -E "basic_string|__cxx11" | head -5

# Check the system libstdc++
nm -D /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep __cxx11 | head -5
```

### Use a standalone libtorch distribution

torch-delegate uses the standalone libtorch instead of the Python wheel version because:

- The standalone build does not export STL symbols (avoiding conflicts with the system libstdc++)
- The ABI version is explicitly marked (the download page has a `_cxx11_abi` or `_pre_cxx11_abi` suffix)

---

## Complete List of Affected APIs

### Directly affected types

| Category | Types |
|------|------|
| **string family** | `std::string`, `std::wstring`, `std::u16string`, `std::u32string`, `std::pmr::string`, `std::pmr::wstring`, `std::pmr::u16string`, `std::pmr::u32string` |
| **list family** | `std::list<T>`, `std::pmr::list<T>` |
| **regex** | `std::regex`, `std::wregex`, `std::regex_error`, `std::sub_match`, `std::match_results` |
| **filesystem** | `std::filesystem::path`, `std::filesystem::directory_entry` |

### Indirectly affected types (elements contain string/list)

| Category | Types |
|------|------|
| **Sequence containers** | `vector<string>`, `deque<string>`, `array<string, N>` |
| **Associative containers** | `set<string>`, `multiset<string>`, `map<string, T>`, `map<K, string>`, `multimap<string, T>` |
| **Unordered containers** | `unordered_set<string>`, `unordered_multiset<string>`, `unordered_map<string, T>`, `unordered_map<K, string>`, `unordered_multimap<string, T>` |
| **Container adaptors** | `stack<string>`, `queue<string>`, `priority_queue<string>` |
| **C++23 flat containers** | `flat_set<string>`, `flat_map<string, T>` |
| **pair/tuple** | `pair<string, T>`, `tuple<string, ...>` |
| **Smart pointers** | `shared_ptr<string>`, `unique_ptr<string>`, `weak_ptr<string>` |
| **Function objects** | `function<void(string)>`, `function<string()>` |
| **C++17/20 types** | `optional<string>`, `variant<string, ...>`, `any` |
| **C++23 types** | `expected<string, E>`, `expected<T, string>` |

### Affected exception types

| Category | Types |
|------|------|
| **logic_error family** | `logic_error`, `invalid_argument`, `domain_error`, `length_error`, `out_of_range` |
| **runtime_error family** | `runtime_error`, `range_error`, `overflow_error`, `underflow_error`, `ios_base::failure`, `system_error`, `future_error` |
| **C++11/17 exceptions** | `regex_error`, `filesystem_error` |

### Unaffected types

| Category | Types |
|------|------|
| **Fundamental types** | `int`, `float`, `double`, `char`, `bool`, `complex<T>` |
| **Containers (elements contain no string/list)** | `vector<int>`, `map<int, int>`, `unordered_map<int, int>` |
| **forward_list** | `forward_list<T>` (no `size()` method, ABI unchanged) |
| **string_view** | `string_view`, `wstring_view` (view only, no ownership) |
| **span** | `span<T>` (C++20, view only) |
| **bitset** | `bitset<N>` |
| **chrono** | `duration`, `time_point`, `ratio` |
| **Exceptions without string members** | `exception`, `bad_alloc`, `bad_cast`, `bad_typeid`, `bad_function_call`, `bad_weak_ptr`, `bad_array_new_length`, `bad_optional_access`, `bad_variant_access`, `bad_any_cast` |

### Key conclusions

1. **`std::unordered_map` is not ABI-affected in itself**
2. **`std::unordered_map<std::string, int>` is affected** (key is string)
3. **`std::unordered_map<int, std::string>` is affected** (value is string)
4. **`std::unordered_map<int, int>` is not affected**
5. **`std::forward_list` is not affected** (no `size()` method)
6. **`std::string_view` is not affected** (view only, no ownership)

---

## Best Practices

1. **Unify the ABI within the project**: all translation units use the same `_GLIBCXX_USE_CXX11_ABI` value
2. **Check third-party libraries**: ensure all dependencies use the same ABI
3. **Avoid exporting STL symbols**: use `-fvisibility=hidden` or a standalone libtorch
4. **Document the ABI choice**: record the ABI used by the project in README/AGENTS.md

---

## GCC Version History and ABI Changes

### ABI changes controlled by `_GLIBCXX_USE_CXX11_ABI` (GCC 5.1+)

| GCC Version | ABI State | Description |
|----------|----------|------|
| GCC < 5.1 | Old ABI | Uses the C++98 ABI by default |
| GCC 5.1+ | New ABI | `_GLIBCXX_USE_CXX11_ABI=1` by default; can be switched back to the old ABI via the macro |

**Main changes:**

- `std::string`: COW → SSO (completely different memory layout)
- `std::list`: adds an `_M_size` member; `size()` goes from O(n) → O(1)
- Different symbol mangling: the new ABI uses the `__cxx11` prefix

### ABI issues in early GCC versions (already fixed)

The following issues existed in early GCC versions and have been fixed in modern versions:

| Issue | GCC Version Range | Status |
|------|--------------|------|
| `std::list` adds an `_M_size` member | 4.7.0 - 4.7.1 | **Fixed** (removed in 4.7.2) |
| `std::pair` move constructor non-trivial | 4.7.0 - 4.7.1 | **Fixed** (4.7.2) |
| `complex::real()`/`imag()` return type change | 4.4 - 4.7 | **Fixed** (different mangling in 4.8.0) |
| `map::erase()` etc. return type from `void` → `iterator` | 4.5 - 4.8.1 | **Fixed** (different mangling in 4.8.2) |
| `vector::data()` return type from `pointer` → `_Tp*` | 4.6+ | **Compatible** (for most allocators `pointer` is `_Tp*`) |
| `_Rb_tree_node<T>` alignment issue | 4.9.0 - 5.1.0 | **Fixed** (5.2) |

### Important conclusion

**Since GCC 5.1, the C++11 ABI has been stable and compatible with C++98** (controlled by the `_GLIBCXX_USE_CXX11_ABI` macro).

With GCC 4.x, linking C++98 and C++11 objects may be risky. Recommendations:

- Recompile all code with GCC >= 5
- Or ensure all code uses the same language version

---

## Other ABI-Related Changes

The following changes are not controlled by the `_GLIBCXX_USE_CXX11_ABI` macro, but may affect compatibility in specific scenarios:

### `std::complex` member function return types

```cpp
// GCC 4.4 - 4.7
_Tp& real();  // non-const version returns a reference
_Tp& imag();

// GCC 4.8+
_Tp real();   // non-const version returns by value
_Tp imag();
```

**Impact**: incompatible if these function signatures are exported. Merely passing/returning `std::complex` objects is safe.

### Associative container `erase()` return type

```cpp
// C++98 / GCC < 4.8.2
void erase(iterator);

// C++11 / GCC >= 4.8.2
iterator erase(const_iterator);  // returns the iterator following the erased element
```

**Impact**: the `erase()` method of `map`, `set`, `multimap`, and `multiset`. Incompatible only when the signature is exported.

### `std::vector::data()` return type

```cpp
// old versions
pointer data();  // pointer is a type defined by the allocator

// new versions (GCC 4.6+)
_Tp* data();     // returns a raw pointer directly
```

**Impact**: only a problem when the allocator's `pointer` is not `_Tp*` (rare).

### `istreambuf_iterator::reference` type

```cpp
// GCC < 4.7
typedef _CharT& reference;

// GCC >= 4.7
typedef _CharT reference;  // C++11 standard
```

**Impact**: extremely unlikely to affect real code.

---

## References

### Official documentation

| Document | Link | Description |
|------|------|------|
| Itanium C++ ABI | https://itanium-cxx-abi.github.io/cxx-abi/ | Authoritative definition of the C++ ABI specification |
| GCC Cxx11AbiCompatibility | https://gcc.gnu.org/wiki/Cxx11AbiCompatibility | Official GCC ABI compatibility notes |
| libstdc++ ABI Policy | https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html | libstdc++ ABI policy |

### Related resources

| Resource | Link | Description |
|------|------|------|
| GCC Dual ABI | https://gcc.gnu.org/gcc-5/changes.html#libstdc++ | GCC 5 dual ABI notes |
| C++ Standard Library ABI | https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html | ABI checking tools and methods |

---

## Appendix Symbol Checking Methods

### Check the ABI used by a library

```bash
# Method 1: check basic_string symbols
nm -D libfoo.so | grep basic_string | head -5

# Output contains "__cxx11" → CXX11 ABI (=1)
# Output does not contain "__cxx11" → old ABI (=0)

# Method 2: check string symbols
nm -D libfoo.so | grep -E "Ss|__cxx11" | head -5

# Ss is the short name of string under the old ABI
# __cxx11 is the identifier of the new ABI
```

### Check the list of exported symbols

```bash
# Get all exported symbols
nm -D libfoo.so | cut -f1 -d' ' | sort -u > symbols.txt

# Demangle C++ symbols
grep '^_Z' symbols.txt | c++filt | sort > demangled.txt

# Check whether std::string or std::list is involved
grep -E "std::string|std::list|basic_string" demangled.txt
```

### Check the libstdc++ ABI version

```bash
# Check which ABIs the system libstdc++ supports
nm -D /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep __cxx11 | head -5

# Check the GLIBCXX version
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5
```
