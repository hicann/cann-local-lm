#pragma once

// engine 内部 tokenizer 模块（SR20260728001 v0.7，per docs/tokenizer/ 与
// .project/AGENTS/tokenizer/）。三路派生：
//   BPETokenizer   —— 手写 BPE，词表 HF tokenizer.json（model.type=BPE），服务 gemma4；
//   QwenTokenizer  —— 手写 ByteLevel BPE（D-10 正则），词表 tokenizer.json，服务 Qwen3.5；
//   SPMTokenizer   —— sentencepiece 库路径占位桩（注册但 create 恒 FAILURE）。
// 错误归一（v0.4 约束 #1）：不返回模块自有错误码，统一用项目 Status（infra/status.h）；
// spec §4.1.9 类目→Status 唯一权威映射见 DESIGN M-01 §2（LoadFailed→FAILURE、
// UnsupportedAlgorithm/InvalidInput→INVALID_ARGUMENT、OutOfVocab→SUCCESS 内部映射 unk、
// RegistrationConflict→开发期 WARN 无返回值）。encode 不返回 attention mask（REQ-04：
// 不提供 mask 查询接口即职责边界实现）。词表产物经 LoadParams 单一 artifact 槽
// （文件路径或内存内容，H1 裁决）；派生经参数化构造函数在构造内完成加载（H4 裁决）。

#include "infra/status.h"

#include <cstddef> // size_t, std::byte
#include <filesystem>
#include <map>    // merges rank 表（派生私有成员）
#include <memory> // unique_ptr
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility> // pair
#include <variant>
#include <vector>

namespace cann_local_lm::engine {

// ── 算法枚举（REQ-06）。v0.7：BPE=手写 BPE（tokenizer.json，gemma4）；
// Qwen=手写 ByteLevel BPE（tokenizer.json，Qwen3.5）；SPM=sentencepiece 依赖的
// 占位桩（注册但 not-ready；将来接库时填充实现）。─────────────────────────────
enum class Algorithm { BPE, Qwen, SPM };

// ── 编码选项（REQ-02）──────────────────────────────────────────────────────
struct EncodeOptions {
    bool add_special_tokens = true;
    std::optional<size_t> max_length; // 截断（special 计入预算）
    enum class Padding { None, Longest, Fixed } padding = Padding::None;
    std::optional<size_t> pad_to_length; // Padding::Fixed 时生效
};

// ── 解码选项（REQ-03）──────────────────────────────────────────────────────
struct DecodeOptions {
    bool skip_special_tokens = true;
};

// ── tokenizer 唯一配置（SR20260903001：并入原 SpecialTokenConfig 与 LoadParams，
//    D-13/D-14——单一结构承载词表产物路径 + special 配置，直通加载与注入通道，零中间转换）──
// 来源：meta_json tokenizer_config 段（tokenizer_file/bos/eos/special_tokens，
// docs/tokenizer/tokenizer_config.md）+ 跨段 pad_token_id（model_config 段）。
struct TokenizerConfig {
    // —— 词表产物（原 LoadParams.artifact 职责，路径形态）——
    std::string tokenizer_file; // tokenizer.json 路径；空 = 不加载（未就绪）
    // —— id 语义（-1 = 未配置；吸收原 SpecialTokenConfig 的 optional 语义为哨兵值）——
    int bos_token_id = -1; // JSON bos_token_id
    int eos_token_id = -1; // JSON eos_token_id
    int pad_token_id = -1; // 跨段：model_config.pad_token_id
    int unk_token_id = -1; // 承载（schema 无来源，保留通道）
    // —— 注入开关（零注入缺省）——
    bool add_bos = false;
    bool add_eos = false;
    // —— 字面量承载（不参与注入判定）——
    std::string bos_token;
    std::string eos_token;
    // —— 附加 special 映射（token 字面量 → id；可空）——
    std::unordered_map<std::string, int> special_tokens;
};

// ── 多态基类（薄接口，REQ-02/03/04/05/08）──────────────────────────────────
// 错误统一返回 Status；OOV 非失败（内部映射 unk 后 SUCCESS）。
class TokenizerBase {
public:
    virtual ~TokenizerBase() = default;
    // 同步、阻塞至返回；encode 不返回 attention mask（REQ-04）
    // 错误: INVALID_ARGUMENT(非法 UTF-8 或 out_ids==nullptr) | SUCCESS(OOV 映射 unk)
    virtual Status encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids) = 0;
    // 同步；skip_special 时剥离控制 token（剥离集 = 产物 special 全集）
    // 错误: SUCCESS(OOV 跳过/替换，不中断)
    virtual Status decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text) = 0;
    virtual size_t vocab_size() const = 0;
    // 按名称返回 special id，未配置返回 nullopt
    virtual std::optional<int> special_token_id(std::string_view name) const = 0;
    // 默认实现：循环单条 encode + padding 对齐（REQ-05）；单条失败不整批崩
    virtual Status batch_encode(const std::vector<std::string>& texts, const EncodeOptions& opts,
                                std::vector<std::vector<int>>* out);
};

// ── 工厂（REQ-06/07）───────────────────────────────────────────────────────
class TokenizerFactory {
public:
    // 查注册表 → 参数化构造派生（构造内经 cfg.tokenizer_file 完成词表加载）→ 返回就绪实例
    // tokenizer_file 空 → 实例未就绪（词表 0，encode FAILURE）；
    // 文件不可加载 → FAILURE（不发布实例，调用方决定置空降级）
    // 错误: INVALID_ARGUMENT(枚举未注册或 out==nullptr) | FAILURE(加载失败，不发布实例)
    static Status create(Algorithm algo, const TokenizerConfig& cfg, std::unique_ptr<TokenizerBase>* out);
    // 返回当前已注册算法集合（静态初始化期已填；本期 = {BPE, Qwen, SPM}）
    static std::vector<Algorithm> list_registered();
};

// ── 自注册机制（REQ-07 开闭；注册宏须与 create 同 TU，见 tokenizer.cpp）─────
class Registrar {
public:
    using Ctor = std::unique_ptr<TokenizerBase> (*)(const TokenizerConfig&, Status*);
    // 静态初始化期调用；重复注册拒绝 + WARN 日志（无返回值），保留先注册者
    static void register_(Algorithm algo, Ctor ctor);
};

#define REGISTER_TOKENIZER(algo, Class)                                                                          \
    static const bool _reg_##Class = [] {                                                                        \
        ::cann_local_lm::engine::Registrar::register_(                                                           \
            algo,                                                                                                \
            [](const ::cann_local_lm::engine::TokenizerConfig& cfg,                                              \
               ::cann_local_lm::engine::Status* st) -> std::unique_ptr<::cann_local_lm::engine::TokenizerBase> { \
                return std::make_unique<Class>(cfg, st);                                                         \
            });                                                                                                  \
        return true;                                                                                             \
    }()

// ── M-05 VocabStore：tokenizer.json 解析与双向查询（BPE/Qwen 共用）──────────
struct VocabSource {
    std::variant<std::filesystem::path, std::vector<std::byte>> artifact;
    enum class Schema { TokenizerJson };
    Schema schema = Schema::TokenizerJson;
};

class VocabStore {
public:
    // 解析 tokenizer.json 建双向映射（含 added_tokens overlay）；加载后只读
    // 错误: FAILURE(缺失/不可读/JSON 非法/vocab 空/merges-vocab 不对齐)
    Status load(const VocabSource& src);
    // L4 .bincache sidecar（perf_optimize.md §3.4；env CANN_LM_TOKENIZER_
    // BINCACHE 显式开启、默认关）：load 成功后把九成员定长序列化落盘（失败
    // 静默）；缓存装载仅 load(path 态) 内部消费——键校验失败/中途任何读取
    // 失败整包丢弃回退 JSON，不发布半加载实例（F-02 同源）。
    Status Serialize(const std::filesystem::path& cache_path) const;
    Status TryLoadFromCache(const std::filesystem::path& json_path, const std::filesystem::path& cache_path);
    // O(1)（unordered_map）；只读可并发；含 added_tokens 字面量
    std::optional<int> token_to_id(std::string_view tok) const;
    std::optional<std::string_view> id_to_token(int id) const;
    // base vocab + added_tokens 总数（Qwen 实物 248070；gemma 实物 262144）
    size_t vocab_size() const;
    const std::vector<std::pair<std::string, std::string>>& merges() const;
    // added_tokens 全集（供 AddedTokenSplitter；special 与非 special 均含）
    const std::vector<std::pair<std::string, int>>& added_tokens() const;
    // special 剥离集：added_tokens 中 special=true 的全部 id（供 M-06 decode 剥离）
    const std::vector<int>& special_strip_ids() const;
    // 产物元信息（BPE 路径消费；Qwen 路径仅记录）
    const std::string& model_type() const { return model_type_; }
    bool byte_fallback() const { return byte_fallback_; }
    const std::string& unk_token() const { return unk_token_; }
    // Replace 型 normalizer（gemma 实物：" "→"▁"）；NFC 等归一化不实现（U-4）
    bool has_replace_normalizer() const { return has_replace_normalizer_; }
    const std::pair<std::string, std::string>& replace_normalizer() const { return replace_normalizer_; }

private:
    // L4：JSON 全量解析（原 load 本体；vocab_store.cpp 实现）——load 分流后
    // path 态走缓存未命中/bytes 态的直通路径
    Status load_from_json_(const VocabSource& src);

    std::unordered_map<std::string, int> tok2id_;
    std::vector<std::string> id2tok_; // id→token 字面量（含 added overlay）
    std::vector<std::pair<std::string, std::string>> merges_;
    std::vector<std::pair<std::string, int>> added_tokens_;
    std::vector<int> special_strip_ids_; // sorted
    std::string model_type_;
    bool byte_fallback_ = false;
    std::string unk_token_;
    bool has_replace_normalizer_ = false;
    std::pair<std::string, std::string> replace_normalizer_;
};

// ── M-06 SpecialTokenHandler：注入/剥离/padding（零注入缺省）────────────────
class SpecialTokenHandler {
public:
    // 构造/加载期调用；strip_ids = 产物 special 全集（M-05::special_strip_ids）
    // cfg 为 TokenizerConfig（SR20260903001 D-13：单一配置直通，-1 哨兵 = 未配置）；
    // meta special_tokens 的 id 并入剥离集（meta special 视为 special，
    // skip_special_tokens 时一并剥离）并建 id→literal 查找表（decode 解映射）
    void configure(const TokenizerConfig& cfg, std::vector<int> strip_ids);
    // encode 尾段：①截断（预算 = max_length − 注入数）②注入 bos/eos ③padding
    // 错误: INVALID_ARGUMENT(空 padding 或 pad 需求无 pad id)
    Status inject(std::vector<int>* ids, const EncodeOptions& opts) const;
    // decode 前段：剥离集内的 id 被剔除
    Status strip(const std::vector<int>& ids, std::vector<int>* kept) const;
    std::optional<int> id(std::string_view name) const; // bos/eos/pad/unk
    // meta special_tokens 映射消费（D-3：meta 优先 + WARN）——encode 提取表：
    // 词表 added_tokens ∪ meta 映射，同字面量冲突 → meta 覆盖 + WARN。
    // 构造期一次性调用，产物由调用方持有；meta 映射不进 VocabStore（.bincache
    // 缓存键只覆盖 tokenizer.json，meta 变更不会失效缓存——配置层归宿）
    std::vector<std::pair<std::string, int>> merged_added_tokens(
        const std::vector<std::pair<std::string, int>>& vocab_added) const;
    // meta 解映射：id 命中 meta special_tokens → 字面量（词表外 id 出字面量
    // 的唯一通道）；未命中返回 nullopt
    std::optional<std::string_view> literal(int id) const;
    const TokenizerConfig& config() const { return cfg_; }

private:
    TokenizerConfig cfg_;
    std::vector<int> strip_ids_; // sorted（产物 special ∪ meta special id）
    // meta special_tokens id→literal（configure 期构建；decode 解映射查表）
    std::unordered_map<int, std::string> id2lit_;
};

// ── M-03 BPETokenizer（手写 BPE，gemma4）───────────────────────────────────
class BPETokenizer final : public TokenizerBase {
public:
    // 经 cfg.tokenizer_file 加载 tokenizer.json（校验 model.type==BPE）+ AddedToken 表
    // + special 配置（cfg 直通，零转换——D-13/D-14）
    // 错误: FAILURE(缺失/JSON 非法/model.type 非 BPE/不对齐)
    explicit BPETokenizer(const TokenizerConfig& cfg, Status* status);
    // AddedToken/meta 映射提取 → UTF-8 校验 → Replace 归一(▁) → Split 预切分
    // → BPE 合并 → id 查表(byte_fallback/unk) → special 注入（M-06）
    Status encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids) override;
    // meta 解映射优先 → id→字面量→▁→空格 还原 + ByteFallback 展开
    Status decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text) override;
    size_t vocab_size() const override;
    std::optional<int> special_token_id(std::string_view name) const override;

private:
    VocabStore vocab_;
    SpecialTokenHandler specials_;
    // 词表 added ∪ meta special_tokens 合并提取表（D-3：meta 优先；构造期一次性）
    std::vector<std::pair<std::string, int>> added_lookup_;
    // merges → 字符串域 rank 表（构造期构建；BPE 合并按优先级查）
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
};

// ── M-04 QwenTokenizer（手写 ByteLevel BPE，Qwen3.5）───────────────────────
class QwenTokenizer final : public TokenizerBase {
public:
    // 经 cfg.tokenizer_file 加载 tokenizer.json + AddedToken Trie + special 配置
    // 错误: FAILURE(缺失/JSON 非法/不对齐)
    explicit QwenTokenizer(const TokenizerConfig& cfg, Status* status);
    // AddedToken/meta 映射提取 → UTF-8 校验 → D-10 正则预切分（字节级）→ 字节转
    // Unicode 映射(wchar_t) → BPE 合并 → subword 转 UTF-8 查表 → special 注入（M-06）
    Status encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids) override;
    // meta 解映射优先 → id→字面量→ByteLevel 字节还原
    Status decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text) override;
    size_t vocab_size() const override;
    std::optional<int> special_token_id(std::string_view name) const override;

private:
    VocabStore vocab_;
    SpecialTokenHandler specials_;
    // 词表 added ∪ meta special_tokens 合并提取表（D-3：meta 优先；构造期一次性）
    std::vector<std::pair<std::string, int>> added_lookup_;
    // merges → wstring 域 rank 表（用户拍板口径：字节映射后 wchar_t 域合并）
    std::map<std::pair<std::wstring, std::wstring>, int> merge_ranks_;
};

// ── M-07 SPMTokenizer（sentencepiece 占位桩，本期不实现）────────────────────
class SPMTokenizer final : public TokenizerBase {
public:
    // 占位：.model 加载恒失败，status 回传 FAILURE，实例 not-ready
    explicit SPMTokenizer(const TokenizerConfig& cfg, Status* status);
    // 占位：恒返回 FAILURE（not-ready 桩行为）
    Status encode(std::string_view text, const EncodeOptions& opts, std::vector<int>* out_ids) override;
    Status decode(const std::vector<int>& ids, const DecodeOptions& opts, std::string* out_text) override;
    size_t vocab_size() const override; // 0（未加载）
    std::optional<int> special_token_id(std::string_view name) const override;
};

} // namespace cann_local_lm::engine
