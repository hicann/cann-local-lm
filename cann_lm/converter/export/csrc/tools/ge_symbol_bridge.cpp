/* ge_symbol_bridge.cpp — 提供 libge_compiler.so 需要的旧签名 CustomPassHelper::Run(3 params)。
 *
 * libge_compiler.so 链接时引用了 3-param Run，但 libregister.so 只提供 4-param 版本。
 * 此桥接源码直接编译进 cann_lm_compile 二进制，提供 3-param 符号。
 *
 * 返回 graphStatus (int): 0 = GRAPH_SUCCESS。
 * 之前用 void 返回类型导致 RAX 未设置，GE 将垃圾值当作失败处理。
 */

extern "C" {
// 旧签名: Run(shared_ptr<Graph>&, CustomPassContext&, CustomPassStage)
// mangled: _ZNK2ge16CustomPassHelper3RunERSt10shared_ptrINS_5GraphEERNS_17CustomPassContextENS_15CustomPassStageE
//
// 不转发到 4-param 版本（避免跨库 ABI 不匹配导致堆损坏），
// 返回 0 (GRAPH_SUCCESS) 表示 custom pass 执行成功（no-op 跳过所有 pass）。
int _ZNK2ge16CustomPassHelper3RunERSt10shared_ptrINS_5GraphEERNS_17CustomPassContextENS_15CustomPassStageE(
    const void* self,
    void* graph,      // shared_ptr<ge::Graph>&
    void* ctx,        // ge::CustomPassContext&
    void* stage       // ge::CustomPassStage
) {
    // No-op: custom passes are optional optimizations, return success
    (void)self; (void)graph; (void)ctx; (void)stage;
    return 0;  // GRAPH_SUCCESS
}
}
