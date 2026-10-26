#include "backend/aten_schema_resolver.h"

#include "common/logging/logging.h"

namespace td {

AtenTargetInfo AtenSchemaResolver::ParseTarget(const std::string& target, ValidationResult& result,
                                               const std::string& context) const
{
    AtenTargetInfo info;
    info.target = target;
    std::string op_str = target;
    constexpr char kPrefix[] = "torch.ops.";
    if (op_str.rfind(kPrefix, 0) == 0) {
        op_str = op_str.substr(sizeof(kPrefix) - 1);
    }
    const size_t last_dot = op_str.find_last_of('.');
    if (last_dot == std::string::npos) {
        AddError(result, "INVALID_ATEN_TARGET", "Target missing overload separator", context);
        return info;
    }

    const std::string overload = op_str.substr(last_dot + 1);
    const std::string ns_op = op_str.substr(0, last_dot);
    const size_t second_dot = ns_op.find_last_of('.');
    if (second_dot == std::string::npos) {
        AddError(result, "INVALID_ATEN_TARGET", "Target missing namespace separator", context);
        return info;
    }

    info.namespace_name = ns_op.substr(0, second_dot);
    info.op_name = ns_op.substr(second_dot + 1);
    info.overload = overload == "default" ? "" : overload;
    info.dispatcher_name = info.namespace_name + "::" + info.op_name;
    return info;
}

AtenSchemaInfo AtenSchemaResolver::Resolve(const ParsedNode& node, ValidationResult& result) const
{
    AtenSchemaInfo info;
    const std::string context = NodeDebugContext(node);
    info.target = ParseTarget(node.effective_schema_target, result, context);
    if (!result.ok) {
        return info;
    }
    info.is_out = info.target.overload == "out";
    info.is_inplace = !info.is_out && !info.target.op_name.empty() && info.target.op_name.back() == '_';

    if (node.inputs.size() < node.schema_metadata_input_count) {
        AddError(result, "INVALID_FALLBACK_INPUTS", "schema metadata input count is invalid", context);
        return info;
    }

    for (size_t index = node.schema_metadata_input_count; index < node.inputs.size(); ++index) {
        const ParsedNodeInput& input = node.inputs[index];
        AtenSchemaArgInfo arg_info;
        arg_info.name = input.name.empty() ? "input_" + std::to_string(info.arguments.size() + 1) : input.name;
        arg_info.is_tensor = input.input_kind == NodeInputKind::Tensor;
        arg_info.is_optional_tensor = false;
        arg_info.is_tensor_list = false;
        arg_info.is_out = false;
        arg_info.has_default = false;
        if (!arg_info.is_tensor) {
            AddError(result, "UNSUPPORTED_FALLBACK_INPUT", "Fallback node currently only supports tensor JSON inputs",
                     context + ":" + arg_info.name);
            return info;
        }
        info.arguments.push_back(std::move(arg_info));
    }
    return info;
}

std::string AtenSchemaResolver::MakeFallbackOpType(const AtenTargetInfo& info) const
{
    std::string id = info.namespace_name;
    if (!id.empty()) {
        id += "__";
    }
    id += info.op_name;
    if (!info.overload.empty()) {
        id += "_" + info.overload;
    }
    return id;
}

} // namespace td
