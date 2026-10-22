#include "core/context.h"

#include <utility>

namespace cann_lm::engine {

class ContextImpl {
public:
    ContextId id;
    explicit ContextImpl(ContextId cid) : id(cid) {}
};

bool ContextId::operator==(const ContextId& other) const noexcept { return value == other.value; }
bool ContextId::operator!=(const ContextId& other) const noexcept { return value != other.value; }
bool ContextId::operator<(const ContextId& other) const noexcept  { return value < other.value; }
ContextId::operator bool() const noexcept                         { return value != 0; }

Context::Context() noexcept = default;
Context::Context(const Context&) noexcept = default;
Context& Context::operator=(const Context&) noexcept = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;
Context::~Context() = default;

Context::Context(std::shared_ptr<ContextImpl> impl) : impl_(std::move(impl)) {}

ContextId Context::id() const noexcept {
    return impl_ ? impl_->id : ContextId{};
}

Context::operator bool() const noexcept {
    return static_cast<bool>(impl_);
}

Context Context::CreateWithId(ContextId id) {
    return Context(std::make_shared<ContextImpl>(id));
}

}  // namespace cann_lm::engine
