#include "api/cpp/request_handle.h"

#include "pipeline/scheduler.h"

#include <utility>

namespace cann_local_lm::engine {

RequestHandle::RequestHandle() noexcept = default;
RequestHandle::RequestHandle(RequestHandle&&) noexcept = default;
RequestHandle& RequestHandle::operator=(RequestHandle&&) noexcept = default;
RequestHandle::~RequestHandle() = default;

RequestHandle::RequestHandle(std::shared_ptr<RequestImpl> impl) : impl_(std::move(impl)) {}

RequestStatus RequestHandle::GetStatus() const noexcept
{
    if (!impl_)
        return RequestStatus::Queued;
    return impl_->status.load(std::memory_order_acquire);
}

bool RequestHandle::IsDone() const noexcept
{
    if (!impl_)
        return false;
    switch (impl_->status.load(std::memory_order_acquire)) {
        case RequestStatus::Completed:
        case RequestStatus::Canceled:
        case RequestStatus::Failed:
            return true;
        default:
            return false;
    }
}

Status RequestHandle::Cancel()
{
    if (!impl_)
        return Status::SUCCESS;
    impl_->canceled.store(true, std::memory_order_release);
    return Status::SUCCESS;
}

Status RequestHandle::Wait(Result* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (!impl_)
        return Status::INVALID_STATE;

    // 等待终态
    {
        std::unique_lock<std::mutex> lk(impl_->mtx);
        impl_->cv.wait(lk, [&] {
            switch (impl_->status.load(std::memory_order_acquire)) {
                case RequestStatus::Completed:
                case RequestStatus::Canceled:
                case RequestStatus::Failed:
                    return true;
                default:
                    return false;
            }
        });
    }

    // 填 *out（按终态语义）
    auto s = impl_->status.load(std::memory_order_acquire);
    switch (s) {
        case RequestStatus::Completed:
            out->token_ids = impl_->token_ids;
            out->text = impl_->text;
            out->finish_reason = impl_->finish_reason;
            return Status::SUCCESS;
        case RequestStatus::Canceled:
            // 丢弃部分 token（不返回给用户）
            return Status::CANCELED;
        case RequestStatus::Failed:
            // 保留部分输出便于排查
            out->token_ids = impl_->token_ids;
            out->text = impl_->text;
            out->finish_reason = impl_->finish_reason;
            return Status::FAILURE;
        default:
            return Status::INVALID_STATE;
    }
}

Status RequestHandle::TryGetResult(Result* out) const
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (!impl_)
        return Status::INVALID_STATE;
    switch (impl_->status.load(std::memory_order_acquire)) {
        case RequestStatus::Completed:
            out->token_ids = impl_->token_ids;
            out->text = impl_->text;
            out->finish_reason = impl_->finish_reason;
            return Status::SUCCESS;
        case RequestStatus::Canceled:
            // 丢弃部分 token（不返回给用户）
            return Status::CANCELED;
        case RequestStatus::Failed:
            out->token_ids = impl_->token_ids;
            out->text = impl_->text;
            out->finish_reason = impl_->finish_reason;
            return Status::FAILURE;
        default:
            return Status::INVALID_STATE;
    }
}

RequestHandle::operator bool() const noexcept { return static_cast<bool>(impl_); }

} // namespace cann_local_lm::engine
