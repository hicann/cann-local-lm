/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * 多进程并发示例：td::EpairModelLoader::RunModel（同步接口）
 *
 * 形态（multi_process，一进程一实例）：
 *   - 父进程拉起 workers（默认 8）个 worker 子进程，各自持有独立的
 *     EpairModelLoader / GeSession / ACL 上下文与 Device 流；
 *   - worker 内部为“生产者线程 + 消费线程”，并行接任务执行 RunModel；
 *   - 观测：host 侧多进程并发下发、device 侧算子多流并行执行。
 *
 * 为什么是多进程：RunModel 封装的 GE 同步 RunGraph 在 InnerSession 内持
 * 进程级 static mutex（try_to_lock，锁在整个图执行期间持有），同进程多线程
 * 并发调用会被 GE_SESS_ALREADY_RUNNING 顶回；一进程一实例可绕开该约束，
 * 恢复 device 侧多流并发。
 *
 * 度量口径（warmup -> 串行 SOLO 基线 -> 并行 RUN 两阶段，CLOCK_MONOTONIC）：
 *   host_max_inflight：RunModel 调用区间的扫描线最大重叠数（>=2 即并发下发）
 *   device_parallelism：并行任务数 x t_solo / wall（RunModel 阻塞至图完成）
 *
 * 用法：
 *   ./run_model_multi_instance <epair> [workers=8] [tasks=48] [dim=1024]
 *   内部（worker 子进程）：--worker <id> <tasks> <epair> <dim> <go_file>
 * 前置：epair 同级目录下需存在 input_0.bin（dim*dim 的 float32 原始数据）
 *
 * 环境变量：
 *   TD_MULTI_TIMEOUT_S     栅栏超时秒数（默认 900）
 */

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "ge/ge_api_v2.h"
#include "torch_delegate/epair_model_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kPollIntervalUs = 1000;  // 栅栏轮询 1ms

double NowMs() {
  return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

std::string Dirname(const std::string &path) {
  const auto pos = path.rfind('/');
  return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

size_t NumElems(int dim) {
  return static_cast<size_t>(dim) * static_cast<size_t>(dim);
}

uint64_t ChecksumHost(const gert::Tensor &tensor, size_t elems) {
  const float *p = reinterpret_cast<const float *>(tensor.GetAddr());
  if (p == nullptr) {
    return 0;
  }
  const size_t step = std::max<size_t>(1, elems / 4096);
  uint64_t sum = 0;
  for (size_t i = 0; i < elems; i += step) {
    sum += static_cast<uint64_t>(static_cast<double>(p[i]) * 1000.0);
  }
  return sum;
}

// ---------------------------------------------------------------------------
// 公共 setup/teardown（两种模式共用）
// ---------------------------------------------------------------------------
bool InitDeviceAndGe() {
  (void)aclInit(nullptr);
  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
  if (ge::GEInitializeV2(ge_options) != ge::SUCCESS) {
    return false;
  }
  if (aclrtSetDevice(0) != ACL_SUCCESS) {
    ge::GEFinalizeV2();
    return false;
  }
  return true;
}

// 创建并加载单个 EpairModelLoader 实例
bool BuildLoader(const std::string &epair, std::unique_ptr<td::EpairModelLoader> *loader) {
  std::map<ge::AscendString, ge::AscendString> options = {
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
  };
  *loader = std::make_unique<td::EpairModelLoader>(epair.c_str());
  if (*loader == nullptr || (*loader)->CompileAndLoad(options, nullptr) != td::SUCCESS) {
    return false;
  }
  return true;
}

void Teardown(void *dev_input) {
  if (dev_input != nullptr) {
    (void)aclrtFree(dev_input);
  }
  (void)aclrtResetDevice(0);
  ge::GEFinalizeV2();
  (void)aclFinalize();
}

// 构造 Device 侧输入 x（dim x dim float32）
void *BuildDeviceInput(const std::string &bin_path, int dim, std::vector<gert::Tensor> &inputs) {
  const size_t elems = NumElems(dim);
  const size_t bytes = elems * sizeof(float);

  std::ifstream file(bin_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open input file: " + bin_path);
  }
  std::vector<float> host(elems);
  file.read(reinterpret_cast<char *>(host.data()), static_cast<std::streamsize>(bytes));
  if (!file) {
    throw std::runtime_error("Failed to read input file: " + bin_path);
  }

  void *dev = nullptr;
  if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    throw std::runtime_error("aclrtMalloc failed");
  }
  if (aclrtMemcpy(dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    (void)aclrtFree(dev);
    throw std::runtime_error("Input H2D copy failed");
  }

  gert::StorageShape shape({dim, dim}, {dim, dim});
  gert::Tensor tensor;
  tensor.GetShape() = shape;
  tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  tensor.SetDataType(ge::DT_FLOAT);
  gert::TensorData tensor_data(dev, nullptr, bytes, gert::kOnDeviceHbm);
  tensor.SetData(std::move(tensor_data));
  inputs.push_back(std::move(tensor));
  return dev;
}

struct TaskResult {
  double t0_ms = 0.0;
  double t1_ms = 0.0;
  unsigned status = 0;
  uint64_t checksum = 0;
  double DurMs() const {
    return t1_ms - t0_ms;
  }
  bool Ok() const {
    return status == static_cast<unsigned>(td::SUCCESS);
  }
};

// 执行单个任务：调用同步接口 RunModel，成功时输出已就绪（Host 侧）可直接校验
TaskResult ExecTask(td::EpairModelLoader *loader, const std::vector<gert::Tensor> &inputs, int dim) {
  TaskResult r;
  std::vector<gert::Tensor> outputs;
  r.t0_ms = NowMs();
  const td::Status status = loader->RunModel(inputs, outputs);
  r.t1_ms = NowMs();
  r.status = static_cast<unsigned>(status);
  if (status == td::SUCCESS && !outputs.empty()) {
    r.checksum = ChecksumHost(outputs[0], NumElems(dim));
  }
  return r;
}

// ---------------------------------------------------------------------------
// 统一报告
// ---------------------------------------------------------------------------
struct ReportVals {
  std::string mode;
  int workers = 0;
  int tasks = 0;
  double t_solo = 0.0;
  double wall = 0.0;
  double est_serial = 0.0;
  int success = 0;
  int fail = 0;
  int mismatch = 0;
  double sum_ms = 0.0;
  int inflight = 0;
  double device_parallelism = 0.0;
};

void PrintReport(const ReportVals &v) {
  const double avg_conc = (v.wall > 0.0) ? v.sum_ms / v.wall : 0.0;
  std::cout << "\n========================================" << std::endl;
  std::cout << " mode                : " << v.mode << std::endl;
  std::cout << " instances (loaders) : " << v.workers << std::endl;
  std::cout << " tasks               : " << v.tasks << std::endl;
  std::cout << " t_solo (median)     : " << v.t_solo << " ms" << std::endl;
  std::cout << " wall            : " << v.wall << " ms (parallel phase)" << std::endl;
  std::cout << " est serial wall : " << v.est_serial << " ms (parallel task count x per-task elapsed time)"
            << std::endl;
  std::cout << " sum task time   : " << v.sum_ms << " ms" << std::endl;
  std::cout << " success / fail  : " << v.success << " / " << v.fail << std::endl;
  std::cout << " checksum mismatch: " << v.mismatch << std::endl;
  std::cout << " avg concurrency : " << avg_conc << " (interface busy time / total time, reference value)" << std::endl;
  std::cout << " host_max_inflight : " << v.inflight << " / " << v.workers
            << " (threads concurrently inside RunModel; >=2 means host-side concurrent dispatch)" << std::endl;
  std::cout << " device_parallelism : " << v.device_parallelism << " / " << v.workers
            << " (estimated device multi-stream parallelism: ~workers = fully parallel, ~1 = serial)" << std::endl;
  std::cout << "========================================" << std::endl;
}

double MedianOf(const std::vector<double> &v) {
  if (v.empty()) {
    return 0.0;
  }
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

// 扫描线求 RunModel 调用区间最大重叠数（同时进入接口的线程数）
int MaxInflight(const std::vector<std::pair<double, double>> &intervals) {
  std::vector<std::pair<double, int>> events;
  events.reserve(intervals.size() * 2);
  for (const auto &iv : intervals) {
    events.emplace_back(iv.first, +1);
    events.emplace_back(iv.second, -1);
  }
  std::sort(events.begin(), events.end(), [](const auto &l, const auto &r) {
    return (l.first != r.first) ? (l.first < r.first) : (l.second < r.second);
  });
  int cur = 0;
  int best = 0;
  for (const auto &e : events) {
    cur += e.second;
    best = std::max(best, cur);
  }
  return best;
}

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
struct Args {
  bool is_worker = false;
  int worker_id = 0;
  int worker_tasks = 0;
  std::string epair;
  int dim = 1024;
  std::string go_path;
  int workers = 8;
  int tasks = 48;
};

bool ParseWorkerArgs(int argc, char *argv[], Args *a) {
  if (argc != 7) {
    std::cerr << "worker usage: --worker <id> <tasks> <epair> <dim> <go_file>" << std::endl;
    return false;
  }
  a->is_worker = true;
  a->worker_id = std::atoi(argv[2]);
  a->worker_tasks = std::atoi(argv[3]);
  a->epair = argv[4];
  a->dim = std::atoi(argv[5]);
  a->go_path = argv[6];
  return true;
}

bool ParseArgs(int argc, char *argv[], Args *a) {
  if (argc >= 2 && std::strcmp(argv[1], "--worker") == 0) {
    return ParseWorkerArgs(argc, argv, a);
  }
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    pos.push_back(argv[i]);
  }
  if (pos.empty()) {
    std::cerr << "Usage: " << argv[0] << " <epair_file_path> [workers=8] [tasks=48] [dim=1024]" << std::endl;
    return false;
  }
  a->epair = pos[0];
  a->workers = (pos.size() > 1) ? std::atoi(pos[1].c_str()) : 8;
  a->tasks = (pos.size() > 2) ? std::atoi(pos[2].c_str()) : 48;
  a->dim = (pos.size() > 3) ? std::atoi(pos[3].c_str()) : 1024;
  if (a->workers <= 0 || a->tasks < 2 * a->workers || a->dim <= 0) {
    std::cerr << "[FAIL] Require workers>0 and tasks>=2*workers (1 SOLO task per instance + at least 1 "
                 "parallel task)"
              << std::endl;
    return false;
  }
  return true;
}

double TimeoutMs() {
  const char *v = std::getenv("TD_MULTI_TIMEOUT_S");
  return (v != nullptr ? std::stod(v) : 900.0) * 1000.0;
}

// ---------------------------------------------------------------------------
// worker 子进程（每进程 1 个 Loader + 生产者/消费线程）
// ---------------------------------------------------------------------------
void WriteGo(const std::string &path, const std::string &phase) {
  std::ofstream f(path, std::ios::trunc);
  f << phase << "\n";
  f.flush();
}

std::string ReadGo(const std::string &path) {
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
  return line;
}

bool WaitForGo(const std::string &path, const std::string &expect, double deadline_ms) {
  while (NowMs() < deadline_ms) {
    if (ReadGo(path) == expect) {
      return true;
    }
    (void)usleep(kPollIntervalUs);
  }
  return false;
}

int RunWorker(const Args &a) {
  const double deadline = NowMs() + TimeoutMs();
  if (!InitDeviceAndGe()) {
    std::printf("ERR|%d|failed to initialize Device/GE\n", a.worker_id);
    return 2;
  }
  std::unique_ptr<td::EpairModelLoader> loader;
  if (!BuildLoader(a.epair, &loader)) {
    std::printf("ERR|%d|CompileAndLoad failed\n", a.worker_id);
    Teardown(nullptr);
    return 2;
  }
  std::vector<gert::Tensor> inputs;
  void *dev_input = nullptr;
  try {
    dev_input = BuildDeviceInput(Dirname(a.epair) + "/input_0.bin", a.dim, inputs);
  } catch (const std::exception &e) {
    std::printf("ERR|%d|%s\n", a.worker_id, e.what());
    Teardown(nullptr);
    return 2;
  }

  (void)ExecTask(loader.get(), inputs, a.dim);  // 预热
  std::printf("READY|%d\n", a.worker_id);
  std::fflush(stdout);

  if (!WaitForGo(a.go_path, "1", deadline)) {
    std::printf("ERR|%d|timeout waiting for SOLO barrier\n", a.worker_id);
    loader.reset();
    Teardown(dev_input);
    return 3;
  }
  {
    const TaskResult r = ExecTask(loader.get(), inputs, a.dim);
    std::printf("SOLO|%d|%.3f|%.3f|%u|%llu\n", a.worker_id, r.t0_ms, r.t1_ms, r.status,
                static_cast<unsigned long long>(r.checksum));
    std::fflush(stdout);
  }
  if (!WaitForGo(a.go_path, "2", deadline)) {
    std::printf("ERR|%d|timeout waiting for RUN barrier\n", a.worker_id);
    loader.reset();
    Teardown(dev_input);
    return 3;
  }

  std::mutex mtx;
  std::condition_variable cv;
  std::queue<int> tasks;
  bool finished = false;
  std::atomic<int> fails{0};
  std::thread producer([&]() {
    for (int t = 0; t < a.worker_tasks; ++t) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        tasks.push(t);
      }
      cv.notify_one();
    }
    {
      std::lock_guard<std::mutex> lk(mtx);
      finished = true;
    }
    cv.notify_one();
  });

  (void)aclrtSetDevice(0);
  int executed = 0;
  while (true) {
    int task = -1;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&]() { return !tasks.empty() || finished; });
      if (tasks.empty()) {
        break;
      }
      task = tasks.front();
      tasks.pop();
    }
    const TaskResult r = ExecTask(loader.get(), inputs, a.dim);
    std::printf("RESULT|%d|%d|%.3f|%.3f|%u|%llu\n", a.worker_id, task, r.t0_ms, r.t1_ms, r.status,
                static_cast<unsigned long long>(r.checksum));
    std::fflush(stdout);
    if (!r.Ok()) {
      ++fails;
      std::printf("ERR|%d|%d|status=%u ge_msg=%s\n", a.worker_id, task, r.status, ge::GEGetErrorMsgV3().GetString());
      std::fflush(stdout);
    }
    ++executed;
  }
  producer.join();
  loader.reset();
  Teardown(dev_input);
  return (fails.load() > 0 || executed != a.worker_tasks) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// 父进程：拉起 workers 个子进程、驱动栅栏、汇总统计
// ---------------------------------------------------------------------------
struct WorkerStat {
  pid_t pid = -1;
  std::string log_path;
  std::vector<std::string> solo_fields;
};

std::vector<std::string> SplitFields(const std::string &line) {
  std::vector<std::string> parts;
  std::string rest = line;
  size_t pos = 0;
  while ((pos = rest.find('|')) != std::string::npos) {
    parts.push_back(rest.substr(0, pos));
    rest.erase(0, pos + 1);
  }
  parts.push_back(rest);
  return parts;
}

std::vector<std::string> ReadLines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

bool HasLineWith(const std::vector<std::string> &lines, const std::string &prefix) {
  for (const auto &l : lines) {
    if (l.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

bool AllWorkersReady(const std::vector<WorkerStat> &ws) {
  for (const auto &w : ws) {
    if (!HasLineWith(ReadLines(w.log_path), "READY|")) {
      return false;
    }
  }
  return true;
}

// 收集所有 worker 的 SOLO 行（每次重新读取，避免轮询累积）；不足 workers 个则返回 false
bool CollectSolo(std::vector<WorkerStat> *ws) {
  bool all = true;
  for (auto &w : *ws) {
    w.solo_fields.clear();
    for (const auto &l : ReadLines(w.log_path)) {
      if (l.rfind("SOLO|", 0) == 0) {
        w.solo_fields.push_back(l);
      }
    }
    if (w.solo_fields.empty()) {
      all = false;
    }
  }
  return all;
}

pid_t SpawnWorker(const std::string &exe, int id, int tasks, const Args &a, const std::string &log_path) {
  const int fd = ::open(log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    return -1;
  }
  const pid_t pid = fork();
  if (pid == 0) {
    (void)dup2(fd, STDOUT_FILENO);
    (void)dup2(fd, STDERR_FILENO);
    for (int f = 3; f < 1024; ++f) {
      (void)close(f);
    }
    const std::string id_str = std::to_string(id);
    const std::string tasks_str = std::to_string(tasks);
    const std::string dim_str = std::to_string(a.dim);
    char *const argv[] = {const_cast<char *>(exe.c_str()),       const_cast<char *>("--worker"),
                          const_cast<char *>(id_str.c_str()),    const_cast<char *>(tasks_str.c_str()),
                          const_cast<char *>(a.epair.c_str()),   const_cast<char *>(dim_str.c_str()),
                          const_cast<char *>(a.go_path.c_str()), nullptr};
    (void)execv(exe.c_str(), argv);
    _exit(127);
  }
  (void)close(fd);
  return pid;
}

void KillWorkers(std::vector<WorkerStat> *ws) {
  for (auto &w : *ws) {
    if (w.pid > 0) {
      (void)kill(w.pid, SIGKILL);
    }
  }
}

int WaitWorkers(std::vector<WorkerStat> *ws) {
  int bad_exit = 0;
  for (auto &w : *ws) {
    int status = 0;
    if (waitpid(w.pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++bad_exit;
    }
  }
  return bad_exit;
}

bool AnyWorkerExitedEarly(const std::vector<WorkerStat> &ws) {
  for (const auto &w : ws) {
    int status = 0;
    if (waitpid(w.pid, &status, WNOHANG) == w.pid) {
      return true;
    }
  }
  return false;
}

struct RunAgg {
  int parallel_tasks = 0;
  int success = 0;
  int fail = 0;
  double sum_ms = 0.0;
  double last_t1 = 0.0;
  std::set<uint64_t> checksums;
  std::vector<std::pair<double, double>> intervals;
  std::string first_err;
};

RunAgg AggregateResults(std::vector<WorkerStat> *ws) {
  RunAgg agg;
  for (auto &w : *ws) {
    for (const auto &l : ReadLines(w.log_path)) {
      if (l.rfind("RESULT|", 0) != 0) {
        if (l.rfind("ERR|", 0) == 0 && agg.first_err.empty()) {
          agg.first_err = l;
        }
        continue;
      }
      const auto parts = SplitFields(l.substr(7));  // w|task|t0|t1|status|checksum
      if (parts.size() != 6) {
        continue;
      }
      ++agg.parallel_tasks;
      const double t0 = std::stod(parts[2]);
      const double t1 = std::stod(parts[3]);
      const unsigned st = static_cast<unsigned>(std::stoul(parts[4]));
      agg.sum_ms += (t1 - t0);
      agg.last_t1 = std::max(agg.last_t1, t1);
      if (st == static_cast<unsigned>(td::SUCCESS)) {
        ++agg.success;
        agg.checksums.insert(std::strtoull(parts[5].c_str(), nullptr, 10));
        agg.intervals.emplace_back(t0, t1);
      } else {
        ++agg.fail;
      }
    }
  }
  return agg;
}

void FillMpReport(const RunAgg &agg, const std::vector<double> &ok_solos, int solo_fails, int bad_exit, const Args &a,
                  int parallel_total, double wall, double t_solo, ReportVals *v) {
  v->mode = "multi_process (" + std::to_string(a.workers) + " processes x 1 Loader + producer/consumer threads each)";
  v->workers = a.workers;
  v->tasks = a.tasks;
  v->t_solo = t_solo;
  v->wall = wall;
  v->est_serial = parallel_total * t_solo;
  v->success = agg.success + static_cast<int>(ok_solos.size());
  v->fail = agg.fail + solo_fails;
  if (bad_exit > 0 && v->fail == 0) {
    v->fail = bad_exit;  // 进程异常退出且无任务失败记录
  }
  v->mismatch = agg.checksums.size() > 1 ? static_cast<int>(agg.checksums.size()) - 1 : 0;
  v->sum_ms = agg.sum_ms;
  v->inflight = MaxInflight(agg.intervals);
  v->device_parallelism = (wall > 0 && t_solo > 0) ? v->est_serial / wall : 0.0;
}

int RunParent(const Args &a) {
  char tpl[] = "/tmp/td_run_model_multi_XXXXXX";
  const char *dir = mkdtemp(tpl);
  if (dir == nullptr) {
    std::cerr << "[FAIL] mkdtemp failed" << std::endl;
    return 1;
  }
  const std::string work_dir(dir);
  const double deadline = NowMs() + TimeoutMs();
  const std::string go_path = work_dir + "/go";
  WriteGo(go_path, "0");
  Args wa = a;  // 传给 worker 的参数副本（携带 go 文件路径）
  wa.go_path = go_path;

  char exe_buf[4096];
  const ssize_t exe_len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (exe_len <= 0) {
    std::cerr << "[FAIL] Cannot resolve own executable path" << std::endl;
    return 1;
  }
  exe_buf[exe_len] = '\0';
  const std::string exe(exe_buf);

  const int parallel_total = a.tasks - a.workers;
  std::vector<WorkerStat> ws(static_cast<size_t>(a.workers));
  for (int i = 0; i < a.workers; ++i) {
    const int per = parallel_total / a.workers + ((i < parallel_total % a.workers) ? 1 : 0);
    ws[static_cast<size_t>(i)].log_path = work_dir + "/worker_" + std::to_string(i) + ".log";
    ws[static_cast<size_t>(i)].pid = SpawnWorker(exe, i, per, wa, ws[static_cast<size_t>(i)].log_path);
    if (ws[static_cast<size_t>(i)].pid <= 0) {
      std::cerr << "[FAIL] fork worker " << i << " failed" << std::endl;
      KillWorkers(&ws);
      return 1;
    }
  }
  std::cout << "[PASS] spawned " << a.workers << " worker processes, workdir=" << work_dir << std::endl;

  while (!AllWorkersReady(ws)) {
    if (NowMs() > deadline) {
      std::cerr << "[FAIL] Timeout waiting for worker READY" << std::endl;
      KillWorkers(&ws);
      return 1;
    }
    if (AnyWorkerExitedEarly(ws)) {
      std::cerr << "[FAIL] Some worker exited early, see " << work_dir << "/worker_*.log" << std::endl;
      KillWorkers(&ws);
      return 1;
    }
    (void)usleep(kPollIntervalUs);
  }

  WriteGo(go_path, "1");
  while (!CollectSolo(&ws)) {
    if (NowMs() > deadline) {
      std::cerr << "[FAIL] Timeout waiting for SOLO results" << std::endl;
      KillWorkers(&ws);
      return 1;
    }
    if (AnyWorkerExitedEarly(ws)) {
      std::cerr << "[FAIL] Some worker exited around the SOLO phase, see " << work_dir << "/worker_*.log" << std::endl;
      KillWorkers(&ws);
      return 1;
    }
    (void)usleep(kPollIntervalUs);
  }

  const double wall_start = NowMs();
  WriteGo(go_path, "2");
  const int bad_exit = WaitWorkers(&ws);
  const double reap_wall = NowMs() - wall_start;

  RunAgg agg = AggregateResults(&ws);
  double wall = reap_wall;
  if (agg.parallel_tasks > 0 && agg.last_t1 > wall_start) {
    wall = agg.last_t1 - wall_start;
  }

  std::vector<double> ok_solos;
  int solo_fails = 0;
  for (const auto &w : ws) {
    for (const auto &l : w.solo_fields) {
      const auto parts = SplitFields(l.substr(5));  // w|t0|t1|status|checksum
      if (parts.size() != 5) {
        continue;
      }
      if (std::stoul(parts[3]) == 0) {
        ok_solos.push_back(std::stod(parts[2]) - std::stod(parts[1]));
        agg.checksums.insert(std::strtoull(parts[4].c_str(), nullptr, 10));
      } else {
        ++solo_fails;
      }
    }
  }

  const double t_solo = MedianOf(ok_solos);
  ReportVals v;
  FillMpReport(agg, ok_solos, solo_fails, bad_exit, a, parallel_total, wall, t_solo, &v);
  PrintReport(v);

  if (!agg.first_err.empty()) {
    std::cout << "[hint] First failure record: " << agg.first_err << std::endl;
    std::cout << "[hint] Worker logs: " << work_dir << "/worker_*.log" << std::endl;
  }

  if (v.fail > 0) {
    std::cerr << "[FAIL] Some tasks/processes failed: tasks_fail=" << agg.fail + solo_fails << " bad_exit=" << bad_exit
              << std::endl;
    return 1;
  }
  if (v.mismatch > 0) {
    std::cerr << "[FAIL] Output checksum mismatch" << std::endl;
    return 1;
  }
  std::cout << "[PASS] All " << a.tasks << " tasks succeeded, output checksums consistent" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  Args a;
  if (!ParseArgs(argc, argv, &a)) {
    return 1;
  }
  if (a.is_worker) {
    return RunWorker(a);
  }
  return RunParent(a);
}
