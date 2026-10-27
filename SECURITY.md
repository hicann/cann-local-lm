# 安全声明

简体中文 | [English](SECURITY_en.md)

## 运行用户建议

基于安全性角度考虑，不建议使用 root 等管理员类型账户执行任何命令，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统 umask 值为 0027 及以上，保障新增文件夹默认最高权限为 750，新增文件默认最高权限为 640。
- 建议用户对个人隐私数据、商业资产、源文件和模型权重等敏感内容做好权限控制等安全措施。设定的权限建议参考 [A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)。
- 引擎编译时可能会缓存编译文件，存储在运行目录下的 `kernel_meta_*` 文件夹内，用户可根据需要自行对生成后的相关文件进行权限控制。

## 构建安全声明

在源码编译安装本项目时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后，对中间文件做好权限控制，以保证文件安全。

## 公网地址声明

本项目代码中包含的公网地址声明如下所示：

| 类型 | 开源代码地址 | 文件名 | 公网地址 | 用途说明 |
|------|------------|--------|----------|----------|
| 依赖 | 不涉及 | cmake/third_party/build/Findmakeself.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | 从 gitcode 下载 makeself 源码，作用编译依赖 |
| 依赖 | 不涉及 | cmake/third_party/build/Findnlohmann_json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz | 从 gitcode 下载 json 源码，作用编译依赖 |
| 依赖 | 不涉及 | cmake/third_party/build/FindGTest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | 从 gitcode 下载 googletest 源码，作用编译依赖 |
| 依赖 | 不涉及 | cmake/third_party/build/Findlibtorch_cpu.cmake | https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip | 从 pytorch.org 下载 libtorch CPU 版本，作用编译依赖 |

---

## 漏洞机制说明

[漏洞管理](https://gitcode.com/cann/community/blob/master/security/security.md)

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型 | Linux 权限参考最大值 |
|------|---------------------|
| 用户主目录 | 750（rwxr-x---） |
| 程序文件（含脚本文件、库文件等） | 550（r-xr-x---） |
| 程序文件目录 | 550（r-xr-x---） |
| 配置文件 | 640（rw-r-----） |
| 配置文件目录 | 750（rwxr-x---） |
| 日志文件（记录完毕或者已经归档） | 440（r--r-----） |
| 日志文件（正在记录） | 640（rw-r-----） |
| 日志文件目录 | 750（rwxr-x---） |
| Debug 文件 | 640（rw-r-----） |
| Debug 文件目录 | 750（rwxr-x---） |
| 临时文件目录 | 750（rwxr-x---） |
| 维护升级文件目录 | 770（rwxrwx---） |
| 业务数据文件 | 640（rw-r-----） |
| 业务数据文件目录 | 750（rwxr-x---） |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx------） |
| 密钥组件、私钥、证书、加密密文 | 600（rw-------） |
| 加解密接口、加解密脚本 | 500（r-x------） |

## 免责

本项目按"现状"提供，不对适销性或特定用途适用性做任何明示或暗示的保证。使用本项目产生的任何直接或间接损失，维护团队不承担责任。
