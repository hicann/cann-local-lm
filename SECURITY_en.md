# Security Statement

English | [简体中文](SECURITY.md)

## Running User Recommendations

For security reasons, it is not recommended to use root or other administrator-type accounts to execute any commands; follow the principle of least privilege.

## File Permission Control

- It is recommended that users set the system umask value to 0027 or higher on the host (including the host machine) and in containers, ensuring that new directories have a default maximum permission of 750 and new files have a default maximum permission of 640.
- It is recommended that users implement permission control and other security measures for sensitive content such as personal privacy data, commercial assets, source files, and model weights. The recommended permissions are referenced in [A. Recommended Maximum File/Directory Permissions by Scenario](#a-recommended-maximum-filedirectory-permissions-by-scenario).
- The engine may cache compilation files during compilation, stored in the `kernel_meta_*` folder under the runtime directory; users may control permissions of the generated files as needed.

## Build Security Statement

When compiling and installing this project from source, you need to compile it yourself; some intermediate files will be generated during compilation. It is recommended that you control the permissions of intermediate files after compilation to ensure file security.

## Public Network Address Statement

The public network addresses included in this project's code are declared as follows:

| Type | Open Source Code Address | File Name | Public Network Address | Usage Description |
|------|------------|--------|----------|----------|
| Dependency | N/A | cmake/third_party/build/Findmakeself.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | Download makeself source from gitcode, used as a compilation dependency |
| Dependency | N/A | cmake/third_party/build/Findnlohmann_json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz | Download json source from gitcode, used as a compilation dependency |
| Dependency | N/A | cmake/third_party/build/FindGTest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | Download googletest source from gitcode, used as a compilation dependency |
| Dependency | N/A | cmake/third_party/build/Findlibtorch_cpu.cmake | https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip | Download libtorch CPU version from pytorch.org, used as a compilation dependency |

---

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A. Recommended Maximum File/Directory Permissions by Scenario

| Type | Linux Permission Reference Maximum |
|------|---------------------|
| User home directory | 750 (rwxr-x---) |
| Program files (including script files, library files, etc.) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration file | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log file (recording complete or archived) | 440 (r--r-----) |
| Log file (being recorded) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug file | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance upgrade file directory | 770 (rwxrwx---) |
| Business data file | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key component, private key, certificate, ciphertext file directory | 700 (rwx------) |
| Key component, private key, certificate, encrypted ciphertext | 600 (rw-------) |
| Encryption/decryption interface, encryption/decryption script | 500 (r-x------) |

## Disclaimer

This project is provided "as is", without any express or implied warranties regarding merchantability or fitness for a particular purpose. The maintenance team shall not be liable for any direct or indirect losses arising from the use of this project.
