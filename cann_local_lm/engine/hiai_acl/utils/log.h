#pragma once
#include <cstdio>
#define ACL_LOGI(fmt, ...) printf("[ACL-I] " fmt "\n", ##__VA_ARGS__)
#define ACL_LOGE(fmt, ...) fprintf(stderr, "[ACL-E] " fmt "\n", ##__VA_ARGS__)
#define ACL_LOGW(fmt, ...) printf("[ACL-W] " fmt "\n", ##__VA_ARGS__)
