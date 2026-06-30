#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool shadow_ai_init(void);
bool shadow_ai_process_frame(uint16_t *rgb565_frame, int width, int height);

#ifdef __cplusplus
}
#endif
