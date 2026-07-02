#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "shadow_elimination_logic.h"

#ifdef __cplusplus
extern "C" {
#endif

bool shadow_ai_init(void);
bool shadow_ai_process_frame(uint16_t *rgb565_frame, int width, int height);
bool shadow_ai_get_last_features(shadow_frame_features_t *features);

#ifdef __cplusplus
}
#endif
