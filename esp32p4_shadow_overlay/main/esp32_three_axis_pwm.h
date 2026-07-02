#pragma once

#include "three_axis_controller.h"

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int gpio[3];
} esp32_three_axis_pwm_config_t;

esp_err_t esp32_three_axis_pwm_init(const esp32_three_axis_pwm_config_t *gpio_config,
                                    const three_axis_config_t *controller_config);
esp_err_t esp32_three_axis_pwm_write(const three_axis_output_t *output,
                                     const three_axis_config_t *controller_config);

#ifdef __cplusplus
}
#endif
