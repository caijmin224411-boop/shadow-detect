#include "esp32_three_axis_pwm.h"

#include <stddef.h>
#include <stdint.h>

#include "driver/ledc.h"
#include "esp_check.h"

#define ESP32_PWM_AXIS_COUNT 3
#define ESP32_PWM_TIMER LEDC_TIMER_0
#define ESP32_PWM_SPEED_MODE LEDC_LOW_SPEED_MODE

static const ledc_channel_t s_channels[ESP32_PWM_AXIS_COUNT] = {
    LEDC_CHANNEL_0,
    LEDC_CHANNEL_1,
    LEDC_CHANNEL_2,
};

static uint32_t pulse_to_ledc_duty(int pulse_us, int pwm_frequency_hz, int resolution_bits)
{
    if (pwm_frequency_hz <= 0) {
        pwm_frequency_hz = 50;
    }
    if (resolution_bits <= 0) {
        resolution_bits = 14;
    }

    const uint32_t max_duty = (1UL << (uint32_t)resolution_bits) - 1UL;
    return ((uint32_t)pulse_us * (uint32_t)pwm_frequency_hz * max_duty) / 1000000UL;
}

esp_err_t esp32_three_axis_pwm_init(const esp32_three_axis_pwm_config_t *gpio_config,
                                    const three_axis_config_t *controller_config)
{
    if (gpio_config == NULL || controller_config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    ledc_timer_bit_t resolution = (ledc_timer_bit_t)controller_config->pwm_resolution_bits;
    ledc_timer_config_t timer_config = {
        .speed_mode = ESP32_PWM_SPEED_MODE,
        .duty_resolution = resolution,
        .timer_num = ESP32_PWM_TIMER,
        .freq_hz = controller_config->pwm_frequency_hz,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_config), "three_axis_pwm", "timer config failed");

    for (int i = 0; i < ESP32_PWM_AXIS_COUNT; ++i) {
        int pulse_us = servo_angle_to_pulse_us(&controller_config->pwm[i],
                                               controller_config->pwm[i].center_angle_deg);
        ledc_channel_config_t channel_config = {
            .gpio_num = gpio_config->gpio[i],
            .speed_mode = ESP32_PWM_SPEED_MODE,
            .channel = s_channels[i],
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = ESP32_PWM_TIMER,
            .duty = pulse_to_ledc_duty(pulse_us,
                                       controller_config->pwm_frequency_hz,
                                       controller_config->pwm_resolution_bits),
            .hpoint = 0,
            .flags.output_invert = 0,
        };
        ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_config),
                            "three_axis_pwm",
                            "channel config failed");
    }

    return ESP_OK;
}

esp_err_t esp32_three_axis_pwm_write(const three_axis_output_t *output,
                                     const three_axis_config_t *controller_config)
{
    if (output == NULL || controller_config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < ESP32_PWM_AXIS_COUNT; ++i) {
        uint32_t duty = pulse_to_ledc_duty(output->pulse_us[i],
                                          controller_config->pwm_frequency_hz,
                                          controller_config->pwm_resolution_bits);
        ESP_RETURN_ON_ERROR(ledc_set_duty(ESP32_PWM_SPEED_MODE, s_channels[i], duty),
                            "three_axis_pwm",
                            "set duty failed");
        ESP_RETURN_ON_ERROR(ledc_update_duty(ESP32_PWM_SPEED_MODE, s_channels[i]),
                            "three_axis_pwm",
                            "update duty failed");
    }

    return ESP_OK;
}
