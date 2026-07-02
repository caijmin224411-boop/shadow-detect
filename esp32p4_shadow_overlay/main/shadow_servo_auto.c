#include "shadow_servo_auto.h"

#include "esp_log.h"
#include "sdkconfig.h"
#include "shadow_ai.h"
#include "shadow_elimination_logic.h"
#include "esp32_three_axis_pwm.h"
#include "three_axis_controller.h"

static const char *TAG = "shadow_servo_auto";

#if CONFIG_SHADOW_SERVO_AUTO_ENABLE && !CONFIG_SHADOW_SERVO_KEYBOARD_ENABLE
static three_axis_config_t s_axis_config;
static three_axis_controller_t s_axis_controller;
static shadow_elimination_config_t s_elimination_config;
static shadow_elimination_state_t s_elimination_state;
static bool s_ready = false;

static void configure_camera_to_paper_calibration(image_to_cartesian_t *calibration)
{
    /*
     * Current firmware feature coordinates are 48x48 mask cells.
     * Demo calibration maps mask center to paper origin and assumes
     * +Y points toward the camera side of the paper.
     */
    calibration->h[0] = 4.0f;
    calibration->h[1] = 0.0f;
    calibration->h[2] = -94.0f;
    calibration->h[3] = 0.0f;
    calibration->h[4] = 4.0f;
    calibration->h[5] = -94.0f;
    calibration->h[6] = 0.0f;
    calibration->h[7] = 0.0f;
    calibration->h[8] = 1.0f;
    calibration->z_mm = 0.0f;
}
#endif

bool shadow_servo_auto_init(void)
{
#if CONFIG_SHADOW_SERVO_AUTO_ENABLE
#if CONFIG_SHADOW_SERVO_KEYBOARD_ENABLE
    ESP_LOGW(TAG, "disabled because keyboard servo control is enabled");
    return false;
#else
    three_axis_default_config(&s_axis_config);
    configure_camera_to_paper_calibration(&s_axis_config.calibration);

    for (int i = 0; i < 3; ++i) {
        s_axis_config.pwm[i].min_angle_deg = 60.0f;
        s_axis_config.pwm[i].max_angle_deg = 120.0f;
        s_axis_config.pwm[i].center_angle_deg = 90.0f;
        s_axis_config.pwm[i].max_step_deg = 1.0f;
    }

    shadow_elimination_default_config(&s_elimination_config);
    s_elimination_config.min_paper_pixels = 1200;
    s_elimination_config.min_shadow_pixels = 8;
    s_elimination_config.min_near_shadow_pixels = 4;
    s_elimination_config.target_near_shadow_ratio = 0.01f;
    s_elimination_config.near_camera_dir_x_mm = 0.0f;
    s_elimination_config.near_camera_dir_y_mm = 1.0f;

    three_axis_init(&s_axis_controller, &s_axis_config);
    shadow_elimination_init(&s_elimination_state,
                            (cartesian_point_t){.x_mm = 0.0f, .y_mm = 0.0f, .z_mm = 0.0f});

    esp32_three_axis_pwm_config_t pwm_gpio = {
        .gpio = {
            CONFIG_SHADOW_SERVO1_GPIO,
            CONFIG_SHADOW_SERVO2_GPIO,
            CONFIG_SHADOW_SERVO3_GPIO,
        },
    };
    esp_err_t ret = esp32_three_axis_pwm_init(&pwm_gpio, &s_axis_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "PWM init failed: %s", esp_err_to_name(ret));
        return false;
    }

    s_ready = true;
    ESP_LOGI(TAG, "automatic near-camera shadow reduction enabled");
    return true;
#endif
#else
    ESP_LOGI(TAG, "automatic shadow servo control disabled");
    return false;
#endif
}

bool shadow_servo_auto_update(void)
{
#if CONFIG_SHADOW_SERVO_AUTO_ENABLE && !CONFIG_SHADOW_SERVO_KEYBOARD_ENABLE
    if (!s_ready) {
        return false;
    }

    shadow_frame_features_t features;
    if (!shadow_ai_get_last_features(&features)) {
        return false;
    }

    shadow_elimination_decision_t decision =
        shadow_elimination_update(&s_elimination_state,
                                  &s_elimination_config,
                                  &s_axis_config.calibration,
                                  &features);
    if (!decision.should_update_pwm) {
        return false;
    }

    three_axis_set_target_mm(&s_axis_controller, decision.target_mm);
    three_axis_output_t output =
        three_axis_update_from_image(&s_axis_controller,
                                     features.near_shadow_centroid_px,
                                     0.033f);
    esp_err_t ret = esp32_three_axis_pwm_write(&output, &s_axis_config);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "PWM write failed: %s", esp_err_to_name(ret));
        return false;
    }
    return true;
#else
    return false;
#endif
}
