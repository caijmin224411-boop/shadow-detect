#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x_mm;
    float y_mm;
    float z_mm;
} cartesian_point_t;

typedef struct {
    float u;
    float v;
} image_point_t;

typedef struct {
    float h[9];
    float z_mm;
} image_to_cartesian_t;

typedef struct {
    float kp;
    float ki;
    float kd;
    float integral_limit;
    float output_limit;
    float deadband_mm;
} axis_pid_config_t;

typedef struct {
    float min_angle_deg;
    float max_angle_deg;
    float center_angle_deg;
    float mm_per_degree;
    float max_step_deg;
    int min_pulse_us;
    int max_pulse_us;
} pwm_axis_config_t;

typedef struct {
    image_to_cartesian_t calibration;
    axis_pid_config_t pid[3];
    pwm_axis_config_t pwm[3];
    int pwm_frequency_hz;
    int pwm_resolution_bits;
} three_axis_config_t;

typedef struct {
    cartesian_point_t target_mm;
    cartesian_point_t measured_mm;
    float angle_deg[3];
    float duty_ratio[3];
    int pulse_us[3];
    float error_mm[3];
} three_axis_output_t;

typedef struct {
    three_axis_config_t config;
    cartesian_point_t setpoint_mm;
    float integral[3];
    float previous_error[3];
    float angle_deg[3];
    bool has_previous_error;
} three_axis_controller_t;

void three_axis_default_config(three_axis_config_t *config);
void three_axis_init(three_axis_controller_t *controller, const three_axis_config_t *config);
void three_axis_set_target_mm(three_axis_controller_t *controller, cartesian_point_t target);
bool image_to_cartesian(const image_to_cartesian_t *calibration, image_point_t image, cartesian_point_t *out);
three_axis_output_t three_axis_update_from_image(three_axis_controller_t *controller,
                                                 image_point_t measured_image,
                                                 float dt_s);
int servo_angle_to_pulse_us(const pwm_axis_config_t *axis, float angle_deg);
float servo_pulse_to_duty_ratio(int pulse_us, int pwm_frequency_hz);

#ifdef __cplusplus
}
#endif
