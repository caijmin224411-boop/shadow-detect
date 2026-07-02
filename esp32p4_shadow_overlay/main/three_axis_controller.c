#include "three_axis_controller.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define AXIS_COUNT 3

static float clampf(float value, float low, float high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static float apply_deadband(float error, float deadband)
{
    if (fabsf(error) <= deadband) {
        return 0.0f;
    }
    return error;
}

static float pid_step(const axis_pid_config_t *cfg,
                      float error,
                      float *integral,
                      float *previous_error,
                      bool has_previous_error,
                      float dt_s)
{
    if (dt_s <= 0.0f) {
        dt_s = 0.02f;
    }

    error = apply_deadband(error, cfg->deadband_mm);
    *integral += error * dt_s;
    *integral = clampf(*integral, -cfg->integral_limit, cfg->integral_limit);

    float derivative = 0.0f;
    if (has_previous_error) {
        derivative = (error - *previous_error) / dt_s;
    }
    *previous_error = error;

    const float out = cfg->kp * error + cfg->ki * (*integral) + cfg->kd * derivative;
    return clampf(out, -cfg->output_limit, cfg->output_limit);
}

void three_axis_default_config(three_axis_config_t *config)
{
    if (config == NULL) {
        return;
    }

    memset(config, 0, sizeof(*config));

    /*
     * Identity-like fallback: x=image u, y=image v, z=0.
     * Replace this with a real paper-plane homography after calibration.
     */
    config->calibration.h[0] = 1.0f;
    config->calibration.h[4] = 1.0f;
    config->calibration.h[8] = 1.0f;
    config->calibration.z_mm = 0.0f;

    config->pwm_frequency_hz = 50;
    config->pwm_resolution_bits = 14;

    for (int i = 0; i < AXIS_COUNT; ++i) {
        config->pid[i].kp = 0.45f;
        config->pid[i].ki = 0.02f;
        config->pid[i].kd = 0.06f;
        config->pid[i].integral_limit = 80.0f;
        config->pid[i].output_limit = 12.0f;
        config->pid[i].deadband_mm = 1.5f;

        config->pwm[i].min_angle_deg = 0.0f;
        config->pwm[i].max_angle_deg = 180.0f;
        config->pwm[i].center_angle_deg = 90.0f;
        config->pwm[i].mm_per_degree = 2.0f;
        config->pwm[i].max_step_deg = 3.0f;
        config->pwm[i].min_pulse_us = 500;
        config->pwm[i].max_pulse_us = 2500;
    }
}

void three_axis_init(three_axis_controller_t *controller, const three_axis_config_t *config)
{
    if (controller == NULL || config == NULL) {
        return;
    }

    memset(controller, 0, sizeof(*controller));
    controller->config = *config;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        controller->angle_deg[i] = config->pwm[i].center_angle_deg;
    }
}

void three_axis_set_target_mm(three_axis_controller_t *controller, cartesian_point_t target)
{
    if (controller == NULL) {
        return;
    }
    controller->setpoint_mm = target;
}

bool image_to_cartesian(const image_to_cartesian_t *calibration, image_point_t image, cartesian_point_t *out)
{
    if (calibration == NULL || out == NULL) {
        return false;
    }

    const float *h = calibration->h;
    const float denom = h[6] * image.u + h[7] * image.v + h[8];
    if (fabsf(denom) < 1e-6f) {
        return false;
    }

    out->x_mm = (h[0] * image.u + h[1] * image.v + h[2]) / denom;
    out->y_mm = (h[3] * image.u + h[4] * image.v + h[5]) / denom;
    out->z_mm = calibration->z_mm;
    return true;
}

int servo_angle_to_pulse_us(const pwm_axis_config_t *axis, float angle_deg)
{
    if (axis == NULL) {
        return 1500;
    }

    angle_deg = clampf(angle_deg, axis->min_angle_deg, axis->max_angle_deg);
    const float span_angle = axis->max_angle_deg - axis->min_angle_deg;
    if (span_angle <= 0.0f) {
        return axis->min_pulse_us;
    }

    const float t = (angle_deg - axis->min_angle_deg) / span_angle;
    const float pulse = (float)axis->min_pulse_us +
        t * (float)(axis->max_pulse_us - axis->min_pulse_us);
    return (int)lroundf(pulse);
}

float servo_pulse_to_duty_ratio(int pulse_us, int pwm_frequency_hz)
{
    if (pwm_frequency_hz <= 0) {
        pwm_frequency_hz = 50;
    }
    return ((float)pulse_us * (float)pwm_frequency_hz) / 1000000.0f;
}

three_axis_output_t three_axis_update_from_image(three_axis_controller_t *controller,
                                                 image_point_t measured_image,
                                                 float dt_s)
{
    three_axis_output_t out;
    memset(&out, 0, sizeof(out));

    if (controller == NULL) {
        return out;
    }

    cartesian_point_t measured = {0};
    if (!image_to_cartesian(&controller->config.calibration, measured_image, &measured)) {
        measured = controller->setpoint_mm;
    }

    const float target[AXIS_COUNT] = {
        controller->setpoint_mm.x_mm,
        controller->setpoint_mm.y_mm,
        controller->setpoint_mm.z_mm,
    };
    const float now[AXIS_COUNT] = {
        measured.x_mm,
        measured.y_mm,
        measured.z_mm,
    };

    out.target_mm = controller->setpoint_mm;
    out.measured_mm = measured;

    for (int i = 0; i < AXIS_COUNT; ++i) {
        const pwm_axis_config_t *axis = &controller->config.pwm[i];
        const float error = target[i] - now[i];
        const float correction_mm = pid_step(&controller->config.pid[i],
                                             error,
                                             &controller->integral[i],
                                             &controller->previous_error[i],
                                             controller->has_previous_error,
                                             dt_s);
        float delta_deg = 0.0f;
        if (fabsf(axis->mm_per_degree) > 1e-6f) {
            delta_deg = correction_mm / axis->mm_per_degree;
        }
        delta_deg = clampf(delta_deg, -axis->max_step_deg, axis->max_step_deg);

        controller->angle_deg[i] = clampf(controller->angle_deg[i] + delta_deg,
                                          axis->min_angle_deg,
                                          axis->max_angle_deg);

        out.error_mm[i] = error;
        out.angle_deg[i] = controller->angle_deg[i];
        out.pulse_us[i] = servo_angle_to_pulse_us(axis, out.angle_deg[i]);
        out.duty_ratio[i] = servo_pulse_to_duty_ratio(out.pulse_us[i],
                                                      controller->config.pwm_frequency_hz);
    }

    controller->has_previous_error = true;
    return out;
}
