#include "servo_keyboard.h"

#include <stdio.h>
#include <stdint.h>

#include "driver/ledc.h"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

static const char *TAG = "servo_keyboard";

#if CONFIG_SHADOW_SERVO_KEYBOARD_ENABLE

#define SERVO_COUNT 3
#define SERVO_MIN_DEG 0
#define SERVO_MAX_DEG 180
#define SERVO_CENTER_DEG 90
#define SERVO_SMALL_STEP_DEG 2
#define SERVO_LARGE_STEP_DEG 10
#define SERVO_MIN_PULSE_US 500
#define SERVO_MAX_PULSE_US 2500
#define SERVO_PWM_FREQ_HZ 50
#define SERVO_LEDC_RES LEDC_TIMER_14_BIT
#define SERVO_LEDC_MAX_DUTY ((1U << SERVO_LEDC_RES) - 1U)

static const int s_servo_gpio[SERVO_COUNT] = {
    CONFIG_SHADOW_SERVO1_GPIO,
    CONFIG_SHADOW_SERVO2_GPIO,
    CONFIG_SHADOW_SERVO3_GPIO,
};

static const ledc_channel_t s_servo_channel[SERVO_COUNT] = {
    LEDC_CHANNEL_0,
    LEDC_CHANNEL_1,
    LEDC_CHANNEL_2,
};

static int s_angle[SERVO_COUNT] = {
    SERVO_CENTER_DEG,
    SERVO_CENTER_DEG,
    SERVO_CENTER_DEG,
};
static bool s_usb_serial_jtag_ready = false;

static uint32_t servo_angle_to_duty(int angle)
{
    if (angle < SERVO_MIN_DEG) {
        angle = SERVO_MIN_DEG;
    } else if (angle > SERVO_MAX_DEG) {
        angle = SERVO_MAX_DEG;
    }

    const uint32_t pulse_us = SERVO_MIN_PULSE_US +
        ((SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US) * (uint32_t)angle) / SERVO_MAX_DEG;
    return (pulse_us * SERVO_PWM_FREQ_HZ * SERVO_LEDC_MAX_DUTY) / 1000000U;
}

static void servo_write_angle(int index, int angle)
{
    if (index < 0 || index >= SERVO_COUNT) {
        return;
    }

    if (angle < SERVO_MIN_DEG) {
        angle = SERVO_MIN_DEG;
    } else if (angle > SERVO_MAX_DEG) {
        angle = SERVO_MAX_DEG;
    }

    s_angle[index] = angle;
    ESP_ERROR_CHECK(ledc_set_duty(LEDC_LOW_SPEED_MODE, s_servo_channel[index],
                                  servo_angle_to_duty(angle)));
    ESP_ERROR_CHECK(ledc_update_duty(LEDC_LOW_SPEED_MODE, s_servo_channel[index]));
}

static void servo_print_help(void)
{
    ESP_LOGI(TAG, "keyboard: q/a servo1 +/-2deg, w/s servo2 +/-2deg, e/d servo3 +/-2deg");
    ESP_LOGI(TAG, "keyboard: Q/A W/S E/D use +/-10deg, r center all, p print angles, h help");
}

static void servo_print_angles(void)
{
    ESP_LOGI(TAG, "angles: servo1=%d servo2=%d servo3=%d",
             s_angle[0], s_angle[1], s_angle[2]);
}

static void servo_adjust(int index, int delta)
{
    servo_write_angle(index, s_angle[index] + delta);
    servo_print_angles();
}

static void servo_center_all(void)
{
    for (int i = 0; i < SERVO_COUNT; ++i) {
        servo_write_angle(i, SERVO_CENTER_DEG);
    }
    servo_print_angles();
}

static void servo_keyboard_task(void *arg)
{
    (void)arg;
    servo_print_help();
    servo_print_angles();

    while (1) {
        uint8_t c = 0;
        int len = 0;
        if (s_usb_serial_jtag_ready) {
            len = usb_serial_jtag_read_bytes(&c, 1, 0);
        }
        if (len <= 0) {
            len = uart_read_bytes((uart_port_t)CONFIG_ESP_CONSOLE_UART_NUM,
                                  &c,
                                  1,
                                  pdMS_TO_TICKS(50));
        }
        if (len <= 0) {
            continue;
        }

        switch (c) {
        case 'q': servo_adjust(0, SERVO_SMALL_STEP_DEG); break;
        case 'a': servo_adjust(0, -SERVO_SMALL_STEP_DEG); break;
        case 'w': servo_adjust(1, SERVO_SMALL_STEP_DEG); break;
        case 's': servo_adjust(1, -SERVO_SMALL_STEP_DEG); break;
        case 'e': servo_adjust(2, SERVO_SMALL_STEP_DEG); break;
        case 'd': servo_adjust(2, -SERVO_SMALL_STEP_DEG); break;
        case 'Q': servo_adjust(0, SERVO_LARGE_STEP_DEG); break;
        case 'A': servo_adjust(0, -SERVO_LARGE_STEP_DEG); break;
        case 'W': servo_adjust(1, SERVO_LARGE_STEP_DEG); break;
        case 'S': servo_adjust(1, -SERVO_LARGE_STEP_DEG); break;
        case 'E': servo_adjust(2, SERVO_LARGE_STEP_DEG); break;
        case 'D': servo_adjust(2, -SERVO_LARGE_STEP_DEG); break;
        case 'r':
        case 'R':
            servo_center_all();
            break;
        case 'p':
        case 'P':
            servo_print_angles();
            break;
        case 'h':
        case 'H':
            servo_print_help();
            break;
        default:
            break;
        }
    }
}

#endif

bool servo_keyboard_init(void)
{
#if CONFIG_SHADOW_SERVO_KEYBOARD_ENABLE
    ledc_timer_config_t timer_config = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = SERVO_LEDC_RES,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = SERVO_PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_config));

    for (int i = 0; i < SERVO_COUNT; ++i) {
        ledc_channel_config_t channel_config = {
            .gpio_num = s_servo_gpio[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = s_servo_channel[i],
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = servo_angle_to_duty(SERVO_CENTER_DEG),
            .hpoint = 0,
            .flags.output_invert = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&channel_config));
    }

    if (!uart_is_driver_installed((uart_port_t)CONFIG_ESP_CONSOLE_UART_NUM)) {
        esp_err_t uart_ret = uart_driver_install((uart_port_t)CONFIG_ESP_CONSOLE_UART_NUM,
                                                 1024,
                                                 0,
                                                 0,
                                                 NULL,
                                                 0);
        if (uart_ret != ESP_OK) {
            ESP_LOGE(TAG, "failed to install console UART driver: %s", esp_err_to_name(uart_ret));
            return false;
        }
    }

    if (!usb_serial_jtag_is_driver_installed()) {
        usb_serial_jtag_driver_config_t usb_config = {
            .tx_buffer_size = 1024,
            .rx_buffer_size = 1024,
        };
        esp_err_t usb_ret = usb_serial_jtag_driver_install(&usb_config);
        if (usb_ret == ESP_OK) {
            s_usb_serial_jtag_ready = true;
        } else {
            ESP_LOGW(TAG, "USB Serial/JTAG keyboard input unavailable: %s", esp_err_to_name(usb_ret));
        }
    } else {
        s_usb_serial_jtag_ready = true;
    }

    servo_center_all();

    BaseType_t task_ok = xTaskCreatePinnedToCore(servo_keyboard_task,
                                                 "servo_keyboard",
                                                 4096,
                                                 NULL,
                                                 5,
                                                 NULL,
                                                 0);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "failed to create keyboard task");
        return false;
    }

    ESP_LOGI(TAG, "PWM servos enabled on GPIO%d/GPIO%d/GPIO%d, keyboard input: UART%s",
             s_servo_gpio[0], s_servo_gpio[1], s_servo_gpio[2],
             s_usb_serial_jtag_ready ? "+USB" : "");
    return true;
#else
    ESP_LOGI(TAG, "PWM servo keyboard disabled");
    return false;
#endif
}
