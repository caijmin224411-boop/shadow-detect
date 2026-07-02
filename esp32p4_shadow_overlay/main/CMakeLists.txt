idf_component_register(SRCS
                       "mipi_isp_dsi_main.c"
                       "servo_keyboard.c"
                       "shadow_servo_auto.c"
                       "three_axis_controller.c"
                       "shadow_elimination_logic.c"
                       "esp32_three_axis_pwm.c"
                       "shadow_ai.cc"
                       "paper_shadow_p4_model_data.cc"
                       INCLUDE_DIRS "."
                       REQUIRES esp_mm esp_driver_isp esp_driver_cam esp_driver_i2c esp_driver_ledc esp_driver_uart esp_driver_usb_serial_jtag dsi_init sensor_init esp-tflite-micro
                       )
