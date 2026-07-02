#pragma once

#include "three_axis_controller.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SHADOW_ELIMINATION_IDLE = 0,
    SHADOW_ELIMINATION_BLOCKED,
    SHADOW_ELIMINATION_TRACKING,
    SHADOW_ELIMINATION_ACCEPTABLE,
    SHADOW_ELIMINATION_SEARCHING,
} shadow_elimination_mode_t;

typedef struct {
    int paper_pixels;
    int shadow_on_paper_pixels;
    int near_shadow_on_paper_pixels;
    int hand_pen_on_paper_pixels;
    image_point_t paper_centroid_px;
    image_point_t shadow_centroid_px;
    image_point_t near_shadow_centroid_px;
    bool has_paper;
    bool has_shadow;
    bool has_near_shadow;
} shadow_frame_features_t;

typedef struct {
    int min_paper_pixels;
    int min_shadow_pixels;
    int min_near_shadow_pixels;
    float target_shadow_ratio;
    float target_near_shadow_ratio;
    float clear_shadow_ratio;
    float max_hand_pen_ratio;
    float smoothing_alpha;
    float centroid_gain;
    float near_centroid_gain;
    float z_lift_per_shadow_ratio_mm;
    float near_z_lift_per_shadow_ratio_mm;
    float max_xy_offset_mm;
    float min_confidence;
    float near_camera_dir_x_mm;
    float near_camera_dir_y_mm;
    float near_shadow_weight;
    float improvement_epsilon;
    int acceptable_frames_required;
    int stuck_frames_required;
    float search_step_mm;
} shadow_elimination_config_t;

typedef struct {
    shadow_elimination_mode_t mode;
    float smoothed_shadow_ratio;
    float smoothed_near_shadow_ratio;
    float best_near_shadow_ratio;
    int stuck_frame_count;
    int clear_frame_count;
    int search_direction;
    cartesian_point_t last_target_mm;
} shadow_elimination_state_t;

typedef struct {
    shadow_elimination_mode_t mode;
    cartesian_point_t target_mm;
    cartesian_point_t paper_center_mm;
    cartesian_point_t shadow_center_mm;
    float shadow_ratio;
    float near_shadow_ratio;
    float hand_pen_ratio;
    float confidence;
    bool is_improving;
    bool should_update_pwm;
    const char *reason;
} shadow_elimination_decision_t;

void shadow_elimination_default_config(shadow_elimination_config_t *config);
void shadow_elimination_init(shadow_elimination_state_t *state, cartesian_point_t initial_target_mm);
shadow_elimination_decision_t shadow_elimination_update(shadow_elimination_state_t *state,
                                                        const shadow_elimination_config_t *config,
                                                        const image_to_cartesian_t *calibration,
                                                        const shadow_frame_features_t *features);

#ifdef __cplusplus
}
#endif
