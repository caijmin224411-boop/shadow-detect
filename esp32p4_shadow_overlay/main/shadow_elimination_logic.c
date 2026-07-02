#include "shadow_elimination_logic.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static float clampf_local(float value, float low, float high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static float safe_ratio(int part, int whole)
{
    if (whole <= 0 || part <= 0) {
        return 0.0f;
    }
    return (float)part / (float)whole;
}

void shadow_elimination_default_config(shadow_elimination_config_t *config)
{
    if (config == NULL) {
        return;
    }

    config->min_paper_pixels = 4000;
    config->min_shadow_pixels = 120;
    config->min_near_shadow_pixels = 80;
    config->target_shadow_ratio = 0.015f;
    config->target_near_shadow_ratio = 0.004f;
    config->clear_shadow_ratio = 0.008f;
    config->max_hand_pen_ratio = 0.18f;
    config->smoothing_alpha = 0.35f;
    config->centroid_gain = 0.65f;
    config->near_centroid_gain = 1.05f;
    config->z_lift_per_shadow_ratio_mm = 35.0f;
    config->near_z_lift_per_shadow_ratio_mm = 70.0f;
    config->max_xy_offset_mm = 45.0f;
    config->min_confidence = 0.25f;
    config->near_camera_dir_x_mm = 0.0f;
    config->near_camera_dir_y_mm = 1.0f;
    config->near_shadow_weight = 2.0f;
    config->improvement_epsilon = 0.0015f;
    config->acceptable_frames_required = 6;
    config->stuck_frames_required = 10;
    config->search_step_mm = 4.0f;
}

void shadow_elimination_init(shadow_elimination_state_t *state, cartesian_point_t initial_target_mm)
{
    if (state == NULL) {
        return;
    }

    memset(state, 0, sizeof(*state));
    state->mode = SHADOW_ELIMINATION_IDLE;
    state->best_near_shadow_ratio = 1.0f;
    state->search_direction = 1;
    state->last_target_mm = initial_target_mm;
}

shadow_elimination_decision_t shadow_elimination_update(shadow_elimination_state_t *state,
                                                        const shadow_elimination_config_t *config,
                                                        const image_to_cartesian_t *calibration,
                                                        const shadow_frame_features_t *features)
{
    shadow_elimination_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    decision.mode = SHADOW_ELIMINATION_IDLE;
    decision.reason = "invalid_input";

    if (state == NULL || config == NULL || calibration == NULL || features == NULL) {
        return decision;
    }

    const float shadow_ratio = safe_ratio(features->shadow_on_paper_pixels, features->paper_pixels);
    const float raw_near_shadow_ratio = safe_ratio(features->near_shadow_on_paper_pixels, features->paper_pixels);
    const float near_shadow_ratio = (features->has_near_shadow &&
                                     features->near_shadow_on_paper_pixels >= config->min_near_shadow_pixels)
        ? raw_near_shadow_ratio
        : 0.0f;
    const float priority_shadow_ratio = shadow_ratio + config->near_shadow_weight * near_shadow_ratio;
    const float hand_ratio = safe_ratio(features->hand_pen_on_paper_pixels, features->paper_pixels);
    state->smoothed_shadow_ratio =
        config->smoothing_alpha * priority_shadow_ratio +
        (1.0f - config->smoothing_alpha) * state->smoothed_shadow_ratio;
    state->smoothed_near_shadow_ratio =
        config->smoothing_alpha * near_shadow_ratio +
        (1.0f - config->smoothing_alpha) * state->smoothed_near_shadow_ratio;

    decision.shadow_ratio = state->smoothed_shadow_ratio;
    decision.near_shadow_ratio = state->smoothed_near_shadow_ratio;
    decision.hand_pen_ratio = hand_ratio;
    decision.target_mm = state->last_target_mm;

    if (!features->has_paper || features->paper_pixels < config->min_paper_pixels) {
        state->mode = SHADOW_ELIMINATION_BLOCKED;
        state->clear_frame_count = 0;
        state->stuck_frame_count = 0;
        decision.mode = state->mode;
        decision.reason = "paper_not_reliable";
        return decision;
    }

    if (hand_ratio > config->max_hand_pen_ratio) {
        state->mode = SHADOW_ELIMINATION_BLOCKED;
        state->clear_frame_count = 0;
        state->stuck_frame_count = 0;
        decision.mode = state->mode;
        decision.reason = "hand_pen_occluding_paper";
        return decision;
    }

    if (!features->has_shadow || features->shadow_on_paper_pixels < config->min_shadow_pixels) {
        state->smoothed_shadow_ratio = 0.0f;
        state->smoothed_near_shadow_ratio = 0.0f;
    }

    const bool near_shadow_acceptable =
        state->smoothed_near_shadow_ratio <= config->target_near_shadow_ratio;

    if (near_shadow_acceptable) {
        state->clear_frame_count += 1;
        state->stuck_frame_count = 0;
        state->mode = (state->clear_frame_count >= config->acceptable_frames_required)
            ? SHADOW_ELIMINATION_ACCEPTABLE
            : SHADOW_ELIMINATION_IDLE;
        decision.mode = state->mode;
        decision.reason = (state->mode == SHADOW_ELIMINATION_ACCEPTABLE)
            ? "near_camera_shadow_acceptable"
            : "near_camera_shadow_nearly_acceptable";
        return decision;
    }

    state->clear_frame_count = 0;

    if (state->smoothed_near_shadow_ratio + config->improvement_epsilon < state->best_near_shadow_ratio) {
        state->best_near_shadow_ratio = state->smoothed_near_shadow_ratio;
        state->stuck_frame_count = 0;
        decision.is_improving = true;
    } else {
        state->stuck_frame_count += 1;
        decision.is_improving = false;
    }

    const bool prefer_near_shadow = state->smoothed_near_shadow_ratio > config->target_near_shadow_ratio;
    const image_point_t active_shadow_centroid = prefer_near_shadow
        ? features->near_shadow_centroid_px
        : features->shadow_centroid_px;

    if (!image_to_cartesian(calibration, features->paper_centroid_px, &decision.paper_center_mm) ||
        !image_to_cartesian(calibration, active_shadow_centroid, &decision.shadow_center_mm)) {
        state->mode = SHADOW_ELIMINATION_BLOCKED;
        decision.mode = state->mode;
        decision.reason = "coordinate_transform_failed";
        return decision;
    }

    const float excess_shadow = fmaxf(0.0f, state->smoothed_shadow_ratio - config->target_shadow_ratio);
    const float excess_near_shadow = fmaxf(0.0f, state->smoothed_near_shadow_ratio - config->target_near_shadow_ratio);
    const float dx = decision.paper_center_mm.x_mm - decision.shadow_center_mm.x_mm;
    const float dy = decision.paper_center_mm.y_mm - decision.shadow_center_mm.y_mm;
    const float active_gain = prefer_near_shadow ? config->near_centroid_gain : config->centroid_gain;
    const float near_push_x = config->near_camera_dir_x_mm * excess_near_shadow * config->near_z_lift_per_shadow_ratio_mm;
    const float near_push_y = config->near_camera_dir_y_mm * excess_near_shadow * config->near_z_lift_per_shadow_ratio_mm;
    const float command_x = clampf_local(dx * active_gain - near_push_x,
                                         -config->max_xy_offset_mm,
                                         config->max_xy_offset_mm);
    const float command_y = clampf_local(dy * active_gain - near_push_y,
                                         -config->max_xy_offset_mm,
                                         config->max_xy_offset_mm);
    const float command_z =
        excess_shadow * config->z_lift_per_shadow_ratio_mm +
        excess_near_shadow * config->near_z_lift_per_shadow_ratio_mm;

    float confidence = 1.0f;
    confidence *= clampf_local((float)features->paper_pixels / (float)config->min_paper_pixels, 0.0f, 1.0f);
    confidence *= 1.0f - clampf_local(hand_ratio / config->max_hand_pen_ratio, 0.0f, 1.0f);
    confidence *= clampf_local((state->smoothed_shadow_ratio + state->smoothed_near_shadow_ratio) /
                               config->target_shadow_ratio, 0.0f, 1.0f);

    decision.confidence = confidence;
    if (confidence < config->min_confidence) {
        state->mode = SHADOW_ELIMINATION_IDLE;
        decision.mode = state->mode;
        decision.reason = "low_confidence";
        return decision;
    }

    if (state->stuck_frame_count >= config->stuck_frames_required) {
        state->mode = SHADOW_ELIMINATION_SEARCHING;
        state->search_direction = -state->search_direction;
        state->last_target_mm.x_mm += config->near_camera_dir_y_mm * config->search_step_mm *
            (float)state->search_direction;
        state->last_target_mm.y_mm -= config->near_camera_dir_x_mm * config->search_step_mm *
            (float)state->search_direction;
        state->last_target_mm.z_mm = decision.paper_center_mm.z_mm + command_z;
        state->stuck_frame_count = 0;
    } else {
        state->mode = SHADOW_ELIMINATION_TRACKING;
        state->last_target_mm.x_mm = decision.paper_center_mm.x_mm + command_x;
        state->last_target_mm.y_mm = decision.paper_center_mm.y_mm + command_y;
        state->last_target_mm.z_mm = decision.paper_center_mm.z_mm + command_z;
    }

    decision.mode = state->mode;
    decision.target_mm = state->last_target_mm;
    decision.should_update_pwm = true;
    if (state->mode == SHADOW_ELIMINATION_SEARCHING) {
        decision.reason = "near_camera_shadow_stuck_search";
    } else {
        decision.reason = prefer_near_shadow ? "reduce_near_camera_shadow" : "reduce_shadow_area";
    }
    return decision;
}
