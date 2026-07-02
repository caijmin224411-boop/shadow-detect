#include "shadow_ai.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const unsigned char g_paper_shadow_p4_model_data[];
extern const unsigned int g_paper_shadow_p4_model_data_len;

namespace {

constexpr char kTag[] = "shadow_ai";
constexpr int kInputW = 96;
constexpr int kInputH = 96;
constexpr int kMaskW = 48;
constexpr int kMaskH = 48;
constexpr int kArenaBytes = 420 * 1024;
constexpr float kShadowThreshold = 0.70f;
constexpr float kExcludeThreshold = 0.65f;
constexpr int kRunEveryFrames = 1;

constexpr int kShadowOutputIndex = 1;
constexpr int kExcludeOutputIndex = 0;
constexpr uint16_t kContourColor = 0xffe0;  // RGB565 yellow

const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_shadow_output = nullptr;
TfLiteTensor *g_exclude_output = nullptr;
uint8_t *g_arena = nullptr;
uint8_t g_gray[kInputW * kInputH];
uint8_t g_norm[kInputW * kInputH];
uint8_t g_mask[kMaskW * kMaskH];
uint8_t g_exclude_mask[kMaskW * kMaskH];
uint8_t g_scratch[kMaskW * kMaskH];
int32_t g_integral[(kInputW + 1) * (kInputH + 1)];
int g_frame_counter = 0;
bool g_ready = false;
shadow_frame_features_t g_last_features = {};
bool g_has_last_features = false;

inline uint8_t clamp_u8(int value) {
    return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}

inline int8_t clamp_i8(int value) {
    return static_cast<int8_t>(std::max(-128, std::min(127, value)));
}

inline uint8_t rgb565_to_luma(uint16_t pixel) {
    const int r5 = (pixel >> 11) & 0x1f;
    const int g6 = (pixel >> 5) & 0x3f;
    const int b5 = pixel & 0x1f;
    const int r = (r5 * 255 + 15) / 31;
    const int g = (g6 * 255 + 31) / 63;
    const int b = (b5 * 255 + 15) / 31;
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

void downsample_rgb565_to_gray96(const uint16_t *frame, int width, int height, uint8_t *gray96) {
    for (int y = 0; y < kInputH; ++y) {
        const int sy = y * height / kInputH;
        for (int x = 0; x < kInputW; ++x) {
            const int sx = x * width / kInputW;
            gray96[y * kInputW + x] = rgb565_to_luma(frame[sy * width + sx]);
        }
    }
}

void local_normalize96(const uint8_t *src, uint8_t *dst) {
    constexpr int radius = 4;
    constexpr int stride = kInputW + 1;
    std::memset(g_integral, 0, sizeof(g_integral));
    for (int y = 0; y < kInputH; ++y) {
        int row_sum = 0;
        for (int x = 0; x < kInputW; ++x) {
            row_sum += src[y * kInputW + x];
            g_integral[(y + 1) * stride + (x + 1)] = g_integral[y * stride + (x + 1)] + row_sum;
        }
    }

    for (int y = 0; y < kInputH; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(kInputH - 1, y + radius);
        for (int x = 0; x < kInputW; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(kInputW - 1, x + radius);
            const int sum = g_integral[(y1 + 1) * stride + (x1 + 1)] -
                            g_integral[y0 * stride + (x1 + 1)] -
                            g_integral[(y1 + 1) * stride + x0] +
                            g_integral[y0 * stride + x0];
            const int count = (y1 - y0 + 1) * (x1 - x0 + 1);
            dst[y * kInputW + x] = clamp_u8(static_cast<int>(src[y * kInputW + x]) - sum / count + 128);
        }
    }
}

void quantize_input(const uint8_t *gray96, TfLiteTensor *input) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    int8_t *dst = input->data.int8;
    for (int i = 0; i < kInputW * kInputH; ++i) {
        const float normalized = static_cast<float>(gray96[i]) / 127.5f - 1.0f;
        dst[i] = clamp_i8(static_cast<int>(std::lround(normalized / scale)) + zero_point);
    }
}

int8_t quantized_logit_threshold(float probability, const TfLiteTensor *tensor) {
    probability = std::max(0.001f, std::min(0.999f, probability));
    const float logit = std::log(probability / (1.0f - probability));
    return clamp_i8(static_cast<int>(std::lround(logit / tensor->params.scale)) + tensor->params.zero_point);
}

void build_mask48(void) {
    const int8_t shadow_threshold = quantized_logit_threshold(kShadowThreshold, g_shadow_output);
    const int8_t exclude_threshold = quantized_logit_threshold(kExcludeThreshold, g_exclude_output);
    const int8_t *shadow = g_shadow_output->data.int8;
    const int8_t *exclude = g_exclude_output->data.int8;

    for (int i = 0; i < kMaskW * kMaskH; ++i) {
        g_exclude_mask[i] = (exclude[i] >= exclude_threshold) ? 255 : 0;
        g_mask[i] = (shadow[i] >= shadow_threshold && exclude[i] < exclude_threshold) ? 255 : 0;
    }
}

void update_last_features(void) {
    shadow_frame_features_t features = {};
    features.paper_pixels = kMaskW * kMaskH;
    features.has_paper = true;
    features.paper_centroid_px.u = (kMaskW - 1) * 0.5f;
    features.paper_centroid_px.v = (kMaskH - 1) * 0.5f;

    int shadow_u_sum = 0;
    int shadow_v_sum = 0;
    int near_u_sum = 0;
    int near_v_sum = 0;
    int exclude_count = 0;
    constexpr int kNearStartY = (kMaskH * 2) / 3;

    for (int y = 0; y < kMaskH; ++y) {
        for (int x = 0; x < kMaskW; ++x) {
            const int idx = y * kMaskW + x;
            if (g_exclude_mask[idx] > 0) {
                ++exclude_count;
            }
            if (g_mask[idx] == 0) {
                continue;
            }
            ++features.shadow_on_paper_pixels;
            shadow_u_sum += x;
            shadow_v_sum += y;
            if (y >= kNearStartY) {
                ++features.near_shadow_on_paper_pixels;
                near_u_sum += x;
                near_v_sum += y;
            }
        }
    }

    features.hand_pen_on_paper_pixels = exclude_count;
    features.has_shadow = features.shadow_on_paper_pixels > 0;
    features.has_near_shadow = features.near_shadow_on_paper_pixels > 0;

    if (features.has_shadow) {
        features.shadow_centroid_px.u =
            static_cast<float>(shadow_u_sum) / static_cast<float>(features.shadow_on_paper_pixels);
        features.shadow_centroid_px.v =
            static_cast<float>(shadow_v_sum) / static_cast<float>(features.shadow_on_paper_pixels);
    }
    if (features.has_near_shadow) {
        features.near_shadow_centroid_px.u =
            static_cast<float>(near_u_sum) / static_cast<float>(features.near_shadow_on_paper_pixels);
        features.near_shadow_centroid_px.v =
            static_cast<float>(near_v_sum) / static_cast<float>(features.near_shadow_on_paper_pixels);
    }

    g_last_features = features;
    g_has_last_features = true;
}

void remove_tiny_islands(void) {
    std::memcpy(g_scratch, g_mask, sizeof(g_mask));
    for (int y = 1; y < kMaskH - 1; ++y) {
        for (int x = 1; x < kMaskW - 1; ++x) {
            const int idx = y * kMaskW + x;
            if (g_scratch[idx] == 0) {
                continue;
            }
            int neighbors = 0;
            for (int yy = y - 1; yy <= y + 1; ++yy) {
                for (int xx = x - 1; xx <= x + 1; ++xx) {
                    if (xx != x || yy != y) {
                        neighbors += g_scratch[yy * kMaskW + xx] > 0 ? 1 : 0;
                    }
                }
            }
            if (neighbors < 2) {
                g_mask[idx] = 0;
            }
        }
    }
}

bool is_boundary_cell(int x, int y) {
    if (g_mask[y * kMaskW + x] == 0) {
        return false;
    }
    if (x == 0 || y == 0 || x == kMaskW - 1 || y == kMaskH - 1) {
        return true;
    }
    return g_mask[y * kMaskW + x - 1] == 0 ||
           g_mask[y * kMaskW + x + 1] == 0 ||
           g_mask[(y - 1) * kMaskW + x] == 0 ||
           g_mask[(y + 1) * kMaskW + x] == 0;
}

void draw_hline(uint16_t *frame, int width, int height, int x0, int x1, int y) {
    if (y < 0 || y >= height) {
        return;
    }
    x0 = std::max(0, x0);
    x1 = std::min(width - 1, x1);
    for (int x = x0; x <= x1; ++x) {
        frame[y * width + x] = kContourColor;
    }
}

void draw_vline(uint16_t *frame, int width, int height, int x, int y0, int y1) {
    if (x < 0 || x >= width) {
        return;
    }
    y0 = std::max(0, y0);
    y1 = std::min(height - 1, y1);
    for (int y = y0; y <= y1; ++y) {
        frame[y * width + x] = kContourColor;
    }
}

void draw_mask_contours(uint16_t *frame, int width, int height) {
    for (int y = 0; y < kMaskH; ++y) {
        for (int x = 0; x < kMaskW; ++x) {
            if (!is_boundary_cell(x, y)) {
                continue;
            }
            const int x0 = x * width / kMaskW;
            const int x1 = ((x + 1) * width / kMaskW) - 1;
            const int y0 = y * height / kMaskH;
            const int y1 = ((y + 1) * height / kMaskH) - 1;
            draw_hline(frame, width, height, x0, x1, y0);
            draw_hline(frame, width, height, x0, x1, y1);
            draw_vline(frame, width, height, x0, y0, y1);
            draw_vline(frame, width, height, x1, y0, y1);
        }
    }
}

}  // namespace

extern "C" bool shadow_ai_init(void) {
    if (g_ready) {
        return true;
    }

    if (g_paper_shadow_p4_model_data_len == 0) {
        ESP_LOGE(kTag, "model array is empty");
        return false;
    }

    g_model = tflite::GetModel(g_paper_shadow_p4_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "schema mismatch: model=%ld runtime=%d", static_cast<long>(g_model->version()), TFLITE_SCHEMA_VERSION);
        return false;
    }

    g_arena = static_cast<uint8_t *>(heap_caps_malloc(kArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (g_arena == nullptr) {
        g_arena = static_cast<uint8_t *>(heap_caps_malloc(kArenaBytes, MALLOC_CAP_8BIT));
    }
    if (g_arena == nullptr) {
        ESP_LOGE(kTag, "failed to allocate %d byte tensor arena", kArenaBytes);
        return false;
    }

    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();

    static tflite::MicroInterpreter static_interpreter(g_model, resolver, g_arena, kArenaBytes);
    g_interpreter = &static_interpreter;
    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors failed");
        return false;
    }

    g_input = g_interpreter->input(0);
    g_shadow_output = g_interpreter->output(kShadowOutputIndex);
    g_exclude_output = g_interpreter->output(kExcludeOutputIndex);
    if (g_input == nullptr || g_shadow_output == nullptr || g_exclude_output == nullptr) {
        ESP_LOGE(kTag, "missing input/output tensors");
        return false;
    }
    if (g_input->type != kTfLiteInt8 || g_shadow_output->type != kTfLiteInt8 || g_exclude_output->type != kTfLiteInt8) {
        ESP_LOGE(kTag, "expected full int8 tensors");
        return false;
    }

    ESP_LOGI(kTag, "ready: model=%u bytes arena=%d input(scale=%.6f,zp=%d) shadow(scale=%.6f,zp=%d) exclude(scale=%.6f,zp=%d)",
             g_paper_shadow_p4_model_data_len,
             kArenaBytes,
             static_cast<double>(g_input->params.scale),
             g_input->params.zero_point,
             static_cast<double>(g_shadow_output->params.scale),
             g_shadow_output->params.zero_point,
             static_cast<double>(g_exclude_output->params.scale),
             g_exclude_output->params.zero_point);
    g_ready = true;
    return true;
}

extern "C" bool shadow_ai_process_frame(uint16_t *rgb565_frame, int width, int height) {
    if (!g_ready || rgb565_frame == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    g_frame_counter++;
    if ((g_frame_counter % kRunEveryFrames) != 0) {
        draw_mask_contours(rgb565_frame, width, height);
        return true;
    }

    downsample_rgb565_to_gray96(rgb565_frame, width, height, g_gray);
    local_normalize96(g_gray, g_norm);
    quantize_input(g_norm, g_input);

    if (g_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGW(kTag, "Invoke failed");
        return false;
    }

    build_mask48();
    remove_tiny_islands();
    update_last_features();
    draw_mask_contours(rgb565_frame, width, height);
    return true;
}

extern "C" bool shadow_ai_get_last_features(shadow_frame_features_t *features) {
    if (features == nullptr || !g_has_last_features) {
        return false;
    }
    *features = g_last_features;
    return true;
}
