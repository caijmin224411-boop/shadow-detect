#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace shadowcam {

constexpr int kInputW = 96;
constexpr int kInputH = 96;
constexpr int kOutputW = 48;
constexpr int kOutputH = 48;
constexpr int kClasses = 3;

struct QuantParams {
  float scale;
  int zero_point;
};

static inline uint8_t clamp_u8(int v) {
  return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

void DownsampleGrayNearest(const uint8_t* src, int src_w, int src_h, uint8_t* dst96) {
  for (int y = 0; y < kInputH; ++y) {
    const int sy = y * src_h / kInputH;
    for (int x = 0; x < kInputW; ++x) {
      const int sx = x * src_w / kInputW;
      dst96[y * kInputW + x] = src[sy * src_w + sx];
    }
  }
}

void LocalNormalize96(const uint8_t* src96, uint8_t* dst96) {
  constexpr int radius = 4;
  for (int y = 0; y < kInputH; ++y) {
    for (int x = 0; x < kInputW; ++x) {
      int sum = 0;
      int count = 0;
      for (int yy = std::max(0, y - radius); yy <= std::min(kInputH - 1, y + radius); ++yy) {
        for (int xx = std::max(0, x - radius); xx <= std::min(kInputW - 1, x + radius); ++xx) {
          sum += src96[yy * kInputW + xx];
          ++count;
        }
      }
      const int local_mean = sum / count;
      dst96[y * kInputW + x] = clamp_u8(static_cast<int>(src96[y * kInputW + x]) - local_mean + 128);
    }
  }
}

void LocalNormalize96Integral(const uint8_t* src96, uint8_t* dst96, int32_t* integral97x97) {
  constexpr int radius = 4;
  constexpr int stride = kInputW + 1;
  std::memset(integral97x97, 0, stride * (kInputH + 1) * sizeof(int32_t));
  for (int y = 0; y < kInputH; ++y) {
    int row_sum = 0;
    for (int x = 0; x < kInputW; ++x) {
      row_sum += src96[y * kInputW + x];
      integral97x97[(y + 1) * stride + (x + 1)] = integral97x97[y * stride + (x + 1)] + row_sum;
    }
  }
  for (int y = 0; y < kInputH; ++y) {
    const int y0 = std::max(0, y - radius);
    const int y1 = std::min(kInputH - 1, y + radius);
    for (int x = 0; x < kInputW; ++x) {
      const int x0 = std::max(0, x - radius);
      const int x1 = std::min(kInputW - 1, x + radius);
      const int sum =
          integral97x97[(y1 + 1) * stride + (x1 + 1)] -
          integral97x97[y0 * stride + (x1 + 1)] -
          integral97x97[(y1 + 1) * stride + x0] +
          integral97x97[y0 * stride + x0];
      const int count = (y1 - y0 + 1) * (x1 - x0 + 1);
      dst96[y * kInputW + x] = clamp_u8(static_cast<int>(src96[y * kInputW + x]) - (sum / count) + 128);
    }
  }
}

void QuantizeInput96(const uint8_t* gray96, int8_t* input, QuantParams q) {
  for (int i = 0; i < kInputW * kInputH; ++i) {
    const float normalized = static_cast<float>(gray96[i]) / 127.5f - 1.0f;
    const int quantized = static_cast<int>(normalized / q.scale) + q.zero_point;
    input[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
  }
}

void BuildShadowMask(const int8_t* logits, uint8_t* mask48, int shadow_margin = 5) {
  for (int i = 0; i < kOutputW * kOutputH; ++i) {
    const int base = i * kClasses;
    const int paper = logits[base + 0];
    const int shadow = logits[base + 1];
    const int hand_object = logits[base + 2];
    mask48[i] = (shadow > paper + shadow_margin && shadow > hand_object + shadow_margin) ? 255 : 0;
  }
}

int8_t QuantizedLogitThreshold(float probability, QuantParams q) {
  probability = std::max(0.001f, std::min(0.999f, probability));
  const float logit = std::log(probability / (1.0f - probability));
  const int quantized = static_cast<int>(std::round(logit / q.scale)) + q.zero_point;
  return static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
}

void BuildModularShadowMask(
    const int8_t* shadow_logits,
    const int8_t* exclude_logits,
    uint8_t* mask48,
    QuantParams shadow_q,
    QuantParams exclude_q,
    float shadow_probability_threshold = 0.45f,
    float exclude_probability_threshold = 0.50f) {
  const int8_t shadow_threshold = QuantizedLogitThreshold(shadow_probability_threshold, shadow_q);
  const int8_t exclude_threshold = QuantizedLogitThreshold(exclude_probability_threshold, exclude_q);
  for (int i = 0; i < kOutputW * kOutputH; ++i) {
    const bool is_shadow = shadow_logits[i] >= shadow_threshold;
    const bool is_excluded = exclude_logits[i] >= exclude_threshold;
    mask48[i] = (is_shadow && !is_excluded) ? 255 : 0;
  }
}

void RemoveTinyIslands(uint8_t* mask48, uint8_t* scratch48, int min_neighbors = 2) {
  std::memcpy(scratch48, mask48, kOutputW * kOutputH);
  for (int y = 1; y < kOutputH - 1; ++y) {
    for (int x = 1; x < kOutputW - 1; ++x) {
      const int idx = y * kOutputW + x;
      if (scratch48[idx] == 0) {
        continue;
      }
      int neighbors = 0;
      for (int yy = y - 1; yy <= y + 1; ++yy) {
        for (int xx = x - 1; xx <= x + 1; ++xx) {
          if (xx == x && yy == y) {
            continue;
          }
          neighbors += scratch48[yy * kOutputW + xx] > 0 ? 1 : 0;
        }
      }
      if (neighbors < min_neighbors) {
        mask48[idx] = 0;
      }
    }
  }
}

void CloseSmallHoles(uint8_t* mask48, uint8_t* scratch48) {
  std::memcpy(scratch48, mask48, kOutputW * kOutputH);
  for (int y = 1; y < kOutputH - 1; ++y) {
    for (int x = 1; x < kOutputW - 1; ++x) {
      const int idx = y * kOutputW + x;
      if (scratch48[idx] != 0) {
        continue;
      }
      int neighbors = 0;
      for (int yy = y - 1; yy <= y + 1; ++yy) {
        for (int xx = x - 1; xx <= x + 1; ++xx) {
          neighbors += scratch48[yy * kOutputW + xx] > 0 ? 1 : 0;
        }
      }
      if (neighbors >= 7) {
        mask48[idx] = 255;
      }
    }
  }
}

}  // namespace shadowcam
