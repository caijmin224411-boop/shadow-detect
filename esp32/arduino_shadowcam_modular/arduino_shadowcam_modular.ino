#include <Arduino.h>
#include "esp_camera.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../shadowcam_runtime.cpp"
#include "../paper_shadow_modular_istd_full_b12_model_data.cc"

// AI-Thinker ESP32-CAM pin map.
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

namespace {

constexpr float kShadowThreshold = 0.70f;
constexpr float kExcludeThreshold = 0.65f;
constexpr int kShadowOutputIndex = 1;
constexpr int kExcludeOutputIndex = 0;
constexpr int kArenaSize = 280 * 1024;

alignas(16) uint8_t tensor_arena[kArenaSize];
uint8_t gray96[shadowcam::kInputW * shadowcam::kInputH];
uint8_t norm96[shadowcam::kInputW * shadowcam::kInputH];
uint8_t mask48[shadowcam::kOutputW * shadowcam::kOutputH];
uint8_t scratch48[shadowcam::kOutputW * shadowcam::kOutputH];
int32_t integral97x97[(shadowcam::kInputW + 1) * (shadowcam::kInputH + 1)];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* shadow_output = nullptr;
TfLiteTensor* exclude_output = nullptr;

uint32_t frame_count = 0;
uint32_t last_report_ms = 0;
uint32_t last_report_frames = 0;

bool InitCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, FRAMESIZE_QQVGA);
    sensor->set_pixformat(sensor, PIXFORMAT_GRAYSCALE);
    sensor->set_gain_ctrl(sensor, 1);
    sensor->set_exposure_ctrl(sensor, 1);
    sensor->set_whitebal(sensor, 0);
  }
  return true;
}

bool InitTflm() {
  model = tflite::GetModel(g_paper_shadow_modular_istd_full_b12_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("schema mismatch: model=%d runtime=%d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddConcatenation();
  resolver.AddExpandDims();
  resolver.AddTile();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddReshape();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed; increase kArenaSize or simplify the model");
    return false;
  }

  input = interpreter->input(0);
  shadow_output = interpreter->output(kShadowOutputIndex);
  exclude_output = interpreter->output(kExcludeOutputIndex);

  Serial.printf("arena_used=%u arena_size=%u\n",
                static_cast<unsigned>(interpreter->arena_used_bytes()),
                static_cast<unsigned>(kArenaSize));
  Serial.printf("input q scale=%.9f zero=%d\n", input->params.scale, input->params.zero_point);
  Serial.printf("shadow output index=%d q scale=%.9f zero=%d\n",
                kShadowOutputIndex, shadow_output->params.scale, shadow_output->params.zero_point);
  Serial.printf("exclude output index=%d q scale=%.9f zero=%d\n",
                kExcludeOutputIndex, exclude_output->params.scale, exclude_output->params.zero_point);
  return true;
}

int CountMaskPixels(const uint8_t* mask) {
  int count = 0;
  for (int i = 0; i < shadowcam::kOutputW * shadowcam::kOutputH; ++i) {
    count += mask[i] > 0 ? 1 : 0;
  }
  return count;
}

void PrintReport(uint32_t now_ms, uint32_t preprocess_us, uint32_t invoke_us, uint32_t post_us) {
  const uint32_t elapsed_ms = now_ms - last_report_ms;
  if (elapsed_ms < 1000) {
    return;
  }
  const uint32_t frames = frame_count - last_report_frames;
  const float fps = frames * 1000.0f / elapsed_ms;
  const int mask_pixels = CountMaskPixels(mask48);
  const float mask_ratio = mask_pixels / float(shadowcam::kOutputW * shadowcam::kOutputH);
  Serial.printf(
      "frames=%lu fps=%.2f pre_us=%lu infer_us=%lu post_us=%lu mask=%.3f heap=%u psram=%u\n",
      static_cast<unsigned long>(frame_count), fps,
      static_cast<unsigned long>(preprocess_us),
      static_cast<unsigned long>(invoke_us),
      static_cast<unsigned long>(post_us),
      mask_ratio,
      static_cast<unsigned>(ESP.getFreeHeap()),
      static_cast<unsigned>(ESP.getFreePsram()));
  last_report_ms = now_ms;
  last_report_frames = frame_count;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("shadowcam modular ISTD full b12");
  Serial.printf("psram=%s heap=%u psram_free=%u\n",
                psramFound() ? "yes" : "no",
                static_cast<unsigned>(ESP.getFreeHeap()),
                static_cast<unsigned>(ESP.getFreePsram()));

  if (!InitCamera()) {
    while (true) delay(1000);
  }
  if (!InitTflm()) {
    while (true) delay(1000);
  }
  last_report_ms = millis();
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("camera frame failed");
    delay(10);
    return;
  }

  const uint32_t t0 = micros();
  shadowcam::DownsampleGrayNearest(fb->buf, fb->width, fb->height, gray96);
  esp_camera_fb_return(fb);
  shadowcam::LocalNormalize96Integral(gray96, norm96, integral97x97);
  shadowcam::QuantizeInput96(norm96, input->data.int8,
                             {input->params.scale, input->params.zero_point});
  const uint32_t t1 = micros();

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(100);
    return;
  }
  const uint32_t t2 = micros();

  shadowcam::BuildModularShadowMask(
      shadow_output->data.int8, exclude_output->data.int8, mask48,
      {shadow_output->params.scale, shadow_output->params.zero_point},
      {exclude_output->params.scale, exclude_output->params.zero_point},
      kShadowThreshold, kExcludeThreshold);
  shadowcam::RemoveTinyIslands(mask48, scratch48);
  shadowcam::CloseSmallHoles(mask48, scratch48);
  const uint32_t t3 = micros();

  ++frame_count;
  PrintReport(millis(), t1 - t0, t2 - t1, t3 - t2);
}
