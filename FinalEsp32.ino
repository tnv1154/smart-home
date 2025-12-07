#define BLYNK_TEMPLATE_ID "TMPL6TLL5h3lZ"
#define BLYNK_TEMPLATE_NAME "ThinghiemIoT"
#define BLYNK_AUTH_TOKEN "greVKTUo_NPABVfz2j6kJjxHNVP4kDKS"
#define BLYNK_PRINT Serial   // In log Blynk ra Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

// ================== WIFI ==================
char ssid[] = "Realme 11";     // Tên WiFi
char pass[] = "110520004";  // Mật khẩu WiFi

// ================== PIN CONFIG ==================
// Chân cho phép driver động cơ (R_EN + L_EN)
#define PIN_MOTOR_ENABLE 14

// Chân PWM điều khiển chiều quay
#define PIN_RPWM 27    // Quay phải
#define PIN_LPWM 26    // Quay trái

// Cảm biến DHT11
#define DHTPIN 4
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);
BlynkTimer timer;

// ================== HÀM DỪNG ĐỘNG CƠ ==================
void stopMotor() {
  digitalWrite(PIN_MOTOR_ENABLE, LOW);  // Tắt driver
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);
  Serial.println("Động cơ: DỪNG");
}



// ================== ĐIỀU KHIỂN QUAY PHẢI (V0) ==================
BLYNK_WRITE(V0) {
  int action = param.asInt();   // 1 = bật, 0 = tắt

  if (action == 1) {
    stopMotor();                // Dừng trước khi đổi chiều
    delay(100);                 // Tránh xung dòng
    digitalWrite(PIN_MOTOR_ENABLE, HIGH); // Bật driver
    digitalWrite(PIN_RPWM, HIGH);         // Quay phải
    digitalWrite(PIN_LPWM, LOW);

    // Tắt nút quay trái nếu đang bật
    Blynk.virtualWrite(V3, 0);
    Serial.println("Động cơ: QUAY PHẢI");
  } else {
    stopMotor();
  }
}

// ================== ĐIỀU KHIỂN QUAY TRÁI (V3) ==================
BLYNK_WRITE(V3) {
  int action = param.asInt();   // 1 = bật, 0 = tắt

  if (action == 1) {
    stopMotor();
    delay(100);
    digitalWrite(PIN_MOTOR_ENABLE, HIGH); // Bật driver
    digitalWrite(PIN_RPWM, LOW);
    digitalWrite(PIN_LPWM, HIGH);         // Quay trái

    // Tắt nút quay phải nếu đang bật
    Blynk.virtualWrite(V0, 0);
    Serial.println("Động cơ: QUAY TRÁI");
  } else {
    stopMotor();
  }
}

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);

  // Cấu hình chân điều khiển động cơ
  pinMode(PIN_MOTOR_ENABLE, OUTPUT);
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);

  stopMotor();      // Đảm bảo ban đầu dừng

  // Kết nối Blynk + WiFi
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

}

// ================== LOOP ==================
void loop() {
  Blynk.run();
  timer.run();
}
