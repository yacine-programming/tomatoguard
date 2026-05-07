/*
 * ============================================================
 *  TomatoGuard - Station Météo ESP32-WROOM-32
 *  Envoi WiFi → Render.com Flask
 *
 *  ⚡ MODE TEST : envoi toutes les 2 minutes
 *     (côté serveur, 2 min = 1 jour climat)
 * ============================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_BMP085.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ============================================================
//  CONFIGURATION — À MODIFIER
// ============================================================
const char* WIFI_SSID     = "TON_WIFI";
const char* WIFI_PASSWORD = "TON_MOT_DE_PASSE";

// URL Render.com (ne change plus une fois déployé !)
const char* SERVER_URL    = "https://tomatoguard.onrender.com/api/climate";

// Intervalle d'envoi : 2 minutes pour le mode test
// (mettre 3600000 = 1h pour le mode normal en production)
constexpr unsigned long INTERVAL_ENVOI_MS = 120000UL;  // 2 min

// ============================================================
//  PINOUT
// ============================================================
constexpr int PIN_DHT         = 4;
constexpr int PIN_PLUVIO      = 14;
constexpr int PIN_VITESSE     = 36;
constexpr int PIN_DIRECTION   = 39;
constexpr int PIN_WM_EXC_A    = 16;
constexpr int PIN_WM_EXC_B    = 17;
constexpr int PIN_WM_ADC      = 34;
constexpr int PIN_RS485_TX    = 33;
constexpr int PIN_RS485_RX    = 32;
constexpr int PIN_RS485_DE_RE = 27;
constexpr int PIN_I2C_SDA     = 21;
constexpr int PIN_I2C_SCL     = 22;

// ============================================================
//  INTERVALLES INTERNES
// ============================================================
constexpr unsigned long INTERVAL_RAPIDE = 2000UL;
constexpr unsigned long INTERVAL_PLUVIO = 1000UL;
constexpr unsigned long INTERVAL_WM     = 30000UL;
constexpr unsigned long INTERVAL_NPK    = 30000UL;

// ============================================================
//  CONSTANTES
// ============================================================
#define DHTTYPE DHT22
constexpr float PLUVIO_RESOLUTION  = 0.2f;
constexpr unsigned long DEBOUNCE_MS = 200UL;
constexpr float VITESSE_MAX        = 30.0f;
constexpr float TENSION_MAX        = 3.30f;
constexpr float V_ZERO_DEFAUT      = 0.255f;
constexpr float V_ZERO_MIN_VALIDE  = 0.05f;
constexpr float V_ZERO_MAX_VALIDE  = 0.80f;
constexpr int   N_ECHANTILLONS     = 50;
constexpr int   DELAI_CALIB_MS     = 100;
constexpr int   N_LECTURES_VENT    = 10;
constexpr float ADC_MAX_VOLTAGE    = 3.3f;
constexpr float WM_R_SERIES       = 10000.0f;
constexpr float WM_DEFAULT_TEMP   = 24.0f;
constexpr float WM_R_SHORT        = 300.0f;
constexpr float WM_R_OPEN         = 50000.0f;
constexpr int   WM_CB_OPEN        = 255;
constexpr unsigned long NPK_BAUD       = 9600;
constexpr unsigned long NPK_TIMEOUT_MS = 1000;

// ============================================================
//  OBJETS GLOBAUX
// ============================================================
DHT             dht(PIN_DHT, DHTTYPE);
Adafruit_BMP085 bmp;
HardwareSerial  RS485Serial(2);
float           v_zero_calibre = V_ZERO_DEFAUT;

struct Donnees {
    float temperature_air = NAN, humidite_air = NAN;
    bool dht_ok = false;
    float pluie_recap_mm = 0.0f;
    float vent_vitesse_kmh = 0.0f;
    int vent_angle = 0;
    float sol_resistance = 0.0f;
    int sol_cb = 0;
    bool wm_ok = false;
    uint16_t npk_n = 0, npk_p = 0, npk_k = 0;
    bool npk_ok = false;
    float pression_hpa = NAN;
    bool bmp_ok = false;
};
Donnees d;

// ============================================================
//  PLUVIOMÈTRE IRQ
// ============================================================
volatile int impulsionCount = 0;
volatile unsigned long lastInterrupt = 0;
void IRAM_ATTR onBascule() {
    unsigned long now = millis();
    if (now - lastInterrupt > DEBOUNCE_MS) { impulsionCount++; lastInterrupt = now; }
}

// ============================================================
//  WIFI
// ============================================================
void connecterWifi() {
    if (WiFi.status() == WL_CONNECTED) return;
    Serial.printf("WiFi %s ...", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int n = 0;
    while (WiFi.status() != WL_CONNECTED && n < 20) {
        delay(500); Serial.print("."); n++;
    }
    if (WiFi.status() == WL_CONNECTED)
        Serial.printf("\n✓ IP: %s\n", WiFi.localIP().toString().c_str());
    else Serial.println("\n✗ Echec WiFi");
}

// ============================================================
//  ENVOI DONNÉES → SERVEUR
// ============================================================
void envoyerDonnees() {
    if (WiFi.status() != WL_CONNECTED) {
        connecterWifi();
        if (WiFi.status() != WL_CONNECTED) return;
    }

    float leaf_wetness = 0.0f;
    if (!isnan(d.humidite_air))
        leaf_wetness = min(1.0f, (d.humidite_air / 100.0f + d.pluie_recap_mm / 10.0f) / 2.0f);

    StaticJsonDocument<512> doc;
    doc["temperature"]   = isnan(d.temperature_air) ? 0 : (float)round(d.temperature_air * 10) / 10;
    doc["humidity"]      = isnan(d.humidite_air)    ? 0 : (float)round(d.humidite_air * 10) / 10;
    doc["precipitation"] = (float)round(d.pluie_recap_mm * 100) / 100;
    doc["wind_speed"]    = (float)round(d.vent_vitesse_kmh * 10) / 10;
    doc["wind_direction"]= d.vent_angle;
    doc["leaf_wetness"]  = (float)round(leaf_wetness * 1000) / 1000;
    doc["pressure_hpa"]  = isnan(d.pression_hpa) ? 0 : (float)round(d.pression_hpa * 10) / 10;
    doc["soil_cb"]       = d.sol_cb;
    doc["soil_resistance"] = (int)d.sol_resistance;
    doc["npk_n"]         = d.npk_n;
    doc["npk_p"]         = d.npk_p;
    doc["npk_k"]         = d.npk_k;

    String payload;
    serializeJson(doc, payload);

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(15000);
    int code = http.POST(payload);
    if (code == 200) {
        Serial.printf("✓ Envoi OK (T=%.1f H=%.1f P=%.1fmm)\n",
                      d.temperature_air, d.humidite_air, d.pluie_recap_mm);
        d.pluie_recap_mm = 0.0f;
    } else {
        Serial.printf("✗ HTTP %d\n", code);
    }
    http.end();
}

// ============================================================
//  CAPTEURS
// ============================================================
void lireDHT() {
    float h = dht.readHumidity(), t = dht.readTemperature();
    if (isnan(h) || isnan(t)) d.dht_ok = false;
    else { d.humidite_air = h; d.temperature_air = t; d.dht_ok = true; }
}

void lirePluviometre() {
    if (impulsionCount > 0) {
        noInterrupts();
        int n = impulsionCount; impulsionCount = 0;
        interrupts();
        d.pluie_recap_mm += n * PLUVIO_RESOLUTION;
    }
}

void calibrerZero() {
    Serial.println("Calibration anémomètre...");
    float s = 0;
    for (int i = 0; i < N_ECHANTILLONS; i++) {
        s += analogRead(PIN_VITESSE) * (3.30f / 4095.0f);
        delay(DELAI_CALIB_MS);
    }
    float v = s / N_ECHANTILLONS;
    v_zero_calibre = (v >= V_ZERO_MIN_VALIDE && v <= V_ZERO_MAX_VALIDE) ? v : V_ZERO_DEFAUT;
    Serial.printf("V_zero = %.3fV\n", v_zero_calibre);
}

void lireVent() {
    float s = 0;
    for (int i = 0; i < N_LECTURES_VENT; i++) {
        s += analogRead(PIN_VITESSE) * (3.30f / 4095.0f); delay(5);
    }
    float t = s / N_LECTURES_VENT;
    float v = (t <= v_zero_calibre) ? 0.0f
            : min((t - v_zero_calibre) * VITESSE_MAX / (TENSION_MAX - v_zero_calibre), VITESSE_MAX);
    d.vent_vitesse_kmh = v * 3.6f;

    int a = (int)(analogRead(PIN_DIRECTION) * (3.30f / 4095.0f) * 360.0f / 3.30f);
    d.vent_angle = constrain(a, 0, 359);
}

float mesurerWM() {
    digitalWrite(PIN_WM_EXC_A, HIGH); digitalWrite(PIN_WM_EXC_B, LOW);
    delayMicroseconds(80);
    float v1 = analogRead(PIN_WM_ADC) * (ADC_MAX_VOLTAGE / 4095.0f);
    digitalWrite(PIN_WM_EXC_A, LOW); delay(1);
    digitalWrite(PIN_WM_EXC_B, HIGH);
    delayMicroseconds(80);
    float v2 = analogRead(PIN_WM_ADC) * (ADC_MAX_VOLTAGE / 4095.0f);
    digitalWrite(PIN_WM_EXC_A, LOW); digitalWrite(PIN_WM_EXC_B, LOW);
    if (v1 <= 0.01f || v2 <= 0.01f) return -1;
    float r1 = WM_R_SERIES * (ADC_MAX_VOLTAGE - v1) / v1;
    float r2 = WM_R_SERIES * v2 / (ADC_MAX_VOLTAGE - v2);
    return (r1 + r2) / 2.0f;
}

int wmCB(float r, float t) {
    if (r <= 0 || r >= WM_R_OPEN) return WM_CB_OPEN;
    if (r < WM_R_SHORT) return 240;
    if (r <= 550.0f) return 0;
    float rK = r / 1000.0f, tf = 1.0f + 0.018f * (t - 24.0f);
    int cb;
    if (r > 8000.0f) cb = (-2.246f - 5.239f*rK*tf - 0.06756f*rK*rK*tf*tf);
    else if (r > 1000.0f) {
        float dn = 1.0f - 0.009733f*rK - 0.01205f*t;
        cb = (dn != 0) ? (-3.213f*rK - 4.093f) / dn : 0;
    } else { cb = -(rK*23.156f - 12.736f) * tf; if (cb < 0) cb = 0; }
    return constrain(cb, 0, 240);
}

void lireWM() {
    float t = d.dht_ok ? d.temperature_air : WM_DEFAULT_TEMP;
    float r = mesurerWM();
    if (r > 0) { d.sol_resistance = r; d.sol_cb = wmCB(r, t); d.wm_ok = true; }
    else d.wm_ok = false;
}

uint16_t crc16(const uint8_t* data, size_t n) {
    uint16_t c = 0xFFFF;
    for (size_t i = 0; i < n; i++) {
        c ^= data[i];
        for (int j = 0; j < 8; j++) c = (c & 1) ? (c >> 1) ^ 0xA001 : c >> 1;
    }
    return c;
}

void lireNPK() {
    uint8_t req[8] = {0x01, 0x03, 0x00, 0x1E, 0x00, 0x03, 0, 0};
    uint16_t c = crc16(req, 6);
    req[6] = c & 0xFF; req[7] = (c >> 8) & 0xFF;
    while (RS485Serial.available()) RS485Serial.read();
    digitalWrite(PIN_RS485_DE_RE, HIGH); delayMicroseconds(50);
    RS485Serial.write(req, 8); RS485Serial.flush();
    delayMicroseconds(50); digitalWrite(PIN_RS485_DE_RE, LOW);
    uint8_t rep[16]; size_t i = 0; unsigned long t = millis();
    while (millis() - t < NPK_TIMEOUT_MS) {
        if (RS485Serial.available() && i < 16) { rep[i++] = RS485Serial.read(); t = millis(); }
        if (i >= 5 && millis() - t > 50) break;
    }
    if (i < 9 || rep[0] != 0x01 || rep[1] != 0x03) { d.npk_ok = false; return; }
    uint16_t cr = (rep[i-1] << 8) | rep[i-2], cc = crc16(rep, i-2);
    if (cr != cc) { d.npk_ok = false; return; }
    d.npk_n = (rep[3] << 8) | rep[4];
    d.npk_p = (rep[5] << 8) | rep[6];
    d.npk_k = (rep[7] << 8) | rep[8];
    d.npk_ok = true;
}

void lireBMP() {
    if (!d.bmp_ok) return;
    float p = bmp.readPressure() / 100.0f;
    if (p > 800 && p < 1200) d.pression_hpa = p;
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=== TomatoGuard ESP32 ===");
    Serial.printf("Mode TEST: envoi toutes les %lu sec\n", INTERVAL_ENVOI_MS / 1000);

    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    calibrerZero();
    dht.begin();
    pinMode(PIN_PLUVIO, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_PLUVIO), onBascule, FALLING);
    pinMode(PIN_WM_EXC_A, OUTPUT); pinMode(PIN_WM_EXC_B, OUTPUT); pinMode(PIN_WM_ADC, INPUT);
    digitalWrite(PIN_WM_EXC_A, LOW); digitalWrite(PIN_WM_EXC_B, LOW);
    pinMode(PIN_RS485_DE_RE, OUTPUT); digitalWrite(PIN_RS485_DE_RE, LOW);
    RS485Serial.begin(NPK_BAUD, SERIAL_8N1, PIN_RS485_RX, PIN_RS485_TX);
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    d.bmp_ok = bmp.begin();

    connecterWifi();
    Serial.println("Mesures démarrées\n");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
    static unsigned long lastRapide=0, lastPluvio=0, lastWM=0, lastNPK=0, lastEnvoi=0;
    unsigned long now = millis();

    if (now - lastPluvio >= INTERVAL_PLUVIO) { lastPluvio = now; lirePluviometre(); }
    if (now - lastRapide >= INTERVAL_RAPIDE) { lastRapide = now; lireDHT(); lireVent(); lireBMP(); }
    if (now - lastWM     >= INTERVAL_WM)     { lastWM = now; lireWM(); }
    if (now - lastNPK    >= INTERVAL_NPK)    { lastNPK = now; lireNPK(); }
    if (now - lastEnvoi  >= INTERVAL_ENVOI_MS) { lastEnvoi = now; envoyerDonnees(); }

    delay(10);
}
