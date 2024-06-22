#include <TMCStepper.h>

// #define EN_PIN           7      // Enable
// #define DIR_PIN          2      // Direction
// #define STEP_PIN         3      // Step
// #define SW_SCK           4      // Software Slave Clock (SCK)
// #define SW_TX            5     // SoftwareSerial receive pin
// #define SW_RX            6      // SoftwareSerial transmit pin
#define EN_PIN           13      // Enable
#define DIR_PIN          8     // Direction
#define STEP_PIN         9      // Step
#define SW_SCK           10      // Software Slave Clock (SCK)
#define SW_TX            11     // SoftwareSerial receive pin
#define SW_RX            12      // SoftwareSerial transmit pin
#define DRIVER_ADDRESS   0b00   // TMC2209 Driver address according to MS1 and MS2
#define R_SENSE 0.11f           // SilentStepStick series use 0.11 ...and so does my fysetc TMC2209 (?)

SoftwareSerial SoftSerial(SW_RX, SW_TX);                          // Be sure to connect RX to TX and TX to RX between both devices

TMC2209Stepper TMCdriver(&SoftSerial, R_SENSE, DRIVER_ADDRESS);   // Create TMC driver

void setup() {
  Serial.begin(115200);

  TMCdriver.begin();
}

void loop() {
  auto version = TMCdriver.version();

  if (version != 0x21)
    Serial.println("Driver communication issue");
  else
    Serial.println("Motor found!");
    
  delay(2000);
}