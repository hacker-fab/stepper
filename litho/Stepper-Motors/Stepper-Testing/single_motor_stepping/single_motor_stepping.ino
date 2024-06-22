#include <TMCStepper.h>

#define EN_PIN           7      // Enable
#define DIR_PIN          2      // Direction
#define STEP_PIN         3      // Step
#define SW_SCK           4      // Software Slave Clock (SCK)
#define SW_TX            5     // SoftwareSerial receive pin
#define SW_RX            6      // SoftwareSerial transmit pin
// #define EN_PIN           13      // Enable
// #define DIR_PIN          8     // Direction
// #define STEP_PIN         9      // Step
// #define SW_SCK           10      // Software Slave Clock (SCK)
// #define SW_TX            11     // SoftwareSerial receive pin
// #define SW_RX            12      // SoftwareSerial transmit pin
#define DRIVER_ADDRESS   0b00   // TMC2209 Driver address according to MS1 and MS2
#define R_SENSE 0.11f           // SilentStepStick series use 0.11 ...and so does my fysetc TMC2209 (?)

// TMC2209Stepper driver(&SERIAL_PORT, R_SENSE, DRIVER_ADDRESS);
TMC2209Stepper driver(SW_RX, SW_TX, R_SENSE, DRIVER_ADDRESS);

uint16_t revolution = 400; //steps per revolution
uint16_t usteps = 0;

void setup() {
  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);      // Enable driver in hardware

                                  // Enable one according to your setup
//SPI.begin();                    // SPI drivers
//SERIAL_PORT.begin(115200);      // HW UART drivers
//driver.beginSerial(115200);     // SW UART drivers

  driver.begin();                 //  SPI: Init CS pins and possible SW SPI pins
                                  // UART: Init SW UART (if selected) with default 115200 baudrate
  driver.toff(5);                 // Enables driver in software
  driver.rms_current(1200);        // Set motor RMS current
  driver.microsteps(usteps);          // Set microsteps to 1/16th

//driver.en_pwm_mode(true);       // Toggle stealthChop on TMC2130/2160/5130/5160
//driver.en_spreadCycle(false);   // Toggle spreadCycle on TMC2208/2209/2224
  driver.pwm_autoscale(true);     // Needed for stealthChop
}

bool shaft = false;

void loop() {
  // Run steps and switch direction in software
  for (uint16_t i = revolution * 1; i>0; i--) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(160);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(160);
  }
  shaft = !shaft;
  driver.shaft(shaft);
  digitalWrite(DIR_PIN,shaft);
}
