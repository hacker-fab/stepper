#include <TMCStepper.h>         // TMCstepper - https://github.com/teemuatlut/TMCStepper
#include <SoftwareSerial.h>     // Software serial for the UART to TMC2209 - https://www.arduino.cc/en/Reference/softwareSerial
#include <Streaming.h>          // For serial debugging output - https://www.arduino.cc/reference/en/libraries/streaming/

#define EN_PIN           7      // Enable
#define DIR_PIN          2      // Direction
#define STEP_PIN         3      // Step
#define SW_SCK           4      // Software Slave Clock (SCK)
#define SW_TX            6     // SoftwareSerial receive pin
#define SW_RX            5      // SoftwareSerial transmit pin
// #define EN_PIN           13      // Enable
// #define DIR_PIN          8     // Direction
// #define STEP_PIN         9      // Step
// #define SW_SCK           10      // Software Slave Clock (SCK)
// #define SW_TX            12     // SoftwareSerial receive pin
// #define SW_RX            11      // SoftwareSerial transmit pin
#define DRIVER_ADDRESS   0b00   // TMC2209 Driver address according to MS1 and MS2
#define R_SENSE 0.11f           // SilentStepStick series use 0.11 ...and so does my fysetc TMC2209 (?)

SoftwareSerial SoftSerial(SW_RX, SW_TX);                          // Be sure to connect RX to TX and TX to RX between both devices

TMC2209Stepper TMCdriver(&SoftSerial, R_SENSE, DRIVER_ADDRESS);   // Create TMC driver

int accel;
long maxSpeed;
int speedChangeDelay;
bool dir = false;

String inputString = "";      // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

//== Setup ===============================================================================

void setup() {

  Serial.begin(115200); Serial.println("Hello world!");               // initialize hardware serial for debugging
  SoftSerial.begin(115200);           // initialize software serial for UART motor control
  TMCdriver.beginSerial(115200);      // Initialize UART
  
  pinMode(EN_PIN, OUTPUT);           // Set pinmodes
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);         // Enable TMC2209 board  

  TMCdriver.begin();                                                                                                                                                                                                                                                                                                                            // UART: Init SW UART (if selected) with default 115200 baudrate
  TMCdriver.toff(5);                 // Enables driver in software
  TMCdriver.rms_current(1800);        // Set motor RMS current
  TMCdriver.microsteps(1);         // Set microsteps

  TMCdriver.en_spreadCycle(false);
  TMCdriver.pwm_autoscale(true);     // Needed for stealthChop
  
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
  
}

//== Loop =================================================================================

void loop() {
  // Serial.println("Spin!");
  
  accel = 10000;                                         // Speed increase/decrease amount
  maxSpeed = 50000;                                      // Maximum speed to be reached
  speedChangeDelay = 100;                                // Delay between speed changes
  
  for (long i = 0; i <= maxSpeed; i = i + accel){             // Speed up to maxSpeed
    TMCdriver.VACTUAL(i);                                     // Set motor speed
    Serial << TMCdriver.VACTUAL() << endl;
    delay(100);
  }
  
  for (long i = maxSpeed; i >=0; i = i - accel){              // Decrease speed to zero
    TMCdriver.VACTUAL(i);
    Serial << TMCdriver.VACTUAL() << endl;
    delay(100);
  }  
  
  // print the string when a newline arrives:
  if (stringComplete) {
    Serial.println(inputString);
    // clear the string:
    inputString = "";
    stringComplete = false;
    
    dir = !dir; // REVERSE DIRECTION
    TMCdriver.shaft(dir); // SET DIRECTION
  }
}

void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
