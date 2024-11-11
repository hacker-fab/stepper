#include <SoftwareSerial.h>
#include <Stepper.h>
#include <Arduino.h>
#include <BasicStepperDriver.h>
#include <MultiDriver.h>

SoftwareSerial mySerial(A0, A1);  // RX, TX
int L = 40;  // Adjust the number of rotations as needed
int state = 0;
int EN = 8;  // Change the stepper enable pin to match CNC Shield


#define DIR_X 5  // Change these pin numbers to match CNC Shield
#define STEP_X 2
#define DIR_Y 6
#define STEP_Y 3
#define MICROSTEPSX 1
#define MICROSTEPSY 16
#define MOTOR_STEPS 200*8

SoftwareSerial SUART(0, 1); // SRX = 0, STX = 1

String inputLine = "";

// Default to moving away
bool clockwise = true;
bool reset = true;

//44107040 -- 1.7cm
//4.9cm 44270000
//5.6cm 44080000
// y = −271428.57x+45600000.00
// 1cm = 271428 delta

// delta 185275 is approxmately 1 cm

BasicStepperDriver stepperX(MOTOR_STEPS, DIR_X, STEP_X);
BasicStepperDriver stepperY(MOTOR_STEPS, DIR_Y, STEP_Y);

// Default to moving away
int directionFlag = 1;

void setup() {
  Serial.begin(9600);
  SUART.begin(9600);

  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW);
  Serial.begin(9600);
  mySerial.begin(9600);
  // Set up default microsteps
  stepperX.begin(60, 1);
  delay(500);
  Serial.println("Setup complete.");  
  // From non-block example of the BasicDriver repo:
  // set the motor to move continuously for a reasonable time to hit the stopper
  // let's say 100 complete revolutions (arbitrary number)

  while (SUART.available() > 0) {
    char ch = SUART.read(); // Read each character
    if (ch == '\n') { // Check if the character is a newline
      Serial.println(inputLine); // Print the complete line
      inputLine = ""; // Clear the buffer for the next line
    } else {
      inputLine += ch; // Append character to buffer
    }
  }
  long parsedNumber = inputLine.toInt();
  setDirectionFlag(true);
}

// If reset is true, always reset
int setDirectionFlag(bool reset) {
  // Convert the string to a long integer
  long parsedNumber = inputLine.toInt();
  // Check if the parsed number is within the 28-bit unsigned integer range (0 to 268435455)
  if (parsedNumber < 0 || parsedNumber > 268435455) {
    Serial.println("Error: Number out of 28-bit range. Please enter a number between 0 and 268435455.");
  } else {
    // Print the parsed number if it's valid
    Serial.println(parsedNumber);
    if (parsedNumber >= 44265275 && !clockwise || reset) {
      clockwise = true;
      Serial.println("Stopped: too close");
      stepperX.stop();
      Serial.println("=== Reset Motor ===");
      directionFlag = 1;
    } else if (parsedNumber <= 44080000 && clockwise || reset){
      clockwise = false;
      Serial.println("Stopped: too far");
      stepperX.stop();
      Serial.println("=== Reset Motor ===");
      directionFlag = -1;
      //stepperX.move(-100);
      // Serial.print("Moving for another ");
      // Serial.print(wait_time_micros);
      // Serial.println("us ---- "
    }
  }

  return parsedNumber;
}

void loop() {
  while (SUART.available() > 0) {
    char ch = SUART.read();
    if (ch == '\n') {
      if (inputLine.length() > 9) { // Check if the input exceeds 28 bits (8 digits max for 28 bits)
        Serial.print("Length: ");
        Serial.print(inputLine.length());
        Serial.println("Error: Input exceeds 28-bit limit. Please enter a valid 28-bit number. ");
      } else {
          int parsedNumber = setDirectionFlag(false);
          if (parsedNumber < 44000000 || parsedNumber > 45000000) {
            stepperX.stop();
            delay(50);
          }else {
            stepperX.move(500 * directionFlag);
            Serial.print("Direction Flag: ");
            Serial.println(directionFlag);
            delay(50);
          }
      }
      inputLine = ""; 
    } else {
      inputLine += ch; 
    }
  }
}