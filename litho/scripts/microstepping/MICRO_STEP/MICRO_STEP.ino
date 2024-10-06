// Credit for basis of the code: https://forum.arduino.cc/t/cnc-shield-v3-nextion/1188821/18
// Library used: https://github.com/laurb9/StepperDriver
#include <Stepper.h>
#include <Arduino.h>
#include <BasicStepperDriver.h>
#include <MultiDriver.h>
#include <SoftwareSerial.h>

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

// Right now only motor X is used
BasicStepperDriver stepperX(MOTOR_STEPS, DIR_X, STEP_X);
BasicStepperDriver stepperY(MOTOR_STEPS, DIR_Y, STEP_Y);

void setup() {
  // Set up control pins
  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW);
  Serial.begin(9600);
  mySerial.begin(9600);
  // Set up default microsteps
  stepperX.begin(60, 1);
  delay(500);
  Serial.println("Setup complete.");  
}

void loop() {
  // Move microsteps
  stepperX.rotate(1.8); // This moves the motor 1.8 degree clockwise
  delay(2000);
  stepperX.move(-1600); // This moves the motor 360 degree counter clockwise
  delay(2000);
}