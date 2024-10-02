#include <Arduino.h>
#include <BasicStepperDriver.h>
#include <SoftwareSerial.h>

#define DIR_X 5
#define STEP_X 2
#define EN 8

// Steps/rev for 28HD1411
#define MOTOR_STEPS 1600

// Motor X on CNC
BasicStepperDriver stepperX(MOTOR_STEPS, DIR_X, STEP_X);

// RX, TX
SoftwareSerial mySerial(A0, A1);

int microsteps = 1;
// 'S' for steps, 'D' for degrees
char moveType;
float moveAmount;

void promptUserInput();

void setup() {
  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW); 
  Serial.begin(9600);
  mySerial.begin(9600);
  Serial.println("\n\n\n\n**************************************************************");
  Serial.println("Setup complete. Use serial input to control the stepper motor.");
  Serial.println("Default Settng: 30 rpm | 1 microsteps");
  stepperX.begin(30, microsteps); 
  promptForUserInput(); 
}

void loop() {
  if (Serial.available() > 0) {
    String userInput = Serial.readStringUntil('\n'); 
    userInput.trim();

    // Get the move type ('S' for steps, 'D' for degrees)
    if (userInput.startsWith("M")) {
      microsteps = userInput.substring(1).toInt(); 
      stepperX.begin(30, microsteps); 
      Serial.print("Microsteps set to: ");
      Serial.println(microsteps);
      delay(500);
      promptForUserInput();
    }
    else if (userInput.startsWith("C")) {
      moveType = userInput.charAt(1); 
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.print("Move type set to: ");
      Serial.println((moveType == 'S') ? "Steps" : "Degrees");
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      delay(500);
      if (moveType == 'S') {
        Serial.println("+++++++++++++++++++++++");
        Serial.println("Current move type: Step");
        Serial.println("Enter move amount (e.g., 1600 steps with microstep = 2 <==> 180 degree)\n");
      } else if (moveType == 'D') {
        Serial.println("+++++++++++++++++++++++");
        Serial.println("Current move type: Degree");
        Serial.println("Enter move amount (e.g., 1600 steps with microstep = 2 <==> 180 degree)\n");
      }
    }
    else {
      moveAmount = userInput.toFloat();  
      if (moveType == 'S') {
        Serial.println("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        Serial.print("Moving ");
        Serial.print(moveAmount);
        Serial.println(" steps...");
        Serial.println("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
        stepperX.move((int)moveAmount); 
      } else if (moveType == 'D') {
        Serial.println("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        Serial.print("Moving ");
        Serial.print(moveAmount);
        Serial.println(" degrees...");

        stepperX.rotate(moveAmount); 
        Serial.println("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
      }
      delay(500);
      promptForUserInput();
    }
  }
}

void promptForUserInput() {
  Serial.println("\n-------------------------");
  Serial.println("Command List:");
  Serial.println("M<number> - Set microsteps (e.g., M1 for full step, M16 for 1/16th step)");
  Serial.println("C<S or D> - Change move type: 'S' for steps, 'D' for degrees (e.g., CS for steps, CD for degrees)");
  Serial.println("-------------------------");
  if (moveType == 'S') {
    Serial.println("+++++++++++++++++++++++");
    Serial.println("Current move type: Step");
    Serial.println("Enter move amount (e.g., 1600 steps with microstep = 2 <==> 180 degree)\n");
  } else if (moveType == 'D') {
    Serial.println("+++++++++++++++++++++++");
    Serial.println("Current move type: Degree");
    Serial.println("Enter move amount (e.g., 1600 steps with microstep = 2 <==> 180 degree)\n");
  }
}