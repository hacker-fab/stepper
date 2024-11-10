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
long parsedNumber;
long minimumApproximation = 500;
long moveDelta = 0;

// Minimum and maximum movement amount in mm
const float MIN_MOVE = 0.02;
const float MAX_MOVE = 4.00;

bool left = true;

void setup() {
  Serial.begin(9600);
  SUART.begin(9600);

  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW);
  // Set up default microsteps
  stepperX.begin(60, 1);
  delay(500);
  Serial.println("Setup complete.");  

  // while (SUART.available() > 0) {
  //   char ch = SUART.read();
  //   if (ch == '\n') { 
  //     Serial.println(inputLine); 
  //     inputLine = ""; 
  //   } else {
  //     inputLine += ch;
  //   }
  // }
  // parsedNumber = inputLine.toInt();
}

bool readCommands() {
  bool firstFlag = true;
  char ch;
  String line = "";
  while (Serial.available() == 0) {
    if (firstFlag) {
      Serial.println();
      Serial.println("Waiting for commands...");
      firstFlag = false;
    }
    // Do this to flush UART
    // TODO: Find better method
    if (SUART.available() > 0) {
      ch = SUART.read();
      if (ch == '\n'){
        line = "";
      } else {
        line += ch; 
      }
    }

    delay(10);
  }

  while (ch != '\n') {
    ch = SUART.read();
    if (ch != '\n') {
      line += ch;
    }
  }
  
  if (parsedNumber == 0) {
    parsedNumber = line.toInt();
  }

  line = "";
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); 
    command.trim(); 

    // Check if the command format is valid
    if (command.length() >= 3 && (command[0] == 'L' || command[0] == 'R') && command[1] == ' ') {
      char direction = command[0]; // Extract direction (L or R)
      float amount = command.substring(2).toFloat(); // Extract the amount and convert to float

      // Ensure the movement is within the allowed range
      if (amount < MIN_MOVE) {
        amount = MIN_MOVE;
      } else if (amount > MAX_MOVE) {
        amount = MAX_MOVE;
      }

      // Print the action for debugging or verification
      if (direction == 'L') {
        Serial.print("Prepare to move left for ");
        left = true;
      } else if (direction == 'R') {
        Serial.print("Prepare to move right for ");
        left = false;
      }
      Serial.print(amount);
      Serial.println(" mm");

      moveDelta = amount / 10 * 271428.57;

      return false;
    } else {
      Serial.println("Invalid command format. Use 'L <amount>' or 'R <amount>'.");
      return true;
    }
  }
}

void loop() {
  while (!SUART.available()) {
  }

  delay(10);

  
  bool interrupted = readCommands();
  long startNumber, targetNumber;
  int direction = left ? 1 : -1;
  bool set = false;
  inputLine = "";
  while (abs(parsedNumber - targetNumber) > minimumApproximation && !interrupted) {
    while (SUART.available() > 0) {
      char ch = SUART.read();
      if (ch == '\n'){
        Serial.println(ch);
        if (inputLine.length() > 9) { // Check if the input exceeds 28 bits (8 digits max for 28 bits)
          Serial.print("Length: ");
          Serial.print(inputLine.length());
          Serial.println("Error: Input exceeds 28-bit limit. Please enter a valid 28-bit number. ");
          inputLine = "";
        } else {
          parsedNumber = inputLine.toInt();
          if (!set) {
            direction = left ? 1 : -1;
            startNumber = parsedNumber;
            targetNumber = parsedNumber + direction * moveDelta;
            Serial.print("Start Number: ");
            Serial.println(startNumber);
            Serial.print("Target Number: ");
            Serial.println(targetNumber);
            set = true;
          }
          inputLine = "";
          break;
        }
      } else {
        inputLine += ch; 
      }
    }

    int steps = 100;
    if (parsedNumber < 44000000 || parsedNumber > 45000000) {
      Serial.println(parsedNumber);
      Serial.println("Maximum Movement Amount exceeded. Stopping Motor.");
      stepperX.stop();
      interrupted = true;
      break;
    } else {
      if (abs(parsedNumber - targetNumber) > 10000 && steps >= 100) {
        steps = 100;
      } else if (abs(parsedNumber - targetNumber) > 5000 && steps > 50) {
        steps = 30;
      } else if (abs(parsedNumber - targetNumber) > 1000) {
        steps = 10;
      } else if (abs(parsedNumber - targetNumber) < 1000) {
        steps = 5;
      }
      Serial.print("Now: ");
      Serial.println(parsedNumber);
      Serial.print("Target: ");
      Serial.println(targetNumber);
      Serial.print(parsedNumber - targetNumber);
      Serial.println(" Amount Left");
      direction = parsedNumber > targetNumber ? 1 : -1;
      stepperX.move(steps * direction);
      delay(50);
    }
  }

  Serial.println();
  Serial.print(startNumber);
  Serial.print(" -> ");
  Serial.print(parsedNumber);
  Serial.println();
  Serial.println();
  Serial.println();
}