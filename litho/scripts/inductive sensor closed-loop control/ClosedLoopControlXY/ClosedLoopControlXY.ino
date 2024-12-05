#include <SoftwareSerial.h>
#include <Stepper.h>
#include <Arduino.h>
#include <BasicStepperDriver.h>
#include <MultiDriver.h>

#include "Seeed_LDC1612.h"

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

// #define MAX_STRINGS 10

SoftwareSerial SUART(0, 1); // SRX = 0, STX = 1

// String commandParts[MAX_STRINGS];

String inputLine = ""; 

//LDC 1612 object
LDC1612 sensor;
long channel0out, channel1out;

// Default to moving away
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
int directionYFlag = 1;
int directionXFlag = 1;

long parsedNumberY, parsedNumberX;
long minimumApproximation = 500;
long moveDeltaY, moveDeltaX = 0;

// Minimum and maximum movement amount in mm
const float MIN_MOVE = 0.02;
const float MAX_MOVE = 5.00;

bool leftX, leftY = true;

int selectedMicrosteps = 1;

int microstepsOptions[] = {1, 2, 4, 8};

void setup() {
  Serial.begin(9600);
  SUART.begin(9600);
  Serial.println();
  Serial.println();
  Serial.println();
  Serial.println();
  Serial.println();
  Serial.println("====================");
  Serial.println("Initializing Setup...");

  pinMode(EN, OUTPUT);
  digitalWrite(EN, HIGH);
  // Set up default microsteps
  selectedMicrosteps = 1;

  Serial.println("Select microstepping (options: 1, 2, 4, 8):");
  while (Serial.available() == 0) { 
    delay(10); 
  }
  if (Serial.available() > 0) {
    selectedMicrosteps = Serial.parseInt();
    bool validMicrostep = false;
    for (int i = 0; i < sizeof(microstepsOptions) / sizeof(microstepsOptions[0]); i++) {
      if (selectedMicrosteps == microstepsOptions[i]) {
        validMicrostep = true;
        break;
      }
    }
    if (!validMicrostep) {
      Serial.println("Invalid microstepping option. Defaulting to 1.");
      selectedMicrosteps = 1;
    }
  }

  stepperX.begin(60, selectedMicrosteps);
  stepperY.begin(60, selectedMicrosteps);

  sensor.init();
  if(sensor.LDC1612_mutiple_channel_config()) {
      Serial.println("can't detect LDC 1612 sensor!");
      while(1);
  }

  Serial.println("LDC 1612 sensor Set.");

  
  Serial.println("Setup complete.");
  Serial.println("====================");
  delay(500);
  // while (SUART.available() > 0) {
  //   char ch = SUART.read();
  //   if (ch == '\n') { 
  //     Serial.println(inputLine); 
  //     inputLine = ""; 
  //   } else {
  //     inputLine += ch;
  //   }
  // }
  // parsedNumberY = inputLine.toInt();
}

void readSensorMeas() {
  u32 result_channel1=0;
  u32 result_channel2=0;

  sensor.get_channel_result(CHANNEL_0,&result_channel1);
  sensor.get_channel_result(CHANNEL_1,&result_channel2);

  if(0!=result_channel1)
  {
      Serial.print("Channel 0 measurement: ");
      Serial.println(result_channel1);
  }
  if(0!=result_channel2)
  {
      Serial.print("Channel 1 measurement: ");
      Serial.println(result_channel2);
  }
  channel0out = result_channel1;
  channel1out = result_channel2;
}

// int splitString(String str) {
//   int startIndex = 0; 
//   int spaceIndex = 0;
//   int count = 0;  

//   while (spaceIndex >= 0 && count < MAX_STRINGS) {
//     spaceIndex = str.indexOf(' ', startIndex); // Find the next space
//     if (spaceIndex == -1) {  // If no more spaces are found, get the rest of the string
//       commandParts[count] = str.substring(startIndex);
//       break;
//     } else {
//       commandParts[count] = str.substring(startIndex, spaceIndex);
//       startIndex = spaceIndex + 1;  // Skip the space
//     }
//     count++;
//   }

//   return count;
// }

// Commands in the format of "DirY xx DirY xx"
// For instance, "L 1 U 1" means moving left and up for 1 mmm
// Maximum movement amount is 5mmm
// Minimum movement amount is 20um
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
  }

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); 
    command.trim(); 
    // Check if the command format is valid
    if (command.length() == 7 && (command[0] == 'L' || command[0] == 'R') && command[1] == ' ') {
      char directionX = command[0]; // Extract directionY (L or R)
      float amount = command.substring(2, 3).toFloat(); // Extract the amount and convert to float
      // Ensure the movement is within allowed range
      if (amount < MIN_MOVE) {
        amount = MIN_MOVE;
      } else if (amount > MAX_MOVE) {
        amount = MAX_MOVE;
      }

      // NOTE: THE LEFT/RIGHT DIRECTIONS ARE INVERTED FOR THE DEMO 12/5/2024
      // CHANGE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (directionX == 'R') {
        Serial.print("Prepare to move X axis Left for ");
        leftX = true;
      } else if (directionX == 'L') {
        Serial.print("Prepare to move X axis Right for ");
        leftX = false;
      }
      Serial.print(amount);
      Serial.println(" mm");
      moveDeltaX = amount / 10 * 271428.57;
    } else {
      Serial.println("Invalid command format. Use 'L <amount>' or 'R <amount>' for X axis.");
      return true;
    }

    if (command.length() == 7 && (command[4] == 'U' || command[4] == 'D') && command[5] == ' ') {
      char directionY = command[4]; // Extract directionY (L or R)
      float amount = command.substring(6).toFloat(); // Extract the amount and convert to float
      // Ensure the movement is within allowed range
      if (amount < MIN_MOVE) {
        amount = MIN_MOVE;
      } else if (amount > MAX_MOVE) {
        amount = MAX_MOVE;
      }

      // NOTE: THE UP/DOWN DIRECTIONS ARE INVERTED FOR THE DEMO 12/5/2024
      // CHANGE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (directionY == 'D') {
        Serial.print("Prepare to move Y axis Up for ");
        leftY = true;
      } else if (directionY == 'U') {
        Serial.print("Prepare to move Y axis Down for ");
        leftY = false;
      }
      Serial.print(amount);
      Serial.println(" mm");
      moveDeltaY = amount / 10 * 271428.57;
      return false;
    } else {
      Serial.println("Invalid command format. Use 'D <amount>' or 'U <amount>' for Y axis.");
      return true;
    }
  }
}

void loop() {
  delay(10);
  bool interrupted = readCommands();
  long startNumberY, targetNumberY;
  long startNumberX, targetNumberX;
  int directionY = leftY ? -1 : 1;
  int directionX = leftX ? 1 : -1;
  bool set = false;

  // Movement params
  if (!interrupted) {
    readSensorMeas();
    parsedNumberX = channel1out;
    inputLine = "";
    directionX = leftX ? 1 : -1;
    startNumberX = parsedNumberX;
    targetNumberX = parsedNumberX + directionX * moveDeltaX;

    Serial.println("Initializing stepper movement: ");
    Serial.print("Start Number: ");
    Serial.println(startNumberX);
    Serial.print("Target Number: ");
    Serial.println(targetNumberX);
  } 

  while (abs(parsedNumberX - targetNumberX) > minimumApproximation && !interrupted) {
    
    readSensorMeas();
    parsedNumberX = channel1out;

    int steps = 100;
    if (parsedNumberX < 41500000 || parsedNumberX > 43000000) {
      Serial.println(parsedNumberX);
      Serial.println("Maximum Movement Amount exceeded. Resetting Target Number...");
      if (parsedNumberX < 41500000) {
        targetNumberX = 41500000 + 1000;
      } else {
        targetNumberX = 43000000 - 1000;
      }
      Serial.print("Target Number Set to: ");
      Serial.println(targetNumberY);
      delay(100);
    }

    if (abs(parsedNumberX - targetNumberX) > 10000 && steps >= 100) {
      steps = 100;
    } else if (abs(parsedNumberX - targetNumberX) > 5000 && steps > 50) {
      steps = 30;
    } else if (abs(parsedNumberX - targetNumberX) > 1000) {
      steps = 10;
    } else if (abs(parsedNumberX - targetNumberX) < 1000) {
      steps = 5;
    }
    Serial.print("Now: ");
    Serial.println(parsedNumberX);
    Serial.print("Target: ");
    Serial.println(targetNumberX);
    Serial.print(parsedNumberX - targetNumberX);
    Serial.println(" Amount left");
    directionX = parsedNumberX > targetNumberX ? 1 : -1;
    Serial.print("directionY: ");
    Serial.println(directionX);
    Serial.println();
    digitalWrite(EN, LOW);
    stepperX.move(steps * directionX);
    digitalWrite(EN, HIGH);
    delay(50);
  }

  if (!interrupted) {
    Serial.println();
    Serial.println("<====== X Movement Complete. =====>");
    Serial.println("Results: ");
    Serial.print(startNumberY);
    Serial.print(" -> ");
    Serial.println(parsedNumberY);
    Serial.print("Interrupted: ");
    Serial.println(interrupted);
    Serial.println();
  }

  delay(1000);
  
  if (!interrupted) {
    // Movement params
    readSensorMeas();
    parsedNumberY = channel0out;
    inputLine = "";
    directionY = leftY ? -1 : 1;
    startNumberY = parsedNumberY;
    targetNumberY = parsedNumberY + directionY * moveDeltaY;
    
    Serial.println("Initializing stepper movement: ");
    Serial.print("Start Number: ");
    Serial.println(startNumberY);
    Serial.print("Target Number: ");
    Serial.println(targetNumberY);
  }

  while (abs(parsedNumberY - targetNumberY) > minimumApproximation && !interrupted) {
    
    readSensorMeas();
    parsedNumberY = channel0out;

    int steps = 100;
    if (parsedNumberY < 41900000 || parsedNumberY > 43000000) {
      Serial.println(parsedNumberY);
      Serial.println("Maximum Movement Amount exceeded. Resetting Target Number...");
      if (parsedNumberY < 41900000) {
        targetNumberY = 41900000 + 1000;
      } else {
        targetNumberY = 43000000 - 1000;
      }
      Serial.print("Target Number Set to: ");
      Serial.println(targetNumberY);
      delay(100);
    }

    if (abs(parsedNumberY - targetNumberY) > 10000 && steps >= 100) {
      steps = 100;
    } else if (abs(parsedNumberY - targetNumberY) > 5000 && steps > 50) {
      steps = 30;
    } else if (abs(parsedNumberY - targetNumberY) > 1000) {
      steps = 10;
    } else if (abs(parsedNumberY - targetNumberY) < 1000) {
      steps = 5;
    }
    Serial.print("Now: ");
    Serial.println(parsedNumberY);
    Serial.print("Target: ");
    Serial.println(targetNumberY);
    Serial.print(parsedNumberY - targetNumberY);
    Serial.println(" Amount left");
    directionY = parsedNumberY > targetNumberY ? -1 : 1;
    Serial.print("directionY: ");
    Serial.println(directionY);
    Serial.println();
    digitalWrite(EN, LOW);
    stepperY.move(steps * directionY);
    digitalWrite(EN, HIGH);
    delay(50);
  }

  if (!interrupted) {
    Serial.println();
    Serial.println("<====== Y Movement Complete. =====>");
    Serial.println("Results: ");
    Serial.print(startNumberY);
    Serial.print(" -> ");
    Serial.println(parsedNumberY);
    Serial.print("Interrupted: ");
    Serial.println(interrupted);
    Serial.println();
  }
  
}
