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
// motor https://blog.protoneer.co.nz/arduino-cnc-shield/ is defined as 200 steps/rev
// motor driver https://biqu.equipment/products/bigtreetech-tmc2209-stepper-motor-driver-for-3d-printer-board-vs-tmc2208?srsltid=AfmBOopezr9e2gOP7n7De8bN6u3wXzksJSTVsHnytXv9BuVOehQGE7xv
// does 1/8 microstepping by default
#define MOTOR_STEPS 200*8

// #define MAX_STRINGS 10

SoftwareSerial SUART(0, 1); // SRX = 0, STX = 1

// String commandParts[MAX_STRINGS]; 

//LDC 1612 object
LDC1612 sensor;

// Default to moving away
bool reset = true;

//44107040 -- 1.7cm
//4.9cm 44270000
//5.6cm 44080000
// y = −271428.57x+45600000.00
// 1cm = 271428 delta

long um_to_hz(long um) {
  //return um * 130.482;
  return um * 112.256;
  //return um * 27.1428;
}

// delta 185275 is approxmately 1 cm

BasicStepperDriver stepperX(MOTOR_STEPS, DIR_X, STEP_X);
BasicStepperDriver stepperY(MOTOR_STEPS, DIR_Y, STEP_Y);

long minimumApproximation = 10;

// Minimum and maximum movement amount in um
const long MAX_MOVE_UM = 10000;

int selectedMicrosteps = 1;

int microstepsOptions[] = {1, 2, 4, 8};

bool LOG_SENSOR = false;

void setup() {
  Serial.begin(9600);
  SUART.begin(9600);

  if (!LOG_SENSOR) {
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println("====================");
    Serial.println("Initializing Setup...");
  }

  pinMode(EN, OUTPUT);
  digitalWrite(EN, HIGH);
  // Set up default microsteps
  selectedMicrosteps = 1;

  /*Serial.println("Select microstepping (options: 1, 2, 4, 8):");
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
  */
  selectedMicrosteps = 1;

  stepperX.begin(60, selectedMicrosteps);
  stepperY.begin(60, selectedMicrosteps);

  sensor.init();
  if(sensor.LDC1612_mutiple_channel_config()) {
      Serial.println("can't detect LDC 1612 sensor!");
      while(1);
  }

  if (!LOG_SENSOR) {
    Serial.println("LDC 1612 sensor Set.");
    Serial.println("Setup complete.");
    Serial.println(sizeof(long));
    Serial.println("====================");
  }
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

struct SensorMeas {
  int32_t x_um;
  int32_t y_um;
};

enum class CommandKind {
  RawStep,
  Relative,
  AbsoluteHz
};

struct Command {
  //CommandKind kind;

  bool relative;
  // these are step counts for RawStep, um for Relative, and Hz for AbsoluteHz
  long x_um;
  long y_um;
};


// works around bug in Arduino IDE, see https://forum.arduino.cc/t/x-does-not-name-a-type/687314/3
SensorMeas readSensorMeas();
bool readCommands(Command &cmd);

#define MIN_X_HZ 41400000l
#define MAX_X_HZ 41600000l

#define MIN_Y_HZ 41900000l
#define MAX_Y_HZ 43000000l

SensorMeas readSensorMeas() {
  int32_t result_channel1=0;
  int32_t result_channel2=0;

  sensor.get_channel_result(CHANNEL_0,&result_channel1);
  sensor.get_channel_result(CHANNEL_1,&result_channel2);

  if (result_channel1 == 0) { while (1) {
    Serial.println("Error while measuring channel0");
    delay(500);
  } }
  
  if (result_channel2 == 0) { while (1) {
    
    Serial.println("Error while measuring channel1");
    delay(500);
  } } /*else { Serial.println(result_channel2); }*/

  long x_um = (MAX_X_HZ - result_channel2) / 40.0;
  long y_um = (MAX_Y_HZ - result_channel1) / 112.256;

  return { x_um, y_um };
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

long clamp(long val, long lo, long hi) {
  if (val < lo) {
    return lo;
  } else if (val > hi) {
    return hi;
  } else {
    return val;
  }
}


// Commands in the format of "DirY xx DirY xx"
// For instance, "L 1 U 1" means moving left and up for 1 mmm
// Maximum movement amount is 5mmm
// Minimum movement amount is 20um
bool readCommands(Command &cmd) {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); 
    command.trim();

    do {
      char cmd_kind[9];
      char dir_x, dir_y;
      long amount_x, amount_y;
      if (sscanf(command.c_str(), "%c %ld %c %ld", &dir_x, &amount_x, &dir_y, &amount_y) != 4) {
        break;
      }

      if (dir_x != 'L' && dir_x != 'R') {
        Serial.print("Expected L or R for dir_x but found `"); Serial.print(dir_x); Serial.println("`");
        break;
      }

      if (dir_y != 'U' && dir_y != 'D') {
        Serial.print("Expected U or D for dir_y but found `"); Serial.print(dir_y); Serial.println("`");
        break;
      }

      // NOTE: THE LEFT/RIGHT AND UP/DOWN DIRECTIONS ARE INVERTED FOR THE DEMO 12/5/2024
      cmd.relative = true;
      cmd.relative = false;
      cmd.x_um = ((dir_x == 'R') ? 1 : -1) * clamp(amount_x, 0, MAX_MOVE_UM);
      cmd.y_um = ((dir_y == 'U') ? 1 : -1) * clamp(amount_y, 0, MAX_MOVE_UM);

      if (!LOG_SENSOR) {
        Serial.print("Moving "); Serial.print(cmd.x_um); Serial.print("um in X and "); Serial.print(cmd.y_um); Serial.println("um in Y");
      }
      return true;
    } while (0);

    //Serial.print("Invalid command format `"); Serial.print(command); Serial.println("`");
  }

  return false;
}

void loop() {
  SensorMeas initial_measured = readSensorMeas();
  Command cmd;
  
  long target_x_um;
  long target_y_um;

  bool x_active = false;
  bool y_active = false;

  // 43104109
  // (42698149 -> 41915254) / 6mm    130482.5 units per mm

  // 130.482 units per micron

  /*while (true) {
    int c = Serial.read();
    if (c == 'u') {
      Serial.println("going up");
      long before_hz = readSensorMeas().y_hz;
      digitalWrite(EN, LOW);
      stepperY.move(-200*8*2);
      digitalWrite(EN, HIGH);
      long after_hz = readSensorMeas().y_hz;
      Serial.print(before_hz); Serial.print(" -> "); Serial.println(after_hz);
    } else if (c == 'd') {
      Serial.println("going down");
      long before_hz = readSensorMeas().y_hz;
      digitalWrite(EN, LOW);
      stepperY.move(200*8*2);
      digitalWrite(EN, HIGH);
      long after_hz = readSensorMeas().y_hz;
      Serial.print(before_hz); Serial.print(" -> "); Serial.println(after_hz);
    }
  }*/

  long x_steps = 0;
  long y_steps = 0;

  while (true) {
    SensorMeas measured = readSensorMeas();

    if (readCommands(cmd)) {
      if (cmd.relative) {
        target_x_um = measured.x_um + cmd.x_um;
        target_y_um = measured.y_um + cmd.y_um;

        x_active = labs(cmd.x_um) > 0;
        y_active = labs(cmd.y_um) > 0;

        x_steps = 0;
        y_steps = 0;

        if (!LOG_SENSOR) {
          Serial.println("Starting relative movement (Hz): ");
          Serial.print("X: "); Serial.print(measured.x_um); if (cmd.x_um > 0) { Serial.print(" +"); } else { Serial.print(" -"); } Serial.print(labs(cmd.x_um)); Serial.print(" -> "); Serial.println(target_x_um); 
          Serial.print("Y: "); Serial.print(measured.y_um); if (cmd.y_um > 0) { Serial.print(" +"); } else { Serial.print(" -"); } Serial.print(labs(cmd.y_um)); Serial.print(" -> "); Serial.println(target_y_um);
        }
      } else {
        Serial.println("Absolute!");

        target_x_um = labs(cmd.x_um);
        target_y_um = labs(cmd.y_um);

        x_active = true;
        y_active = true;

        x_steps = 0;
        y_steps = 0;
      }
    }

    digitalWrite(EN, (x_active || y_active) ? LOW : HIGH);

    if (x_active) {
      long error_um = target_x_um - measured.x_um;
      if (labs(error_um) > minimumApproximation) {
        /*if (measured.x_um < MIN_X_HZ || measured.x_hz > MAX_X_HZ) {
          target_x_hz = clamp(measured.x_hz, MIN_X_HZ + 1000, MAX_X_HZ - 1000);

          if (!LOG_SENSOR) {
            Serial.println(measured.x_hz);
            Serial.println("Maximum X Movement Amount exceeded. Resetting Target Number...");
            Serial.print("Target Number Set to: ");
            Serial.println(target_x_hz);
          }
          delay(100);
        }*/

        int steps;
        if (abs(error_um) > 100) {
          steps = 100;
        } else if (abs(error_um) > 50) {
         steps = 30;
        } else if (abs(error_um) > 10) {
          steps = 10;
        } else {
          steps = 5;
        }

        x_steps += steps;
        steps *= (error_um > 0) ? 1 : -1;
        if (!LOG_SENSOR) {
          Serial.print("X Error: "); Serial.print(error_um); Serial.print(", Raw Steps: "); Serial.println(steps);
          Serial.println();
        }
        stepperX.move(steps);
      } else {
        x_active = false;
        if (!LOG_SENSOR) {
          Serial.println();
          Serial.println("<====== X Movement Complete. =====>");
          Serial.println("Results: ");
          Serial.print("Traveled "); Serial.print(x_steps); Serial.print(" steps -> ");
          Serial.println(measured.x_um);
          Serial.println();
        }
      }
    }

    if (y_active) {
      long error_um = target_y_um - measured.y_um;
      if (labs(error_um) > minimumApproximation) {
        /*if (measured.y_hz < MIN_Y_HZ || measured.y_hz > MAX_Y_HZ) {
          target_y_hz = clamp(measured.y_um, MIN_Y_HZ + 1000, MAX_Y_HZ - 1000);
          if (!LOG_SENSOR) {
            Serial.println(measured.y_um);
            Serial.println("Maximum Y Movement Amount exceeded. Resetting Target Number...");
            Serial.print("Target Number Set to: ");
            Serial.println(target_y_hz);
          }
          delay(100);
        }*/
        
        Serial.print(y_steps); Serial.print(","); Serial.println(measured.y_um);

        /*int steps;
        if (abs(error_hz) > 10000) {
          steps = 100;
        } else if (abs(error_hz) > 5000) {
         steps = 30;
        } else if (abs(error_hz) > 1000) {
          steps = 10;
        } else {
          steps = 5;
        }*/
        int steps;
        if (abs(error_um) > 100) {
          steps = 100;
        } else if (abs(error_um) > 50) {
         steps = 30;
        } else if (abs(error_um) > 10) {
          steps = 10;
        } else {
          steps = 5;
        }

        y_steps += steps;
        steps *= (error_um > 0) ? -1 : 1;
        if (!LOG_SENSOR) {
          Serial.print("Y Error: "); Serial.print(error_um); Serial.print(", Raw Steps: "); Serial.println(steps);
          Serial.println();
        }
        stepperY.move(steps);
      } else {
        y_active = false;
        if (!LOG_SENSOR) {
          Serial.println();
          Serial.println("<====== Y Movement Complete. =====>");
          Serial.println("Results: ");
          Serial.print("Traveled "); Serial.print(y_steps); Serial.print(" steps -> ");
          Serial.print(" -> ");
          Serial.println(measured.y_um);
          Serial.println();
        }
      }
    }

    /*
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

    }

    delay(1000);
    */

    /*while (abs(parsedNumberY - targetNumberY) > minimumApproximation && !interrupted) {
      readSensorMeas();
      parsedNumberY = channel0out;


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
    }*/
  }
}
