#include <SoftwareSerial.h>
#include <Stepper.h>
#include <Arduino.h>
#include <BasicStepperDriver.h>
#include <MultiDriver.h>

#include "Seeed_LDC1612.h"

#define PIN_EN 8
#define PIN_DIR_X 5  // Change these pin numbers to match CNC Shield
#define PIN_STEP_X 2
#define PIN_DIR_Y 6
#define PIN_STEP_Y 3
#define PIN_DIR_Z 7
#define PIN_STEP_Z 4
#define MICROSTEPSX 1
#define MICROSTEPSY 16
// motor https://blog.protoneer.co.nz/arduino-cnc-shield/ is defined as 200 steps/rev
// motor driver https://biqu.equipment/products/bigtreetech-tmc2209-stepper-motor-driver-for-3d-printer-board-vs-tmc2208?srsltid=AfmBOopezr9e2gOP7n7De8bN6u3wXzksJSTVsHnytXv9BuVOehQGE7xv
// does 1/8 microstepping by default
#define MOTOR_STEPS 200*8

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

BasicStepperDriver stepperX(MOTOR_STEPS, PIN_DIR_X, PIN_STEP_X);
BasicStepperDriver stepperY(MOTOR_STEPS, PIN_DIR_Y, PIN_STEP_Y);
BasicStepperDriver stepperZ(MOTOR_STEPS, PIN_DIR_Z, PIN_STEP_Z);

long minimumApproximation = 10;

// Minimum and maximum movement amount in um
const long MAX_MOVE_UM = 10000;

int selectedMicrosteps = 1;

int microstepsOptions[] = {1, 2, 4, 8};

bool LOG_SENSOR = false;


struct RawSensorMeas {
  int32_t x_hz;
  int32_t y_hz;
};

struct SensorMeas {
  int32_t x_um;
  int32_t y_um;
};


enum class CommandKind {
  RelativeStepsZ, // "z" integer_um
  RelativeSteps, // "s" integer_steps integer_steps
  Relative, // "r" integer_um integer_um
  Absolute, // "a" integer_um integer_um
  QueryPosition, // "q" -> "$ " integer_um "," integer_um
};


struct Command {
  CommandKind kind;

  // these are step counts for RawStep, um for Relative, and Hz for AbsoluteHz
  union {
    struct {
      long x;
      long y;
    } rel;

    struct {
      long x_um;
      long y_um;
    } abs;

    struct {
      long z;
    } rel_z;
  };
};

// works around bug in Arduino IDE, see https://forum.arduino.cc/t/x-does-not-name-a-type/687314/3
RawSensorMeas readRawSensorMeas();
SensorMeas readSensorMeas();
bool readCommands(Command &cmd);

void setup() {
  Serial.begin(115200);

  if (!LOG_SENSOR) {
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println();
    Serial.println("====================");
    Serial.println("Initializing Setup...");
  }

  pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_EN, HIGH);
 
  stepperX.begin(60, 1);
  stepperY.begin(60, 1);
  stepperZ.begin(60, 1);

  sensor.init();
  if(sensor.LDC1612_mutiple_channel_config()) {
      Serial.println("can't detect LDC 1612 sensor!");
      while(1);
  }

  // sets about ~10ms sampling time
  sensor.set_conversion_time(CHANNEL_0, 0x5460);
  sensor.set_conversion_time(CHANNEL_1, 0x5460);

  if (!LOG_SENSOR) {
    Serial.println("LDC 1612 sensor Set.");
    Serial.println("Setup complete.");
    Serial.println(sizeof(long));
    Serial.println("====================");
  }
  delay(500);
  Serial.print("foobar "); Serial.println(readRawSensorMeas().x_hz);
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


#define MIN_X_HZ 41400000l
#define MAX_X_HZ 41600000l

#define MIN_Y_HZ 41900000l
#define MAX_Y_HZ 43000000l

RawSensorMeas readRawSensorMeas() {
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
  } }

  return { result_channel2, result_channel1 };
}

SensorMeas readSensorMeas() {
  RawSensorMeas raw = readRawSensorMeas();

  long x_um = (MAX_X_HZ - raw.x_hz) / 40.0;

  long b = 43012077 - raw.y_hz;

  long y_steps;
  if (b <= 985002) {
    y_steps = 9.52 + 6.33e-3 * b + 1.2e-9 * b * b + 5.08e-15 * b * b * b;
  } else {
    y_steps = -220190.0 + 0.62 * b - 5.74e-7 * b * b + 1.84e-13 * b * b * b;
  }
  long y_um = y_steps / 3.2;

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
      if (command.length() == 0) {
        break;
      }

      char kind = command[0];
      if (kind == 'a') {
        if (sscanf(command.c_str(), "%c %ld %ld", &kind, &cmd.abs.x_um, &cmd.abs.y_um) != 3) {
          break;
        }

        cmd.kind = CommandKind::Absolute;
      } else if (kind == 'z') {
        if (sscanf(command.c_str(), "%c %ld", &kind, &cmd.rel_z.z) != 2) {
          break;
        }

        cmd.kind = CommandKind::RelativeStepsZ;
      } else if (kind == 'r' || kind == 's') {
        cmd.kind = (kind == 'r' ? CommandKind::Relative : CommandKind::RelativeSteps);

        if (sscanf(command.c_str(), "%c %ld %ld", &kind, &cmd.rel.x, &cmd.rel.y) != 3) {
          break;
        }
      } else if (kind == 'q') {
        cmd.kind = CommandKind::QueryPosition;
      }

      /*if (!LOG_SENSOR) {
        Serial.print("Moving "); Serial.print(cmd.x_um); Serial.print("um in X and "); Serial.print(cmd.y_um); Serial.println("um in Y");
      }*/
      return true;
    } while (0);

    if (!LOG_SENSOR) {
      Serial.print("Invalid command format `"); Serial.print(command); Serial.println("`");
    }
  }

  return false;
}

void report_position() {
  SensorMeas meas = readSensorMeas();
  Serial.print('$');
  Serial.print(meas.x_um);
  Serial.print(',');
  Serial.println(meas.y_um);
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

  int state = 2;
  x_active = true;
  y_active = true;
  target_x_um = 0;
  target_y_um = 0;

  while (true) {
    SensorMeas measured = readSensorMeas();

    if (state == 1) {
      Serial.println("Step,Y Sensor Measurement");
      // Take up backlash
      stepperY.move(800);
      stepperY.move(-800);

      for (int i = 0; i < 2800; ++i) {
        Serial.print(i * 10);
        Serial.print(",");
        //RawSensorMeas raw = readRawSensorMeas();
        SensorMeas meas = readSensorMeas();
        //Serial.print(raw.x_hz);
        //Serial.print(",");
        Serial.print(meas.y_um);
        Serial.print(",");
        Serial.println(meas.y_um - (long)(i * 10 / 3.2));
        stepperY.move(-10);
      }

      state = 2;
    }

    /*if (state == 1) {
      Serial.println("Step,X Sensor Measurement");
      // Take up backlash
      stepperY.move(800);
      stepperY.move(-800);

      for (int i = 0; i < 1920; ++i) {
        Serial.print(i * 10);
        Serial.print(",");
        RawSensorMeas raw = readRawSensorMeas();
        Serial.print(raw.x_hz);
        stepperY.move(-10);
      }

      state = 2;
    }*/

    if (readCommands(cmd)) {
      switch (cmd.kind) {
      case CommandKind::Absolute:
        target_x_um = labs(cmd.abs.x_um);
        target_y_um = labs(cmd.abs.y_um);

        x_active = true;
        y_active = true;

        x_steps = 0;
        y_steps = 0;
        break;
      case CommandKind::Relative:
        target_x_um = measured.x_um + cmd.rel.x;
        target_y_um = measured.y_um + cmd.rel.y;

        x_active = labs(cmd.rel.x) > 0;
        y_active = labs(cmd.rel.y) > 0;

        x_steps = 0;
        y_steps = 0;

        if (!LOG_SENSOR) {
          Serial.println("Starting relative movement (Hz): ");
          Serial.print("X: "); Serial.print(measured.x_um); if (cmd.rel.x > 0) { Serial.print(" +"); } else { Serial.print(" -"); } Serial.print(labs(cmd.rel.x)); Serial.print(" -> "); Serial.println(target_x_um); 
          Serial.print("Y: "); Serial.print(measured.y_um); if (cmd.rel.y > 0) { Serial.print(" +"); } else { Serial.print(" -"); } Serial.print(labs(cmd.rel.y)); Serial.print(" -> "); Serial.println(target_y_um);
        }
        break;
      case CommandKind::RelativeSteps:
        if (!LOG_SENSOR) {
          Serial.println("Doing relative steps!");
        }

        digitalWrite(PIN_EN, LOW);
        if (cmd.rel.x != 0) { stepperX.move(cmd.rel.x); }
        if (cmd.rel.y != 0) { stepperY.move(cmd.rel.y); }
        digitalWrite(PIN_EN, HIGH);
        report_position();
        break;
      case CommandKind::RelativeStepsZ:
        if (!LOG_SENSOR) {
          Serial.println("Doing relative Z steps!");
        }
        digitalWrite(PIN_EN, LOW);
        stepperZ.move(cmd.rel_z.z);
        digitalWrite(PIN_EN, HIGH);
        break;
      case CommandKind::QueryPosition:
        report_position();
        break;
      }
    }

    digitalWrite(PIN_EN, (x_active || y_active) ? LOW : HIGH);

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
          report_position();
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
        
        //Serial.print(y_steps); Serial.print(","); Serial.println(measured.y_um);
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
        if (state == 0) {
          state = 1;
        }
        if (!LOG_SENSOR) {
          Serial.println();
          Serial.println("<====== Y Movement Complete. =====>");
          Serial.println("Results: ");
          Serial.print("Traveled "); Serial.print(y_steps); Serial.print(" steps -> ");
          Serial.print(" -> ");
          Serial.println(measured.y_um);
          Serial.println();
          report_position();
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
