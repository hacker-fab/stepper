/*
 * Sangaboard firmware
 * 
 * Most of the code is reused from the old openflexure_nano_motorcontroller firmware
 * 
 * This firmware was written by
 * Richard Bowman
 * Julian Stirling
 * Boyko Vodenicharski
 * Filip Ayazi
 * 
 * Much of the code is based on older code written by
 * James Sharkey and Fergus Riche
 * 
 * Released under GPL v3, 2017
 */
#include "Seeed_LDC1612.h" //homemade inductive sensor library
#include "StepperF_alt.h"   //Fergus's hacked stepper library
#include <assert.h>
#include <EEPROM.h>
#include <limits.h>

// LIGHT SENSOR SUPPORT
// Uncomment (exactly) one of the lines below to enable support for that sensor.
//#define ADAFRUIT_TSL2591
//#define ADAFRUIT_ADS1115

#ifdef ADAFRUIT_TSL2591
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#define LIGHT_SENSOR
#endif
#ifdef ADAFRUIT_ADS1115
#include <Wire.h>
#include <Adafruit_ADS1015.h>
#define LIGHT_SENSOR
#endif

//debug serial prints
#define DEBUG_ON

#ifdef DEBUG_ON
  #define D(x)  Serial.println (x)
#else
  #define D(x)
#endif


/*
 ************
 * ENDSTOPS *
 ************
*/
//#define ENDSTOPS_MIN
//#define ENDSTOPS_MAX

#if defined(ENDSTOPS_MIN) || defined(ENDSTOPS_MAX)
  //endstops are closed when the pin is down  by default, this inverts the behaviour
  #define ENDSTOPS_INVERT false

  #define ENDSTOPS_PULLUPS true

  //soft endstops trigger when 0/MAX is reached in the direction without endstops
  #define ENDSTOPS_SOFT

  //endstops wiring and positions
  const int endstops_min[]={A0,A1,A2};
  const int endstops_max[]={A3,A4,A5};
  long axis_max[3];
#endif

#ifdef ARDUINO_AVR_LEONARDO
  #define SANGABOARDv3
  #define BOARD_STRING "Sangaboard v0.3"
#elif ARDUINO_AVR_SANGABOARD
  #define SANGABOARDv3
  #define BOARD_STRING "Sangaboard v0.3"
#else
  #define SANGABOARDv2
  #define BOARD_STRING "Sangaboard v0.2"
#endif

#define EACH_MOTOR for(int i=0; i<n_motors; i++)
#define VER_STRING "Sangaboard Firmware v0.5.1"

// The array below has 3 stepper objects, for X,Y,Z respectively
const int n_motors = 3;
long min_step_delay;
const int min_step_delay_eeprom = sizeof(long)*n_motors;
long ramp_time;
const int ramp_time_eeprom = sizeof(long)*(n_motors+1);
const int axis_max_eeprom = sizeof(long)*(n_motors+2);
Stepper* motors[n_motors];
signed long current_pos[n_motors];
long steps_remaining[n_motors];

long duty_cycle[n_motors];
long stepper_cycle;

bool test_mode=false;

// We'll use this input buffer for serial comms
const int INPUT_BUFFER_LENGTH = 64;
char input_buffer[INPUT_BUFFER_LENGTH];

int command_prefix(String command, const char ** prefixes, int n_prefixes){
  // Check if the command starts with any of the prefixes in the list.
  // returns the index if so, otherwise returns -1
  for(int i=0; i<n_prefixes; i++){
    if(command.startsWith(prefixes[i])){
      return i;
    }
  }
  return -1;
}

//initialize inductive sensor:
// connect pin 2 (white wire) on inductive sensor to pin A4 (SDA) on Arduino
// connect pin 1 (yellow wire) on inductive sensor to pin A5 (SCL) on Arduino
//IndSensor sensor1;
LDC1612 sensor1;
void setup() {
  
  // initialise serial port
  Serial.begin(115200);
  while (! Serial )
    delay(1000);
  //setup inductive sensor
  sensor1.init();
  if (sensor1.LDC1612_mutiple_channel_config()) {
        Serial.println("can't detect sensor1!");
        while (1);
    }
  Serial.println("end of the setup");
  EACH_MOTOR{
    duty_cycle[i] = 10000000000;
    stepper_cycle = 0;
  }
  
  // get the stepoper objects from the motor shield objects
  #if defined(SANGABOARDv2)
    motors[0] = new Stepper(8, 13, 12, 11, 10);
    motors[1] = new Stepper(8, 9, 8, 7, 6);
    motors[2] = new Stepper(8, 5, 4, 3, 2);
  #elif defined(SANGABOARDv3)
    motors[0] = new Stepper(8, 8, 9, 10, 11);
    motors[1] = new Stepper(8, 5, 13, 4, 12);
    motors[2] = new Stepper(8, 6, 7, A5, A4);
  #endif
  EACH_MOTOR{
    motors[i]->setSpeed(10); // as a default set to 10 rpm, though this is ignored...
    steps_remaining[i]=0;
    EEPROM.get(sizeof(long)*i, current_pos[i]); //read last saved position from EEPROM
    //current_pos[i] = 0; //alternatively, reset on power cycle!
  }

  EEPROM.get(min_step_delay_eeprom, min_step_delay);
  if(min_step_delay < 0){ // -1 seems to be what we get if it's uninitialised.
    min_step_delay = 1000;
    EEPROM.put(min_step_delay_eeprom, min_step_delay);
  }
  EEPROM.get(ramp_time_eeprom, ramp_time);
  if(ramp_time < 0){ // -1 seems to be what we get if it's uninitialised.
    ramp_time = 0;
    EEPROM.put(ramp_time_eeprom, ramp_time);
  }
  #ifdef LIGHT_SENSOR
  setup_light_sensor();
  #endif /* LIGHT_SENSOR */

  #if defined(ENDSTOPS_MIN) || defined(ENDSTOPS_MAX)
    EACH_MOTOR{
      EEPROM.get(axis_max_eeprom+i*sizeof(long), axis_max[i]);
    }
  #endif

  #ifdef ENDSTOPS_MIN
    EACH_MOTOR{
      if(ENDSTOPS_PULLUPS)
        pinMode(endstops_min[i], INPUT_PULLUP);
      else
        pinMode(endstops_min[i], INPUT);
    }
  #endif

  #ifdef ENDSTOPS_MAX
    EACH_MOTOR{
      if(ENDSTOPS_PULLUPS)
        pinMode(endstops_max[i], INPUT_PULLUP);
      else
        pinMode(endstops_max[i], INPUT);
    }
  #endif

  Serial.println(F(VER_STRING));

}


void stepMotor(int motor, long dx){
  //make a single step of a single motor.
  current_pos[motor] += dx;
  motors[motor]->stepMotor(((current_pos[motor] % 8) + 8) % 8); //forgive the double-modulo; I need 0-7 even for -ve numbers
}

void releaseMotor(int motor){
  //release the stepper motor (de-enegrise the coils)
  motors[motor]->stepMotor(8);
}

void print_position(){
  EACH_MOTOR{
    if(i>0) Serial.print(" ");
    Serial.print(current_pos[i]);
  }
  Serial.println();
}

#if defined(ENDSTOPS_MIN) || defined(ENDSTOPS_MAX)

/**
* Check if endstop is triggered, direction {-1,+1} for {min,max} endstop
*/
boolean endstopTriggered(int axis, int direction){
#if defined(ENDSTOPS_MIN)
    if(direction<0){
      int value=analogRead(endstops_min[axis]);
      if(ENDSTOPS_INVERT!=(value<100))
        return true;
    }
#elif defined(ENDSTOPS_SOFT)
    if(direction<0 && current_pos[axis]<1)
        return true;
#endif

#if defined(ENDSTOPS_MAX)
  if(direction>0){
      int value=analogRead(endstops_max[axis]);
      if(ENDSTOPS_INVERT!=(value<100))
        return true;
  }
#elif defined(ENDSTOPS_SOFT)
  if(direction>0 && current_pos[axis]>=axis_max[axis])
    return true;
#endif
  return false;
}

void print_endstops_status(){
   EACH_MOTOR{
    if(i>0) Serial.print(" ");
      if(endstopTriggered(i,-1))
        Serial.print("-1");
      else if(endstopTriggered(i,1))
        Serial.print("1");
      else
    Serial.print("0");
  }
  Serial.println();
}

void print_axes_max(){
  EACH_MOTOR{
    if(i>0) Serial.print(" ");
    Serial.print(axis_max[i]);
  }
  Serial.println();
}

void home_min(){
    byte axes=7; //00000111
    home_min(axes);
}

void home_max(){
  byte axes=7;
  home_max(axes);
}

void home_min(byte axes){
  #ifdef ENDSTOPS_MIN
    int hit=0;
    long shift_min[3];
    long displ[3];

    EACH_MOTOR{
      if(axes&(1<<i)){
        shift_min[i]=3500;
        displ[i]=-100000;
      }else{
        shift_min[i]=0;
        displ[i]=0;
      }
    }

    EACH_MOTOR{
      hit=move_axes(displ);
      displ[-hit-1]=0;
    }

    //if in test mode, print positions when the endstops were hit the first time
    if(test_mode){
      Serial.print("First hits: ");
      print_position();
    }

    move_axes(shift_min);
    //now we should be <shift> steps away in chosen axes, first shift a bit closer
    //as the close and open positions some distance apart, then go
    //slowly back for maximum accuracy
    long shift_back[]={-shift_min[0]/2,-shift_min[1]/2,-shift_min[2]/2};
    move_axes(shift_back);
    delay(200);
    EACH_MOTOR{
      if(axes&(1<<i)){
        while(!endstopTriggered(i,-1)){
          stepMotor(i,-1);
          delayMicroseconds(4000);
        }
        if(test_mode){
          Serial.print("Homed endstop ");
          Serial.print(i);
          Serial.print(":");
          Serial.print(current_pos[i]);
          Serial.println();
        }
        current_pos[i]=0;
      }
    }
    EEPROM.put(0, current_pos);
    //move a little higher so the endstops are released
    if(!test_mode)
      move_axes(shift_min);
  #endif
  Serial.println("done.");
}

// UNTESTED
void home_max(byte axes){
  #ifdef ENDSTOPS_MAX
    //same as above
    int hit=0;
    long shift_max[3];
    long displ[3];
    EACH_MOTOR{
      if(axes&(1<<i)){
        shift_max[i]=-2000;
        displ[i]=150000;
      }else{
        shift_max[i]=0;
        displ[i]=0;
      }
    }

    EACH_MOTOR{
      hit=move_axes(displ);
      displ[hit-1]=0;
    }

    if(test_mode){
      Serial.print("First hits: ");
      print_position();
    }

    move_axes(shift_max);
    long shift_forw[]= {shift_max[0]/2,shift_max[1]/2,shift_max[2]/2};
    move_axes(shift_forw);

    EACH_MOTOR{
      if(axes&(1<<i)){
        while(!endstopTriggered(i,1)){
          stepMotor(i,1);
          delay(150);
        }
        if(test_mode){
          Serial.print("Homed endstop ");
          Serial.print(i);
          Serial.print(":");
          Serial.print(current_pos[i]);
          Serial.println();
        }
        axis_max[i]=current_pos[i];
      }
    }
    EEPROM.put(0, current_pos);
    EEPROM.put(axis_max_eeprom, axis_max);
  #endif
  Serial.println("done.");
}


#endif //ENDSTOPS


int move_axes(long displ[n_motors]){
   long displacement[n_motors];
  // move all the axes in a nice move
  // split displacements into magnitude and direction, and find max. travel
    int ret=0;
    long max_steps = 0;
    long dir[n_motors];
    EACH_MOTOR{
      dir[i] = displ[i] > 0 ? 1 : -1;
      displacement [i] = displ[i]*dir[i];
      if(displacement[i]>max_steps) max_steps=displacement[i];
    }
    // scale the step delays so the move goes in a straight line, with >=1 motor
    // running at max. speed.
    float step_delay[n_motors];
    EACH_MOTOR if(displacement[i]>0){
      step_delay[i] = float(max_steps)/float(displacement[i])*float(min_step_delay);
    }else{
      step_delay[i] = 9999999999;
    }
    // actually make the move
    bool finished = false;
    long distance_moved[n_motors];
    EACH_MOTOR distance_moved[i] = 0;
    unsigned long start = micros();
    float final_scaled_t = (float) max_steps * min_step_delay; //NB total time taken will be final_scaled_t + 2*ramp_time
    while(!finished){
      #if defined(ENDSTOPS_MIN) || defined(ENDSTOPS_MAX)
        int endstop_break=0;
        EACH_MOTOR{
          if(endstopTriggered(i,dir[i]*(displacement[i]-distance_moved[i])))
            endstop_break=dir[i]*(i+1);
        }
        if(endstop_break!=0){
          //Serial.print("Endstop hit:");
          //Serial.println(endstop_break);
          //if we have both min/max endstops, axis_max is adjusted to the correct value
          //if we only have min, we go from 0 -> predefined axis_max
          //if we only have max, we go from 0 -> predefined axis_max
          //the predefined axis_max are the travel distances in steps
          if(!test_mode && endstop_break<0) //min is always 0
            current_pos[-endstop_break-1]=0;
          else if(!test_mode) //max
            #if defined(ENDSTOPS_MIN) && defined(ENDSTOPS_MAX)
              axis_max[endstop_break-1]=current_pos[endstop_break-1];
            #elif defined(ENDSTOP_MAX) //we do not do this for ENDSTOPS_SOFT
              current_pos[endstop_break-1]=axis_max[endstop_break-1];
            #else
              ;
            #endif
          ret=endstop_break;
          break;
        }
      #endif

      unsigned long now=micros();
      float elapsed_t;
      if(now<start) //overflow in micros() after ~70 min
        elapsed_t=(float) (now+(ULONG_MAX-start));
      else
        elapsed_t = (float) (now-start);

      float scaled_t; //scale time to allow for acceleration
      if(ramp_time > 0){
        // if a ramp time is specified, accelerate at a constant acceleration for the
        // ramp time, then move at constant (maximum) speed, then decelerate.  If the
        // move is shorter than 2*ramp_time, accelerate then decelerate.
        float remaining_t = final_scaled_t + ramp_time - elapsed_t;
        if(elapsed_t < ramp_time && remaining_t > elapsed_t){ //for the first ramp_time, gradually accelerate
          scaled_t = elapsed_t*elapsed_t/(2*ramp_time);
        }else if(remaining_t < ramp_time){
          scaled_t = final_scaled_t - remaining_t*remaining_t/(2*ramp_time);
        }else{
          scaled_t = elapsed_t - ramp_time/2;
        }
      }else{
        scaled_t = elapsed_t;
      }
      finished = true;
      EACH_MOTOR{
        if(distance_moved[i] < displacement[i]){
          finished = false; //only if all axes are done are we truly finished.

          // check if it's time to take another step and move if needed.
          if(scaled_t > ((float)distance_moved[i] + 0.5) * step_delay[i]){
            stepMotor(i, dir[i]);
            distance_moved[i]++;
          }
        }
      }
      //delayMicroseconds(2000);
    }
    EEPROM.put(0, current_pos);
    return ret;
}

#ifdef ADAFRUIT_TSL2591
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591); // pass in a number for the sensor identifier (for your use later)

void setup_light_sensor(){
  if (tsl.begin())
  {
    tsl.setGain(TSL2591_GAIN_MED);
    tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);  // shortest integration time (bright light)
  }
  else
  {
    Serial.println(F("No light sensor found.  NB your board will start up faster if you recompile without light sensor support."));
  }
}

void print_light_sensor_gain(){
  // Print the current gain value of the light sensor
  Serial.print  (F("light sensor gain "));
  tsl2591Gain_t gain = tsl.getGain();
  switch(gain)
  {
    case TSL2591_GAIN_LOW:
      Serial.println(F("1x (Low)"));
      break;
    case TSL2591_GAIN_MED:
      Serial.println(F("25x (Medium)"));
      break;
    case TSL2591_GAIN_HIGH:
      Serial.println(F("428x (High)"));
      break;
    case TSL2591_GAIN_MAX:
      Serial.println(F("9876x (Max)"));
      break;
  }
}
void print_light_sensor_gain_values(){
  // Print the allowable gain values of the light sensor
  Serial.println(F("light sensor gains: 1x, 25x, 428x, 9876x"));
}

void set_light_sensor_gain(int newgain){
  // Set the current gain value of the light sensor, and print a confirmation.
  switch(newgain){
    case 1:
      tsl.setGain(TSL2591_GAIN_LOW);
      break;
    case 25:
      tsl.setGain(TSL2591_GAIN_MED);
      break;
    case 428:
      tsl.setGain(TSL2591_GAIN_HIGH);
      break;
    case 9876:
      tsl.setGain(TSL2591_GAIN_MAX);
      break;
    default:
      Serial.println(F("Error: gain may only be 1, 25, 428, or 9876."));
      return;
  }
  print_light_sensor_gain();
}

void print_light_sensor_integration_time(){
  // Print the current integration time in milliseconds.
  Serial.print  (F("light sensor integration time "));
  Serial.print((tsl.getTiming() + 1) * 100, DEC);
  Serial.println(F(" ms"));
}

void print_light_sensor_intensity(){
  // Print the current light value
  uint16_t x = tsl.getLuminosity(TSL2591_FULLSPECTRUM);
  Serial.println(x, DEC);
}
#endif // ADAFRUIT_TSL2591


#ifdef ADAFRUIT_ADS1115
Adafruit_ADS1115 ads; // pass in a number for the sensor identifier (for your use later)

void setup_light_sensor(){
  ads.begin();
  ads.setGain(GAIN_ONE);
}

void print_light_sensor_gain(){
  // Print the current gain value of the light sensor
  Serial.print  (F("light sensor gain "));
  adsGain_t gain = ads.getGain();
  switch(gain)
  {
    case GAIN_TWOTHIRDS:
      Serial.println(F("0.66x (specify 0)"));
      break;
    case GAIN_ONE:
      Serial.println(F("1x"));
      break;
    case GAIN_TWO:
      Serial.println(F("2x"));
      break;
    case GAIN_FOUR:
      Serial.println(F("4x"));
      break;
    case GAIN_EIGHT:
      Serial.println(F("8x"));
      break;
    case GAIN_SIXTEEN:
      Serial.println(F("16x"));
      break;
  }
}
void print_light_sensor_gain_values(){
  // Print the allowable gain values of the light sensor
  Serial.println(F("light sensor gains: 0.66x (specify 0), 1x, 2x, 4x, 8x, 16x"));
}

void set_light_sensor_gain(int newgain){
  // Set the current gain value of the light sensor, and print a confirmation.
  switch(newgain){
    case 0:
      ads.setGain(GAIN_TWOTHIRDS);
      break;
    case 1:
      ads.setGain(GAIN_ONE);
      break;
    case 2:
      ads.setGain(GAIN_TWO);
      break;
    case 4:
      ads.setGain(GAIN_FOUR);
      break;
    case 8:
      ads.setGain(GAIN_EIGHT);
      break;
    case 16:
      ads.setGain(GAIN_SIXTEEN);
      break;
    default:
      Serial.println(F("Error: gain may only be 0, 1, 2, 4, 8, 16 (0 means 2/3)."));
      return;
  }
  print_light_sensor_gain();
}

void print_light_sensor_integration_time(){
  // Print the current integration time in milliseconds.
  Serial.println(F("integration time not supported for ADS1115"));
}

void print_light_sensor_intensity(){
  // Print the current light value
  // uint16_t x = ads.readADC_SingleEnded(0); //single ended measurement on pin 0
  uint16_t x = ads.readADC_Differential_0_1(); //differential measurement on pins 0,1
  Serial.println(x, DEC);
}

#endif // ADAFRUIT_ADS1115

void loop() {
  if(Serial.available() > 0){
    // wait for a serial command and read it
    int received_bytes = Serial.readBytesUntil('\n',input_buffer,INPUT_BUFFER_LENGTH-1);

    if(received_bytes > 0){
        input_buffer[received_bytes] = '\0';
        String command = String(input_buffer);


        if(command.startsWith("move_rel ") or command.startsWith("mr ")){ //relative move
          int preceding_space = -1;
          long displacement[n_motors];
          EACH_MOTOR{ //read three integers and store in steps_remaining
            preceding_space = command.indexOf(' ',preceding_space+1);
            if(preceding_space<0){
              Serial.println(F("Error: command is mr <int> <int> <int>"));
              break;
            }
            duty_cycle[i] = command.substring(preceding_space+1).toInt();
          }
        }
        u32 result = 0;
        sensor1.get_channel_result(CHANNEL_0, &result);
        Serial.println(result);
      // Serial.print(",");
      // sensor1.get_channel_result(CHANNEL_1, &result);
      // Serial.println(result);
    }
  }
  EACH_MOTOR{
    if (duty_cycle[i] >= -1000000000 && duty_cycle[i] < 1000000000){
      if (stepper_cycle % duty_cycle[i] == 0){
        for (int j=0;j<5;j++){
          stepMotor(i,duty_cycle[i] > 0 ? 1 : -1);
          delayMicroseconds(750);
        }
      }
    } else{
      releaseMotor(i);
    }
  }
  stepper_cycle += 1;
  delayMicroseconds(750);
  return;
  
//   // wait for a serial command and read it
//   int received_bytes = Serial.readBytesUntil('\n',input_buffer,INPUT_BUFFER_LENGTH-1);

//   if(received_bytes > 0){
//     input_buffer[received_bytes] = '\0';
//     String command = String(input_buffer);
// //    Serial.println("Got command: <"+command+">");

//     const char* single_axis_moves[3] = {"mrx ", "mry ", "mrz "};
//     int axis = command_prefix(command, single_axis_moves, 3);
//     if(axis >= 0){
//       int preceding_space = command.indexOf(' ',0);
//       if(preceding_space <= 0) Serial.println("Bad command.");
//       long n_steps = command.substring(preceding_space+1).toInt();
//       long displacement[n_motors];
//       EACH_MOTOR displacement[i]=0;
//       displacement[axis]=n_steps;
//       move_axes(displacement);
//       Serial.println("done");
//       return;
//     }

//     if(command.startsWith("list_modules")){//list available modules
//       #if defined ADAFRUIT_TSL2591
//       Serial.println(F("Light Sensor: TSL2591"));
//       #elif defined ADAFRUIT_ADS1115
//       Serial.println(F("Light Sensor: ADS1115"));
//       #endif
//       #if defined(ENDSTOPS_MIN)||defined(ENDSTOPS_MAX)
//       Serial.print("Endstops:");
//       #ifdef ENDSTOPS_MIN
//         Serial.print(" min");
//       #endif
//       #ifdef ENDSTOPS_MAX
//         Serial.print(" max");
//       #endif
//       #ifdef ENDSTOPS_SOFT
//         Serial.print(" soft");
//       #endif
//       Serial.println();
//       #endif
//       Serial.println("--END--");
//       return;
//     }

//     if(command.startsWith("move_rel ") or command.startsWith("mr ")){ //relative move
//       int preceding_space = -1;
//       long displacement[n_motors];
//       EACH_MOTOR{ //read three integers and store in steps_remaining
//         preceding_space = command.indexOf(' ',preceding_space+1);
//         if(preceding_space<0){
//           Serial.println(F("Error: command is mr <int> <int> <int>"));
//           break;
//         }
//         displacement[i] = command.substring(preceding_space+1).toInt();
//       }
//       move_axes(displacement);
//       Serial.println("done.");
//       return;
//     }

//     if(command.startsWith("release")){ //release steppers (de-energise coils)
//       EACH_MOTOR releaseMotor(i);
//       Serial.println("motors released");
//       return;
//     }
//     if(command.startsWith("p?") or command.startsWith("position?")){
//       print_position();
//       return;
//     }
//     if(command.startsWith("ramp_time ")){
//       int preceding_space = command.indexOf(' ',0);
//       if(preceding_space <= 0){
//         Serial.println("Bad command.");
//         return;
//       }
//       ramp_time = command.substring(preceding_space+1).toInt();
//       EEPROM.put(ramp_time_eeprom, ramp_time);
//       Serial.println("done.");
//       return;
//     }
//     if(command.startsWith("ramp_time?")){
//       Serial.print("ramp time ");
//       Serial.println(ramp_time);
//       return;
//     }
//     if(command.startsWith("min_step_delay ") || command.startsWith("dt ")){
//       int preceding_space = command.indexOf(' ',0);
//       if(preceding_space <= 0){
//         Serial.println("Bad command.");
//         return;
//       }
//       min_step_delay = command.substring(preceding_space+1).toInt();
//       EEPROM.put(min_step_delay_eeprom, min_step_delay);
//       Serial.println("done.");
//       return;
//     }
//     if(command.startsWith("min_step_delay?") || command.startsWith("dt?")){
//       Serial.print("minimum step delay ");
//       Serial.println(min_step_delay);
//       return;
//     }
//     if(command.startsWith("zero")){
//       EACH_MOTOR current_pos[i]=0;
//       Serial.println("position reset to 0 0 0");
//       EEPROM.put(0, current_pos);
//       return;
//     }
//     #ifdef LIGHT_SENSOR
//     if(command.startsWith("light_sensor_gain?")){
//       print_light_sensor_gain();
//       return;
//     }
//     if(command.startsWith("light_sensor_gain_values?")){
//       print_light_sensor_gain_values();
//       return;
//     }
//     if(command.startsWith("light_sensor_gain ")){
//       int preceding_space = command.indexOf(' ',0);
//       if(preceding_space <= 0){
//         Serial.println("Bad command.");
//         return;
//       }
//       set_light_sensor_gain(command.substring(preceding_space+1).toInt());
//       return;
//     }
//     if(command.startsWith("light_sensor_integration_time?")){
//       print_light_sensor_integration_time();
//       return;
//     }
//     if(command.startsWith("light_sensor_intensity?")){
//       print_light_sensor_intensity();
//       return;
//     }
//     #endif //LIGHT_SENSOR
//     #if defined(ENDSTOPS_MIN)||defined(ENDSTOPS_MAX)
//     if(command.startsWith("endstops?")){
//        print_endstops_status();
//        return;
//     }
//     if(command.startsWith("home_min")){
//       if(command.indexOf(" ")==-1)
//         return home_min();
//       byte val=command.substring(9).toInt();
//       return home_min(val);
//     }
//     if(command.startsWith("home_max")){
//       if(command.indexOf(" ")==-1)
//         return home_max();
//       byte val=command.substring(9).toInt();
//       return home_max(val);
//     }
//     if(command.startsWith("max_p?")){
//       print_axes_max();
//       return;
//     }
//     if(command.startsWith("max ")){
//       int s2;
//       int s1=command.indexOf(" ",4);
//       axis_max[0]=command.substring(4,s1).toInt();
//       s2=command.indexOf(" ", s1+1);
//       axis_max[1]=command.substring(s1,s2).toInt();
//       axis_max[2]=command.substring(s2).toInt();
//       EEPROM.put(axis_max_eeprom, axis_max);
//       Serial.println("done");
//       return;
//     }
//     if(command.startsWith("test_mode")){
//       if(command.equals("test_mode on")){
//         test_mode=true;
//         Serial.println("test_mode on");
//       }else{
//         test_mode=false;
//         Serial.println("test_mode off");
//       }
//       return;
//     }
//     #endif //ENDSTOPS

//     if(command.startsWith("version")){
//       Serial.println(F(VER_STRING));
//       return;
//     }
//     if(command.startsWith("board")){
//       Serial.println(F(BOARD_STRING));
//       return;
//     }

//     //inductive reading
//     if(command.startsWith("i0?")){
//       u32 result = 0;
//       sensor1.get_channel_result(CHANNEL_0, &result);
//       Serial.println(result);
//     }
    
//     if(command.startsWith("i1?")){
//       u32 result = 0;
//       sensor1.get_channel_result(CHANNEL_1, &result);
//       Serial.println(result);
//     }
    
//     if(command.startsWith("help")){
//       Serial.println("");
//       Serial.print("Board: ");
//       Serial.println(F(BOARD_STRING));
//       Serial.println("");
//       Serial.println(F(VER_STRING));
      
//       #if defined ADAFRUIT_TSL2591
//       Serial.println(F("Compiled with Adafruit TSL2591 support"));
//       #elif defined ADAFRUIT_ADS1115
//       Serial.println(F("Compiled with Adafruit ADS1115 support"));
//       #endif
//       #ifdef ENDSTOPS_MIN
//         Serial.println(F("Compiled with min endstops support"));
//       #endif
//       #ifdef ENDSTOPS_MAX
//         Serial.println(F("Compiled with max endstops support"));
//       #endif
      

//       Serial.println("");
//       Serial.println(F("Commands (terminated by a newline character):"));
//       Serial.println(F("mrx <d>                        - relative move in x"));
//       Serial.println(F("mry <d>                        - relative move in y"));
//       Serial.println(F("mrz <d>                        - relative move in z"));
//       Serial.println(F("mr <d> <d> <d>                 - relative move in all 3 axes"));
//       Serial.println(F("release                        - de-energise all motors"));
//       Serial.println(F("p?                             - print position (3 space-separated integers"));
//       Serial.println(F("ramp_time <d>                  - set the time taken to accelerate/decelerate in us"));
//       Serial.println(F("min_step_delay <d>             - set the minimum time between steps in us."));
//       Serial.println(F("dt <d>                         - set the minimum time between steps in us."));
//       Serial.println(F("ramp_time?                     - get the time taken to accelerate/decelerate in us"));
//       Serial.println(F("min_step_delay?                - get the minimum time between steps in us."));
//       Serial.println(F("zero                           - set the current position to zero."));
//       #ifdef LIGHT_SENSOR
//       Serial.println(F("light_sensor_gain <d>          - set the gain of the light sensor"));
//       Serial.println(F("light_sensor_gain?             - get the gain of the light sensor"));
//       Serial.println(F("light_sensor_gain_values?      - get the allowable gain values of the light sensor"));
//       Serial.println(F("light_sensor_integration_time? - get the integration time in milliseconds"));
//       Serial.println(F("light_sensor_intensity?        - read the current value from the full spectrum diode"));
//       #endif //LIGHT_SENSOR

//       #if defined(ENDSTOPS_MIN)||defined(ENDSTOPS_MAX)
//       Serial.println(F("endstops?                      - get triggered endstops in (1,0,-1) format for max, none, min"));
//       Serial.println(F("home_min <axes?>               - home given (00000zyx byte, e.g. 1 for x) or all axes to their min position"));
//       Serial.println(F("home_max <axes?>               - home given (00000zyx byte, e.g. 3 for x and y) or all axes to their max position"));
//       Serial.println(F("max_p?                         - return positions of max endstops"));
//       Serial.println(F("max <d> <d> <d>                - set maximum positions"));
//       #endif
//       Serial.println(F("test_mode <s>                  - set test_mode <on> <off>"));
//       Serial.println(F("version                        - get firmware version string"));
//       Serial.println("");
//       Serial.println("Input Key:");
//       Serial.println(F("<d>                            - a decimal integer."));
//       Serial.println("");
//       Serial.println("--END--");
//       return;
//     }
//     //Serial.println(F("Type 'help' for a list of commands."));
//   }else{
//     delay(1);
//     return;
}
