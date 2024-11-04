#include<SoftwareSerial.h>
#include "Seeed_LDC1612.h"

LDC1612 sensor;
SoftwareSerial SUART(0, 1); //SRX = 0, STX = 1

void setup() 
{
  Serial.begin(9600);
  delay(100);
  Serial.println("start!");
  sensor.init();
  if(sensor.single_channel_config(CHANNEL_0))
  {
      Serial.println("can't detect sensor!");
      while(1);
  }
}

void loop() 
{
  u32 result_channel1=0;
  u32 result_channel2=0;

  sensor.get_channel_result(CHANNEL_0,&result_channel1);

  if(0!=result_channel1)
  {
      Serial.println(result_channel1);
  }

  // SUART.println("Hello UNO-2!");
  delay(1000);
}
