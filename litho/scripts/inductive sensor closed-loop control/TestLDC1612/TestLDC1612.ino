#include <Wire.h>

void setup()
{
  Wire.begin();        // join i2c bus (address optional for master)
  Serial.begin(9600);  // start serial for output
  Wire.beginTransmission(0x2A);
  byte busStatus = Wire.endTransmission();
  if (busStatus != 0)
  {
    Serial.print("Device/Sensor is not found!");
    while(1);  //wait for ever
  }
  Serial.print("Device/Sensor is found.");
}

void loop()
{
  byte m = Wire.requestFrom(0x2A, 6);    // request 6 bytes from peripheral 0x2A
  for (int i = 0; i < m; i++)
  {
    byte y = Wire.read();
    Serial.println(y, HEX);
  }
  delay(1000);  //test interval
}