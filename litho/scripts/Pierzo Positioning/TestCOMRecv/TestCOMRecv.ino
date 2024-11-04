#include <SoftwareSerial.h>
SoftwareSerial SUART(0, 1); // SRX = 0, STX = 1

String inputLine = ""; // Buffer to hold incoming characters

void setup() {
  Serial.begin(9600);
  SUART.begin(9600);
}

void loop() {
  while (SUART.available() > 0) {
    char ch = SUART.read(); // Read each character
    if (ch == '\n') { // Check if the character is a newline
      Serial.println(inputLine); // Print the complete line
      inputLine = ""; // Clear the buffer for the next line
    } else {
      inputLine += ch; // Append character to buffer
    }
  }
}
