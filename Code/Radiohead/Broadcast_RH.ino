#include <RH_ASK.h>
#include <SPI.h> // Not actually used but needed to compile

const int txPin = 12;  // Der Pin, über den das Signal gesendet wird, falls nötig anzupassen

// Erstellen des RH_ASK Treibers
RH_ASK driver(10000, txPin, 11, 2, false);  // Argumente: (speed, txPin, rxPin, pttPin, usecs)

const char message[] = "Hi"; // Die Nachricht, die gesendet werden soll

void setup() {
  // Initialisiere den RH_ASK Treiber
  if (!driver.init()) {
    Serial.begin(9600);
    Serial.println("Initialization failed");
  } else {
    Serial.begin(9600);
    Serial.println("Ready to transmit");
  }
}

void loop() {
  // Sende die Nachricht als Byte-Array
  driver.send((uint8_t *)message, sizeof(message) - 1);
  driver.waitPacketSent(); // Warte bis das Paket gesendet wurde
  Serial.println("Message sent");

  // Delay bis zum nächsten Senden
  delay(100);
}
