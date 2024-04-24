const int txPin = 12;  // Der Pin, über den das Signal gesendet wird
const int dotDuration = 250;  // Dauer eines Punktes in Millisekunden
const int dashDuration = dotDuration * 3;  // Dauer eines Strichs, das Dreifache eines Punktes
const int elementSpace = dotDuration;  // Pause zwischen Elementen eines Buchstabens
const int letterSpace = dotDuration * 3;  // Pause zwischen Buchstaben
const int wordSpace = dotDuration * 7;  // Pause zwischen Wiederholungen des Wortes

enum State {
  SEND_PREAMBLE,
  SEND_D,
  SEND_H,
  SEND_I,
  WAITING
};

State currentState = SEND_PREAMBLE;
unsigned long lastActionTime = 0;
int bitIndex = 0;

void setup() {
  pinMode(txPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  unsigned long currentTime = millis();
  
  switch (currentState) {
    case SEND_PREAMBLE:
      if (currentTime - lastActionTime >= dotDuration) {
        digitalWrite(txPin, bitIndex % 2 == 0 ? HIGH : LOW);
        bitIndex++;
        lastActionTime = currentTime;

        if (bitIndex == 8) {
          bitIndex = 0;
          currentState = SEND_D;
        }
      }
      break;

    case SEND_D:
      sendMorse("-..");
      currentState = SEND_H;
      break;
    
    case SEND_H:
      sendMorse("....");
      currentState = SEND_I;
      break;
    
    case SEND_I:
      sendMorse("..");
      currentState = WAITING;
      break;

    case WAITING:
      if (currentTime - lastActionTime >= wordSpace) {
        currentState = SEND_PREAMBLE;
        lastActionTime = currentTime;
      }
      break;
  }
}

void sendMorse(const char *code) {
  while (*code) {
    switch (*code++) {
      case '.':
        digitalWrite(txPin, HIGH);
        delay(dotDuration);
        digitalWrite(txPin, LOW);
        delay(elementSpace);
        break;
      case '-':
        digitalWrite(txPin, HIGH);
        delay(dashDuration);
        digitalWrite(txPin, LOW);
        delay(elementSpace);
        break;
    }
    lastActionTime = millis();
  }
  delay(letterSpace - elementSpace);  // Abzüglich der letzten Elementpause
}
