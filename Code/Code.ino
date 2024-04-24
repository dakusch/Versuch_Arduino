const int txPin = 12;
const unsigned long bitDuration = 500;
const unsigned long delayBetweenMessages = 3000;

const char wort[] = "Hi";
const int wordLength = sizeof(wort) - 1;
int binaryArray[8 * wordLength];

enum State {
  SEND_PREAMBLE,
  SEND_WORD,
  WAITING
};

State currentState = SEND_PREAMBLE;
unsigned long lastActionTime = 0;
int bitIndex = 0;

void asciiToBinary(const char wort[], int binaryArray[]) {
  for (int i = 0; i < wordLength; i++) {
    char c = wort[i];
    for (int j = 0; j < 8; j++) {
      binaryArray[i * 8 + j] = (c >> (7 - j)) & 1;
    }
  }
}

void setup() {
  pinMode(txPin, OUTPUT);
  asciiToBinary(wort, binaryArray);
}

void loop() {
  unsigned long currentTime = millis();
  
  switch (currentState) {
    case SEND_PREAMBLE:
      if (currentTime - lastActionTime >= bitDuration) {
        digitalWrite(txPin, bitIndex % 2 == 0 ? HIGH : LOW);
        bitIndex++;
        lastActionTime = currentTime;

        if (bitIndex == 8) {
          bitIndex = 0;
          currentState = SEND_WORD;
        }
      }
      break;
    
    case SEND_WORD:
      if (currentTime - lastActionTime >= bitDuration) {
        if (bitIndex < 8 * wordLength) {
          digitalWrite(txPin, binaryArray[bitIndex] ? HIGH : LOW);
          bitIndex++;
          lastActionTime = currentTime;
        } else {
          digitalWrite(txPin, LOW);
          bitIndex = 0;
          currentState = WAITING;
          lastActionTime = currentTime;
        }
      }
      break;

    case WAITING:
      if (currentTime - lastActionTime >= delayBetweenMessages) {
        currentState = SEND_PREAMBLE;
        lastActionTime = currentTime;
      }
      break;
  }
}
