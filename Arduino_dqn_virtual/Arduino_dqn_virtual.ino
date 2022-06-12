#include <Servo.h>

Servo mServo;

int state[2];
int pin_number = 10;

void setup() {
  mServo.attach(9);
  mServo.write(10);
  delay(500);
  Serial.begin(9600);
  pinMode(pin_number, OUTPUT);
  state[0] = 0;
  state[1] = 1;
  digitalWrite(pin_number, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'p') {
      if (state[0] == 0) {
        Serial.print("0");
        mServo.write(90);
        delay(500);
        state[0] = 1;
      }
      else {
        Serial.print("0");
        mServo.write(10);
        delay(500);
        state[0] = 0;
        state[1] = 1;
        digitalWrite(10, LOW);
      }
    }
    else if (c == 'i') {
      if (state[0] == 1 && state[1] == 1) {
        Serial.print("1");
        delay(500);
        digitalWrite(10, HIGH);
        state[1] = 0;
      }
      else {
        Serial.print("0");
      }
    }
    else if (c == 'c') {
      state[0] = 0;
      state[1] = 1;
      digitalWrite(10, LOW);
      mServo.write(10);
      delay(500);
    }
  }
}
