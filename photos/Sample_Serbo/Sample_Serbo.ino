#include <Servo.h>
 
Servo myservo;//Servoオブジェクトの宣言
 
void setup() {
  myservo.attach(9);//servo変数をピンに割り当てる、ここでは9番ピン
  myservo.write(90);//角度を指定、ここでは90度
 
}
 
void loop() {
  myservo.write(0);
  delay(1000);
  myservo.write(90);
  delay(1000);
  myservo.write(180);
  delay(1000);
  myservo.write(90);
  delay(1000);
 
}