#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  150 
#define SERVOMAX  600
#define SERVO_FREQ 50
#define LOW_POS 0  
#define HIGH_POS 60 

bool servoStates[4] = {false, false, false, false};
void setup() {
  Serial.begin(9600);
  Serial.println("Awaiting commands...");

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  delay(10);
}

void loop() {
  if (Serial.available()) {
    int command = Serial.read() - '0';

    if (command >= 0 && command <= 3) {
      toggleServo(command);
    }
  }
}

void toggleServo(int servoNum) {
  servoStates[servoNum] = !servoStates[servoNum];
  int angle = servoStates[servoNum] ? HIGH_POS : LOW_POS; 
  moveToAngle(servoNum, angle);
}

void moveToAngle(int servoNum, int angle) {
  int pulseLength = map(angle, 0, 180, SERVOMIN, SERVOMAX); 
  pwm.setPWM(servoNum, 0, pulseLength);

  Serial.print("Servo ");
  Serial.print(servoNum);
  Serial.print(" moved to ");
  Serial.print(angle);
  Serial.println(" degrees");
}
