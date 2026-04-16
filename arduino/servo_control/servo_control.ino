/*
 * EcoVision — Servo Control for Smart Bin
 *
 * Receives a servo angle (0, 90, or 180) from Python via Serial,
 * then rotates the servo to that position.
 *
 * Wiring:
 *   - Servo signal → Pin 9
 *   - Servo VCC    → 5V (or external supply for high-torque servos)
 *   - Servo GND    → GND
 *
 * Serial: 9600 baud, angle sent as ASCII string terminated with '\n'
 */

#include <Servo.h>

Servo binServo;

const int SERVO_PIN = 9;
const int DEFAULT_ANGLE = 90;  // starting position (Organic / center)

void setup() {
    Serial.begin(9600);
    binServo.attach(SERVO_PIN);
    binServo.write(DEFAULT_ANGLE);  // move to center on boot
    Serial.println("EcoVision Servo Ready");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        int angle = input.toInt();

        // Validate angle range
        if (angle >= 0 && angle <= 180) {
            binServo.write(angle);
            Serial.print("OK: moved to ");
            Serial.print(angle);
            Serial.println(" degrees");
        } else {
            Serial.print("ERR: invalid angle ");
            Serial.println(input);
        }
    }
}
