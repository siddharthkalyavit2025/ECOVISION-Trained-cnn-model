"""
Servo motor control via Arduino serial communication.

Maps predicted waste classes to bin categories and sends the
corresponding servo angle to an Arduino over USB serial.
"""

from __future__ import annotations

import logging
import os
import time

import serial  # pyserial

logger = logging.getLogger(__name__)

# ── Bin category → servo angle ──────────────────────────────────────────
BIN_ANGLES: dict[str, int] = {
    "recyclable": 0,
    "organic": 90,
    "other": 180,
}

# ── Waste class → bin category ──────────────────────────────────────────
CLASS_TO_BIN: dict[str, str] = {
    "plastic":   "recyclable",
    "paper":     "recyclable",
    "metal":     "recyclable",
    "glass":     "recyclable",
    "cardboard": "recyclable",
    "biological": "organic",
    "batteries": "other",
    "clothes":   "other",
    "shoes":     "other",
    "trash":     "other",
}

# ── Serial settings (configurable via env vars) ────────────────────────
SERIAL_PORT: str = os.getenv("ARDUINO_PORT", "COM3")
BAUD_RATE: int = int(os.getenv("ARDUINO_BAUD", "9600"))
SERIAL_TIMEOUT: float = 1.0

# ── Global connection ──────────────────────────────────────────────────
_arduino: serial.Serial | None = None


def connect_arduino() -> serial.Serial | None:
    """Open (or reopen) the serial connection to the Arduino.

    Returns the serial object on success, or None if the device
    is unavailable.  Never raises — failures are logged as warnings.
    """
    global _arduino
    try:
        _arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)  # wait for Arduino to reset after serial open
        logger.info(
            "✅ Arduino connected on %s @ %d baud.", SERIAL_PORT, BAUD_RATE
        )
        return _arduino
    except serial.SerialException as exc:
        logger.warning("⚠️ Arduino not available on %s: %s", SERIAL_PORT, exc)
        _arduino = None
        return None


def disconnect_arduino() -> None:
    """Safely close the Arduino serial connection."""
    global _arduino
    if _arduino is not None and _arduino.is_open:
        _arduino.close()
        logger.info("🔌 Arduino disconnected.")
    _arduino = None


def get_bin_category(predicted_class: str) -> str:
    """Return the bin category for a given waste class."""
    return CLASS_TO_BIN.get(predicted_class.lower(), "other")


def get_servo_angle(predicted_class: str) -> int:
    """Return the servo angle (0 / 90 / 180) for a given waste class."""
    category = get_bin_category(predicted_class)
    return BIN_ANGLES[category]


def send_angle_to_arduino(angle: int) -> bool:
    """Send a servo angle to the Arduino.

    Returns True if the angle was sent successfully, False otherwise.
    Automatically attempts to reconnect once if the connection is lost.
    """
    global _arduino

    for attempt in range(2):
        # Ensure we have a connection
        if _arduino is None or not _arduino.is_open:
            if attempt == 0:
                logger.info("Attempting to (re)connect to Arduino …")
                connect_arduino()
            if _arduino is None or not _arduino.is_open:
                continue

        try:
            message = f"{angle}\n"
            _arduino.write(message.encode("utf-8"))
            logger.info(
                "🔧 Sent angle %d° to Arduino (bin: %s).",
                angle,
                {0: "Recyclable", 90: "Organic", 180: "Other"}.get(angle, "?"),
            )
            return True
        except serial.SerialException as exc:
            logger.warning("Serial write failed (attempt %d): %s", attempt + 1, exc)
            disconnect_arduino()

    logger.warning("⚠️ Could not send angle to Arduino — continuing without servo.")
    return False


def rotate_servo_for_class(predicted_class: str) -> dict[str, str | int | bool]:
    """High-level helper: map a class to an angle and send it.

    Returns a small dict with the bin info (useful for logging/debugging).
    """
    category = get_bin_category(predicted_class)
    angle = BIN_ANGLES[category]
    sent = send_angle_to_arduino(angle)

    return {
        "bin_category": category,
        "servo_angle": angle,
        "servo_sent": sent,
    }
