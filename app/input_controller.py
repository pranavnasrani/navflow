"""
Global Input Controller for the USB HID Macropad and Keyboard.
Maintains synchronized (Z_index, Wind_Speed) state across all 4 exhibition windows.
"""

from typing import Callable
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent


class ExhibitionInputController(QObject):
    # Signals emitted whenever visitor changes inputs
    state_changed = Signal(int, float)  # (z_step_index, wind_speed)

    def __init__(self,
                 total_z_steps: int = 10,
                 min_speed: float = 10.0,
                 max_speed: float = 50.0,
                 speed_step: float = 2.5,
                 controls_cfg: dict = None):
        super().__init__()
        self.total_z_steps = total_z_steps
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_step = speed_step
        self.controls_cfg = controls_cfg or {}

        # Default initial state: middle Z-step and moderate wind speed
        self.current_z_index = total_z_steps // 2
        self.current_wind_speed = (min_speed + max_speed) / 2.0

    def set_z_step(self, z_idx: int):
        """Sets the active Z-step index (0..total_z_steps-1)."""
        new_idx = max(0, min(self.total_z_steps - 1, z_idx))
        if new_idx != self.current_z_index:
            self.current_z_index = new_idx
            self.state_changed.emit(self.current_z_index, self.current_wind_speed)

    def step_z(self, delta: int):
        """Moves Z-step up (+1) or down (-1)."""
        self.set_z_step(self.current_z_index + delta)

    def set_wind_speed(self, speed: float):
        """Sets continuous wind speed in m/s."""
        new_speed = max(self.min_speed, min(self.max_speed, speed))
        if abs(new_speed - self.current_wind_speed) > 1e-4:
            self.current_wind_speed = new_speed
            self.state_changed.emit(self.current_z_index, self.current_wind_speed)

    def step_wind_speed(self, delta_m_s: float):
        """Adjusts wind speed (e.g. from rotary knob twist)."""
        self.set_wind_speed(self.current_wind_speed + delta_m_s)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """
        Processes HID keypresses from the macropad or regular keyboard.
        Returns True if handled, False otherwise.
        """
        key_text = event.text()
        key_code = event.key()

        # 1. 10 Discrete Z-Step Buttons (Macropad row: '1'..'9', '0')
        digit_map = {
            Qt.Key_1: 0, Qt.Key_2: 1, Qt.Key_3: 2, Qt.Key_4: 3, Qt.Key_5: 4,
            Qt.Key_6: 5, Qt.Key_7: 6, Qt.Key_8: 7, Qt.Key_9: 8, Qt.Key_0: 9
        }
        if key_code in digit_map:
            self.set_z_step(digit_map[key_code])
            return True

        # 2. Z Step Up / Down
        if key_code in [Qt.Key_Up, Qt.Key_W]:
            self.step_z(1)
            return True
        elif key_code in [Qt.Key_Down, Qt.Key_S]:
            self.step_z(-1)
            return True

        # 3. Wind Velocity Knob (Clockwise / Counter-Clockwise pulses)
        if key_code in [Qt.Key_Right, Qt.Key_D, Qt.Key_Plus, Qt.Key_Equal, Qt.Key_BracketRight]:
            self.step_wind_speed(self.speed_step)
            return True
        elif key_code in [Qt.Key_Left, Qt.Key_A, Qt.Key_Minus, Qt.Key_Underscore, Qt.Key_BracketLeft]:
            self.step_wind_speed(-self.speed_step)
            return True

        return False

    def handle_wheel_event(self, event: QWheelEvent) -> bool:
        """Mouse wheel can simulate turning the physical velocity rotary knob."""
        num_degrees = event.angleDelta().y() / 8.0
        num_steps = num_degrees / 15.0
        if abs(num_steps) > 0.01:
            self.step_wind_speed(num_steps * (self.speed_step * 0.5))
            return True
        return False
