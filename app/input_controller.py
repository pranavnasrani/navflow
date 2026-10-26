"""
Global Input Controller for the USB HID Macropad and Keyboard.
Maintains one synchronized (wand_index, knob, view) state across all 4 exhibition windows.

  * 10 macropad buttons pick the smoke-wand height (wand index 0..9).
  * The rotary knob sweeps the wind direction: knob in [0, 1], where the state manager maps
    it to d = d_min + knob * (d_max - d_min)  (IGCSE 0606 linear interpolation).
  * Tab cycles the view mode 0 -> 1 -> 2 -> 0.
"""

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent

VIEW_COUNT = 3


class ExhibitionInputController(QObject):
    # Signals emitted whenever visitor changes inputs
    state_changed = Signal(int, float)  # (wand_index, knob)
    view_changed = Signal(int)          # view mode

    def __init__(self, wand_count: int = 10, knob_step: float = 1 / 32):
        super().__init__()
        self.wand_count = wand_count
        self.knob_step = knob_step

        # Default initial state: a mid-height wand, knob fully anticlockwise (first direction).
        self.current_wand_index = 4
        self.current_knob = 0.0
        self.current_view = 0

    def set_wand(self, index: int):
        """Sets the active smoke-wand index (0..wand_count-1)."""
        new_idx = max(0, min(self.wand_count - 1, int(index)))
        if new_idx != self.current_wand_index:
            self.current_wand_index = new_idx
            self.state_changed.emit(self.current_wand_index, self.current_knob)

    def step_wand(self, delta: int):
        """Moves the wand up (+1) or down (-1)."""
        self.set_wand(self.current_wand_index + delta)

    def set_knob(self, knob: float):
        """Sets the direction knob, clamped to [0, 1]."""
        new_knob = max(0.0, min(1.0, float(knob)))
        if abs(new_knob - self.current_knob) > 1e-9:
            self.current_knob = new_knob
            self.state_changed.emit(self.current_wand_index, self.current_knob)

    def step_knob(self, steps: float):
        """Turns the knob by `steps` detents (one detent is knob_step of the full sweep)."""
        self.set_knob(self.current_knob + steps * self.knob_step)

    def set_view(self, view: int):
        """Sets the view mode (0..VIEW_COUNT-1)."""
        new_view = int(view) % VIEW_COUNT
        if new_view != self.current_view:
            self.current_view = new_view
            self.view_changed.emit(self.current_view)

    def cycle_view(self):
        """Next view mode: 0 -> 1 -> 2 -> 0."""
        self.set_view(self.current_view + 1)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """
        Processes HID keypresses from the macropad or regular keyboard.
        Returns True if handled, False otherwise.
        """
        key_code = event.key()

        # 1. 10 Discrete Wand Buttons (Macropad row: '1'..'9', '0')
        digit_map = {
            Qt.Key_1: 0, Qt.Key_2: 1, Qt.Key_3: 2, Qt.Key_4: 3, Qt.Key_5: 4,
            Qt.Key_6: 5, Qt.Key_7: 6, Qt.Key_8: 7, Qt.Key_9: 8, Qt.Key_0: 9
        }
        if key_code in digit_map:
            self.set_wand(digit_map[key_code])
            return True

        # 2. Wand Up / Down
        if key_code in [Qt.Key_Up, Qt.Key_W]:
            self.step_wand(1)
            return True
        elif key_code in [Qt.Key_Down, Qt.Key_S]:
            self.step_wand(-1)
            return True

        # 3. Wind Direction Knob (Clockwise / Counter-Clockwise pulses)
        if key_code in [Qt.Key_Right, Qt.Key_D, Qt.Key_Plus, Qt.Key_Equal, Qt.Key_BracketRight]:
            self.step_knob(1)
            return True
        elif key_code in [Qt.Key_Left, Qt.Key_A, Qt.Key_Minus, Qt.Key_Underscore, Qt.Key_BracketLeft]:
            self.step_knob(-1)
            return True

        # 4. View mode
        if key_code == Qt.Key_Tab:
            self.cycle_view()
            return True

        return False

    def handle_wheel_event(self, event: QWheelEvent) -> bool:
        """Mouse wheel can simulate turning the physical rotary knob (one notch = 15 degrees = 1 detent)."""
        num_degrees = event.angleDelta().y() / 8.0
        num_steps = num_degrees / 15.0
        if abs(num_steps) > 0.01:
            self.step_knob(num_steps)
            return True
        return False
