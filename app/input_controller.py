"""
Global Input Controller for the USB HID Macropad and Keyboard.
Maintains one synchronized (wand_index, knob, view) state across all 4 exhibition windows.

  * 10 macropad buttons pick the smoke-wand height (wand index 0..9).
  * The rotary knob sweeps the wind direction: knob in [0, 1], where the state manager maps
    it to d = d_min + knob * (d_max - d_min)  (IGCSE 0606 linear interpolation).
  * Tab cycles the view mode 0 -> 1 -> 2 -> 0.
  * Space switches every screen between the simple and the detailed instrument strip.
  * R puts every camera back to its starting view (after visitors have dragged or zoomed it).
  * When nobody has touched anything for `idle_seconds`, a show runs by itself (app/explainers.py): the
    knob, the wand and the view move on their own and each screen explains what it is showing. The first
    touch from a visitor, the macropad or the iPad stops it.
"""

import time

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent

from app.explainers import SHOW_STEPS, smoothstep

VIEW_COUNT = 3
IDLE_SECONDS = 25.0
SHOW_TICK_MS = 66  # about 15 updates a second while the show moves the knob


class ExhibitionInputController(QObject):
    # Signals emitted whenever visitor changes inputs
    state_changed = Signal(int, float)  # (wand_index, knob)
    view_changed = Signal(int)          # view mode
    detail_changed = Signal(bool)       # True = detailed instrument strip, False = simple
    camera_reset = Signal()             # every window returns to its starting camera
    show_changed = Signal(bool)         # True while the idle show runs
    show_step_changed = Signal(int)     # which of the show's steps is playing

    def __init__(self, wand_count: int = 10, knob_step: float = 1 / 32, idle_seconds: float = IDLE_SECONDS):
        super().__init__()
        self.wand_count = wand_count
        self.knob_step = knob_step
        self.idle_seconds = idle_seconds  # 0 turns the show off

        # Default initial state: a mid-height wand, knob fully anticlockwise (first direction),
        # and the simple strip, since most visitors walk up cold.
        self.current_wand_index = 4
        self.current_knob = 0.0
        self.current_view = 0
        self.detailed = False

        # The idle show. `_driving` is True while the show itself moves a control, so that is not mistaken
        # for a visitor touching it.
        self.show_active = False
        self.show_step = 0
        self._driving = False
        self._last_input = time.monotonic()
        self._step_started = 0.0
        self._knob_from = 0.0
        self._clock = QTimer(self)
        self._clock.setInterval(SHOW_TICK_MS)
        self._clock.timeout.connect(self._tick)

    # ----------------------------------------------------------------------------------------------
    # Idle show
    # ----------------------------------------------------------------------------------------------
    def start_clock(self, start_show: bool = True):
        """Begin watching for idleness (call once the windows exist). `start_show` runs the show at once."""
        self._last_input = time.monotonic()
        self._clock.start()
        if start_show and self.idle_seconds > 0:
            self.start_show()

    def _human(self):
        """Something other than the show touched a control: restart the idle count and stop the show."""
        if self._driving:
            return
        self._last_input = time.monotonic()
        if self.show_active:
            self.stop_show()

    def touch(self):
        """Windows call this for mouse drags, pinches and Cmd + scroll: they are visitors too."""
        self._human()

    def start_show(self):
        if self.show_active:
            return
        self.show_active = True
        self._driving = True
        if self.detailed:
            self.toggle_detail()  # the show runs in the simple view
        self._driving = False
        self.show_changed.emit(True)
        self._enter_step(0)

    def stop_show(self):
        if not self.show_active:
            return
        self.show_active = False
        self.show_changed.emit(False)

    def _enter_step(self, index: int):
        self.show_step = index % len(SHOW_STEPS)
        step = SHOW_STEPS[self.show_step]
        self._step_started = time.monotonic()
        self._knob_from = self.current_knob
        self._driving = True
        self.set_wand(step.wand)
        self.set_view(step.view)
        self._driving = False
        self.show_step_changed.emit(self.show_step)

    def _tick(self):
        now = time.monotonic()
        if not self.show_active:
            if self.idle_seconds > 0 and now - self._last_input >= self.idle_seconds:
                self.start_show()
            return
        step = SHOW_STEPS[self.show_step]
        t = (now - self._step_started) / step.seconds
        if t >= 1.0:
            self._enter_step(self.show_step + 1)
            return
        if step.knob_to is not None:
            # straight line from where the knob was to where this step sends it, eased by smoothstep
            knob = self._knob_from + (step.knob_to - self._knob_from) * smoothstep(t)
            self._driving = True
            self.set_knob(knob)
            self._driving = False

    # ----------------------------------------------------------------------------------------------
    def toggle_detail(self):
        """Simple <-> detailed instrument strip, on all screens at once."""
        self._human()
        self.detailed = not self.detailed
        self.detail_changed.emit(self.detailed)

    def set_wand(self, index: int):
        """Sets the active smoke-wand index (0..wand_count-1)."""
        self._human()
        new_idx = max(0, min(self.wand_count - 1, int(index)))
        if new_idx != self.current_wand_index:
            self.current_wand_index = new_idx
            self.state_changed.emit(self.current_wand_index, self.current_knob)

    def step_wand(self, delta: int):
        """Moves the wand up (+1) or down (-1)."""
        self.set_wand(self.current_wand_index + delta)

    def set_knob(self, knob: float):
        """Sets the direction knob, clamped to [0, 1]."""
        self._human()
        new_knob = max(0.0, min(1.0, float(knob)))
        if abs(new_knob - self.current_knob) > 1e-9:
            self.current_knob = new_knob
            self.state_changed.emit(self.current_wand_index, self.current_knob)

    def step_knob(self, steps: float):
        """Turns the knob by `steps` detents (one detent is knob_step of the full sweep)."""
        self.set_knob(self.current_knob + steps * self.knob_step)

    def set_view(self, view: int):
        """Sets the view mode (0..VIEW_COUNT-1)."""
        self._human()
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
        self._human()  # any key counts as a visitor, even one that does nothing

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

        # 5. Simple / detailed strip, and camera reset
        if key_code == Qt.Key_Space:
            self.toggle_detail()
            return True
        if key_code == Qt.Key_R:
            self.camera_reset.emit()
            return True

        return False

    def handle_wheel_event(self, event: QWheelEvent) -> bool:
        """Mouse wheel can simulate turning the physical rotary knob (one notch = 15 degrees = 1 detent)."""
        self._human()
        num_degrees = event.angleDelta().y() / 8.0
        num_steps = num_degrees / 15.0
        if abs(num_steps) > 0.01:
            self.step_knob(num_steps)
            return True
        return False
