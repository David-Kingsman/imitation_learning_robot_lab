import numpy as np
from pynput import keyboard

from ..handler import Handler


class PickBoxKeyboardHandler(Handler):
    _name = "pick_box_keyboard"

    def __init__(self):
        super().__init__()

        self._action = np.zeros(4)

        self._done = False
        self._vel = 0.005

        self._listener: keyboard.Listener = None

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if not self._sync:

            if key_char == 'Key.ctrl_r':
                self._sync = True
        else:
            if key_char == 'Key.shift_r':
                self._sync = False

        if key_char == 'Key.enter':
            self._done = True

        if not self._sync:
            return

        if key_char == '2':
            self._action[2] += self._vel

        if key_char == '8':
            self._action[2] -= self._vel

        if key_char == "6":
            self._action[0] -= self._vel

        if key_char == '4':
            self._action[0] += self._vel

        if key_char == '7':
            self._action[1] += self._vel

        if key_char == '1':
            self._action[1] -= self._vel

        if key_char == '9':
            self._action[3] += 0.05

        if key_char == '3':
            self._action[3] -= 0.05

        self._action[3] = np.clip(self._action[3], 0.0, 1.0)

    def start(self):
        self._listener = keyboard.Listener(on_press=self.on_press)
        self._listener.start()

    def close(self):
        self._listener.stop()

    def print_info(self):
        print("------------------------------")
        print("Start:           Right Ctrl")
        print("Pause:           Right Shift")
        print("Stop:            Enter")
        print("+X:              Keypad 1")
        print("-X:              Keypad 7")
        print("+Y:              Keypad 6")
        print("-Y:              Keypad 4")
        print("+Z:              Keypad 8")
        print("-Z:              Keypad 2")
        print("Open:            Keypad 3")
        print("Close:           Keypad 9")
