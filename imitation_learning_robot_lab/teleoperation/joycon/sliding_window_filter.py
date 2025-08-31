from collections import deque


class SlidingFilter:
    def __init__(self, window_size):
        super().__init__()

        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def add_sample(self, sample):
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]

        self.window.append(sample)
        self.sum += sample

        return self.sum / len(self.window)