import time


class RealTimeSync:
    def __init__(self, time_step):
        self.time_step = time_step
        self.next_time = time.time() + time_step

    def sync(self):
        now = time.time()
        if now < self.next_time:
            time.sleep(self.next_time - now)
        self.next_time += self.time_step

    def reset(self):
        self.next_time = time.time() + self.time_step
