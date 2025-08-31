import abc


class Imu(abc.ABC):

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def get_gyro(self):
        pass

    @abc.abstractmethod
    def get_acc(self):
        pass
