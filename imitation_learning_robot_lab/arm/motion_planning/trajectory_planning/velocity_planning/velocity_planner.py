from imitation_learning_lerobot.arm.interface import StrategyWrapper


class VelocityPlanner(StrategyWrapper):

    def interpolate(self, t: float):
        return self.strategy.interpolate(t)
