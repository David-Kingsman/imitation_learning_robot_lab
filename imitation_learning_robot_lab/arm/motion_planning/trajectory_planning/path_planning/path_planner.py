from imitation_learning_lerobot.arm.interface import StrategyWrapper


class PathPlanner(StrategyWrapper):

    def interpolate(self, s: float):
        return self.strategy.interpolate(s)
