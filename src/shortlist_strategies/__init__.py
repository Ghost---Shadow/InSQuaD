from shortlist_strategies.quaild_gain_counter import QuaildGainCounterStrategy
from shortlist_strategies.random_strategy import RandomStrategy
from shortlist_strategies.zero_shot_strategy import ZeroShotStrategy


SHORTLIST_STRATEGIES_LUT = {
    QuaildGainCounterStrategy.NAME: QuaildGainCounterStrategy,
    RandomStrategy.NAME: RandomStrategy,
    ZeroShotStrategy.NAME: ZeroShotStrategy,
}
