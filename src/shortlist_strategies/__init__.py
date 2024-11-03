from shortlist_strategies.insquad_combinatorial import InsquadCombinatorialStrategy
from shortlist_strategies.least_confidence import LeastConfidenceStrategy
from shortlist_strategies.oracle_strategy import OracleStrategy
from shortlist_strategies.quaild_gain_counter import QuaildGainCounterStrategy
from shortlist_strategies.quaild_similar_strategy import QuaildSimilarStrategy
from shortlist_strategies.random_strategy import RandomStrategy
from shortlist_strategies.shortlist_then_topk_strategy import ShortlistThenTopK
from shortlist_strategies.zero_shot_strategy import ZeroShotStrategy


SHORTLIST_STRATEGIES_LUT = {
    InsquadCombinatorialStrategy.NAME: InsquadCombinatorialStrategy,
    LeastConfidenceStrategy.NAME: LeastConfidenceStrategy,
    OracleStrategy.NAME: OracleStrategy,
    QuaildGainCounterStrategy.NAME: QuaildGainCounterStrategy,
    QuaildSimilarStrategy.NAME: QuaildSimilarStrategy,
    RandomStrategy.NAME: RandomStrategy,
    ShortlistThenTopK.NAME: ShortlistThenTopK,
    ZeroShotStrategy.NAME: ZeroShotStrategy,
}
