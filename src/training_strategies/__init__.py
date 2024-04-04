from training_strategies.quaild import QuaildStrategy
from training_strategies.pick_and_rerank import PickAndRerankStrategy


TRAINING_STRATEGIES_LUT = {
    "pick_and_rerank": PickAndRerankStrategy,
    "quaild": QuaildStrategy,
}
