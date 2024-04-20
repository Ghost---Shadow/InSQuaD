from training_strategies.no_operation import NoOperation
from training_strategies.pick_and_rerank import PickAndRerankStrategy
from training_strategies.quaild import QuaildStrategy


TRAINING_STRATEGIES_LUT = {
    NoOperation.NAME: NoOperation,
    PickAndRerankStrategy.NAME: PickAndRerankStrategy,
    QuaildStrategy.NAME: QuaildStrategy,
}
