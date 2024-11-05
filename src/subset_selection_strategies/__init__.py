from subset_selection_strategies.diversity import DiversitySubsetSelectionStrategy
from subset_selection_strategies.flat_cutoff import FlatCutoffStrategy
from subset_selection_strategies.no_operation import NoOperationSubsetSelectionStrategy
from subset_selection_strategies.quaild_submodular import QuaildSubmodularStrategy
from subset_selection_strategies.random_subset import RandomSubsetStrategy
from subset_selection_strategies.top_k_strategy import TopKStrategy


SUBSET_SELECTION_STRATEGIES_LUT = {
    DiversitySubsetSelectionStrategy.NAME: DiversitySubsetSelectionStrategy,
    FlatCutoffStrategy.NAME: FlatCutoffStrategy,
    NoOperationSubsetSelectionStrategy.NAME: NoOperationSubsetSelectionStrategy,
    QuaildSubmodularStrategy.NAME: QuaildSubmodularStrategy,
    RandomSubsetStrategy.NAME: RandomSubsetStrategy,
    TopKStrategy.NAME: TopKStrategy,
}
