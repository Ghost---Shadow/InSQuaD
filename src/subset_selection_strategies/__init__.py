from subset_selection_strategies.fast_vote_k_subset import FastVoteKSubsetStrategy
from subset_selection_strategies.flat_cutoff import FlatCutoffStrategy
from subset_selection_strategies.ideal_subset import IdealSubsetStrategy
from subset_selection_strategies.no_operation import NoOperationSubsetSelectionStrategy
from subset_selection_strategies.quaild_submodular import QuaildSubmodularStrategy
from subset_selection_strategies.random_subset import RandomSubsetStrategy
from subset_selection_strategies.top_k_strategy import TopKStrategy


SUBSET_SELECTION_STRATEGIES_LUT = {
    FastVoteKSubsetStrategy.NAME: FastVoteKSubsetStrategy,
    FlatCutoffStrategy.NAME: FlatCutoffStrategy,
    IdealSubsetStrategy.NAME: IdealSubsetStrategy,
    QuaildSubmodularStrategy.NAME: QuaildSubmodularStrategy,
    RandomSubsetStrategy.NAME: RandomSubsetStrategy,
    TopKStrategy.NAME: TopKStrategy,
    NoOperationSubsetSelectionStrategy.NAME: NoOperationSubsetSelectionStrategy,
}
