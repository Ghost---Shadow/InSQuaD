from subset_selection_strategies.fast_vote_k import FastVoteKStrategy
from subset_selection_strategies.quaild_gain_cutoff import QuaildGainCutoffStrategy
from subset_selection_strategies.top_k_strategy import TopKStrategy


SUBSET_SELECTION_STRATEGIES_LUT = {
    QuaildGainCutoffStrategy.NAME: QuaildGainCutoffStrategy,
    TopKStrategy.NAME: TopKStrategy,
    FastVoteKStrategy.NAME: FastVoteKStrategy,
}
