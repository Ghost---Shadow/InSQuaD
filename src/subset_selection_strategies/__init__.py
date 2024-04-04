from subset_selection_strategies.gain_cutoff import GainCutoffStrategy
from subset_selection_strategies.top_k_strategy import TopKStrategy


SUBSET_SELECTION_STRATEGIES_LUT = {
    "top_k": TopKStrategy,
    "gain_cutoff": GainCutoffStrategy,
}
