from copy import deepcopy
import random
import numpy as np
from subset_selection_strategies.base_strategy import BaseSubsetSelectionStrategy
import torch
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class IdealSubsetStrategy(BaseSubsetSelectionStrategy):
    NAME = "ideal"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.k = config.architecture.subset_selection_strategy.k
        self.number_to_select = self.config.offline_validation.annotation_budget
        self.rand_iter = config.architecture.subset_selection_strategy.rand_iter

    def ic_diffusion_model(self, G, seed):
        """
        Borrowed from
        https://github.com/skzhang1/IDEAL/blob/23b1e1cb049004d8296c81933a692b49c42ab527/core_method.py#L152
        Then asked ChatGPT to clean it up
        """
        total_count = 0
        iter = self.rand_iter
        max_iterations = 100

        for _ in range(iter):
            activated_nodes = deepcopy(seed)
            current_seed = deepcopy(seed)

            iteration = 0  # It is taking forever
            while current_seed and iteration < max_iterations:
                new_activated_nodes = []
                for v in current_seed:
                    for w in G.successors(v):
                        if (
                            w not in activated_nodes
                            and random.random() < G[v][w]["weight"]
                        ):
                            new_activated_nodes.append(w)
                            activated_nodes.append(w)
                iteration += 1

                current_seed = new_activated_nodes

            total_count += len(activated_nodes)

        average_activation = total_count / iter
        return average_activation

    def subset_select(self, embeddings):
        """
        Borrowed from
        https://github.com/skzhang1/IDEAL/blob/23b1e1cb049004d8296c81933a692b49c42ab527/core_method.py#L172
        Then asked ChatGPT to clean it up
        """
        select_num = self.number_to_select
        k = self.k

        if type(embeddings) == torch.Tensor:
            embeddings = embeddings.detach().cpu().numpy()

        n = len(embeddings)
        graph = nx.DiGraph()

        # Construct the graph with node connections based on cosine similarity
        bar = tqdm(range(n), desc="Constructing graph")
        for i in bar:
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k - 1 : -1]
            sum_weight = cur_scores[sorted_indices].sum()

            for idx in sorted_indices:
                if idx != i:
                    graph.add_edge(i, idx, weight=cur_scores[idx] / sum_weight)

        # Initialize the selection process
        selected_nodes = []
        out_degrees = dict(graph.out_degree())
        initial_point = max(out_degrees, key=out_degrees.get)
        selected_nodes.append(initial_point)
        scores = [0]

        # Select nodes based on influence propagation
        bar = tqdm(range(select_num - 1), desc="Voting nodes")
        for _ in bar:
            max_influence = 0
            best_node = None
            for node in graph.nodes():
                if node not in selected_nodes:
                    selected_nodes.append(node)
                    influence = self.ic_diffusion_model(graph, selected_nodes)
                    selected_nodes.pop()

                    if influence > max_influence:
                        max_influence = influence
                        best_node = node

            if best_node is not None:
                selected_nodes.append(best_node)
                scores.append(max_influence)

        return selected_nodes, scores
