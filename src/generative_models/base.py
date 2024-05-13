class BaseGenerativeModel:
    def _compute_rouge(self, target, predicted):
        scores = self.rouge_scorer.score(target, predicted)
        rouge_result = {
            metric: {
                "precision": score.precision,
                "recall": score.recall,
                "fmeasure": score.fmeasure,
            }
            for metric, score in scores.items()
        }
        return rouge_result
