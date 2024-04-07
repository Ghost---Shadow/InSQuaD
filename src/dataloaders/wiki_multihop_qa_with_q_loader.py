from dataloaders.base import BaseDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from datasets import load_dataset


class WikiMultihopQaWithQDataset(BaseDataset):
    NAME = "wiki_multihop_qa_with_q"

    def __init__(self, config):
        super().__init__(config)
        self.dataset = load_dataset(
            "scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35"
        )

    @staticmethod
    def collate_fn(batch):
        upstream_batch = WikiMultihopQaDataset.collate_fn(batch)
        downstream_batch = HotpotQaWithQDataset.downstream_collate(
            batch, upstream_batch
        )

        return {
            **upstream_batch,
            **downstream_batch,
        }
