from dataloaders.dummy import DummyDataset
from dataloaders.mrpc import MRPC
from dataloaders.dummy_hotpot_qa_with_q_loader import DummyHotpotQaWithQDataset
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from dataloaders.wiki_multihop_qa_with_q_loader import WikiMultihopQaWithQDataset


DATALOADERS_LUT = {
    MRPC.NAME: MRPC,
    DummyDataset.NAME: DummyDataset,
    HotpotQaDataset.NAME: HotpotQaDataset,
    HotpotQaWithQDataset.NAME: HotpotQaWithQDataset,
    DummyHotpotQaWithQDataset.NAME: DummyHotpotQaWithQDataset,
    WikiMultihopQaDataset.NAME: WikiMultihopQaDataset,
    WikiMultihopQaWithQDataset.NAME: WikiMultihopQaWithQDataset,
}
