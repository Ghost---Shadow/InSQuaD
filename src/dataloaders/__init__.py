from dataloaders.dbpedia import DBPedia
from dataloaders.dummy import DummyDataset
from dataloaders.dummy_hotpot_qa_with_q_loader import DummyHotpotQaWithQDataset
from dataloaders.hellaswag import Hellaswag
from dataloaders.hotpot_qa_loader import HotpotQaDataset
from dataloaders.hotpot_qa_with_q_loader import HotpotQaWithQDataset
from dataloaders.mnli import MNLI
from dataloaders.mrpc import MRPC
from dataloaders.rte import RTE
from dataloaders.sst2 import SST2
from dataloaders.sst5 import SST5
from dataloaders.wiki_multihop_qa_loader import WikiMultihopQaDataset
from dataloaders.wiki_multihop_qa_with_q_loader import WikiMultihopQaWithQDataset
from dataloaders.xsum import XsumDataset


DATALOADERS_LUT = {
    DBPedia.NAME: DBPedia,
    DummyDataset.NAME: DummyDataset,
    DummyHotpotQaWithQDataset.NAME: DummyHotpotQaWithQDataset,
    Hellaswag.NAME: Hellaswag,
    HotpotQaDataset.NAME: HotpotQaDataset,
    HotpotQaWithQDataset.NAME: HotpotQaWithQDataset,
    MNLI.NAME: MNLI,
    MRPC.NAME: MRPC,
    RTE.NAME: RTE,
    SST2.NAME: SST2,
    SST5.NAME: SST5,
    WikiMultihopQaDataset.NAME: WikiMultihopQaDataset,
    WikiMultihopQaWithQDataset.NAME: WikiMultihopQaWithQDataset,
    XsumDataset.NAME: XsumDataset,
}
