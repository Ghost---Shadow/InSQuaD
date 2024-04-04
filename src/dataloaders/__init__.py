from dataloaders.dummy import DummyDataset
from dataloaders.mrpc import MRPC


DATALOADERS_LUT = {
    MRPC.NAME: MRPC,
    DummyDataset.NAME: DummyDataset,
}
