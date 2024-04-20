from dense_indexes.in_memory import InMemory
from dense_indexes.no_operation import NoOperation
from dense_indexes.wrapped_faiss import WrappedFaiss


DENSE_INDEXES_LUT = {
    InMemory.NAME: InMemory,
    NoOperation.NAME: NoOperation,
    WrappedFaiss.NAME: WrappedFaiss,
}
