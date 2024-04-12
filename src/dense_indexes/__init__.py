from dense_indexes.in_memory import InMemory
from dense_indexes.wrapped_faiss import WrappedFaiss


DENSE_INDEXES_LUT = {
    "faiss": WrappedFaiss,
    "in_memory": InMemory,
}
