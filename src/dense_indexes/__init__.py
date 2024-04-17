from dense_indexes.no_operation import NoOperation
from dense_indexes.wrapped_faiss import WrappedFaiss


DENSE_INDEXES_LUT = {
    WrappedFaiss.NAME: WrappedFaiss,
    NoOperation.NAME: NoOperation,
}
