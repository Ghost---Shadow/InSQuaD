from semantic_search_models.no_operation import NoOp
from semantic_search_models.wrapped_mpnet import WrappedMpnetModel


SEMANTIC_SEARCH_MODELS_LUT = {
    NoOp.NAME: NoOp,
    WrappedMpnetModel.NAME: WrappedMpnetModel,
}
