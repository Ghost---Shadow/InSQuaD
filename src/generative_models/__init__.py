from generative_models.no_operation import NoOp
from generative_models.wrapped_automodel import WrappedAutoModel
from generative_models.wrapped_t5 import WrappedT5


GENERATIVE_MODELS_LUT = {
    NoOp.NAME: NoOp,
    WrappedAutoModel.NAME: WrappedAutoModel,
    WrappedT5.NAME: WrappedT5,
}