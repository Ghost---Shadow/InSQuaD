from generative_models.wrapped_t5 import WrappedT5
from generative_models.no_operation import NoOp

GENERATIVE_MODELS_LUT = {
    "t5": WrappedT5,
    "noop": NoOp,
}
