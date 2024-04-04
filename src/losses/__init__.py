from losses.kl_div_loss import KLDivLoss
from losses.mse_loss import MSELoss
from losses.triplet_loss import TripletLoss


LOSSES_LUT = {
    "mean_squared_error": MSELoss,
    "kl_divergence": KLDivLoss,
    "triplet": TripletLoss,
}
