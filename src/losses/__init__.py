from losses.kl_div_loss import KLDivLoss
from losses.mse_loss import MSELoss
from losses.quaild_facility_location_loss import QuaildFacilityLocationLoss
from losses.quaild_graph_cut_loss import QuaildGraphCutLoss
from losses.triplet_loss import TripletLoss


LOSSES_LUT = {
    "mean_squared_error": MSELoss,
    "kl_divergence": KLDivLoss,
    "triplet": TripletLoss,
    "facility_location": QuaildFacilityLocationLoss,
    "graph_cut": QuaildGraphCutLoss,
}
