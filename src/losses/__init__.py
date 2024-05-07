from losses.kl_div_loss import KLDivLoss
from losses.mse_loss import MSELoss
from losses.quaild_facility_location_loss import QuaildFacilityLocationLoss
from losses.quaild_graph_cut_loss import QuaildGraphCutLoss


LOSSES_LUT = {
    KLDivLoss.NAME: KLDivLoss,
    MSELoss.NAME: MSELoss,
    QuaildFacilityLocationLoss.NAME: QuaildFacilityLocationLoss,
    QuaildGraphCutLoss.NAME: QuaildGraphCutLoss,
}
