# Licensed under LGPL 2.1, please see LICENSE for details
# https://www.gnu.org/licenses/lgpl-2.1.html
#
# The work on this project has been performed at the GEG Group at ETH Zurich:
# --> https://geg.ethz.ch
#
# The initial version of this file has been implemented by:
#
#     Philipp Schaedle (https://github.com/philippschaedle)
#     Benjamin M. Adams
#
# Further changes are done by:
#

############################

import numpy as np


class PowerPlantEnergyOutput(object):
    """PowerPlantEnergyOutput."""
    q_recuperator = np.nan
    q_preheater = np.nan
    q_boiler = np.nan
    q_superheater = np.nan
    q_desuperheater = np.nan
    q_condenser = np.nan
    w_turbine = np.nan
    w_pump = np.nan
    #w_desuperheater = np.nan
    #w_condenser = np.nan
    w_pump_cooler = np.nan
    w_fan = np.nan
    w_net = np.nan
    C_turb = np.nan
    C_pump_orc = np.nan
    C_ACC = np.nan
    C_eco = np.nan
    C_eva = np.nan
    C_sh = np.nan
    C_HE = np.nan
    C_rec =np.nan
    C_tot_orc = np.nan
    Specific_cost = np.nan


class PowerPlantOutput(PowerPlantEnergyOutput):
    """PowerPlantOutput."""
    dT_range_PHE = np.nan
    dT_LMTD_preheater = np.nan
    dT_LMTD_boiler = np.nan
    dT_LMTD_superheater = np.nan
    dT_range_ACC = np.nan
    dT_LMTD_desuperheater = np.nan
    dT_LMTD_condenser = np.nan
    dT_LMTD_recuperator = np.nan

    state = np.nan
