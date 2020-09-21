import os
import math
import numpy as np

from utils.globalConstants import getProjectRoot
from utils.fluidStates import FluidState
from src.parasiticPowerFractionCoolingTower import parasiticPowerFractionCoolingTower
from src.results import Results
from src.heatExchanger import heatExchanger
from src.heatExchangerOptMdot import heatExchangerOptMdot

class ORCCycleSupercritPboil(object):
    """ ORCCycleSupercritPboil.
    Heat/power is output as specific heat and specific work. To find the
    actual power, multiply by the flowrate of geofluid through the system.
    """
    def __init__(self, T_ambient_C, dT_approach, dT_pinch, eta_pump, eta_turbine, coolingMode, orcFluid, P_boil_Pa = False):

        self.orcFluid = orcFluid
        self.T_ambient_C = T_ambient_C
        self.dT_approach = dT_approach
        self.dT_pinch = dT_pinch
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.coolingMode = coolingMode
        self.P_boil_Pa = P_boil_Pa
        self.filepath = os.path.join(getProjectRoot(), 'data', 'ORC_Pboil_optimum.csv')

    def solve(self, T_in_C):

        if not self.P_boil_Pa:
            data = np.genfromtxt(self.filepath, delimiter=',')
            self.P_boil_Pa = np.interp(T_in_C, data[:,0], data[:,1])
        # Critical point of R245fa
        # if Pboil is below critical, throw error
        if self.P_boil_Pa < FluidState.getPcrit(self.orcFluid):
            raise ValueError('ORC_Cycle_Supercrit_Pboil:lowBoilingPressure - Boiling Pressure Below Critical Pressure')
        # The line of minimum entropy to keep the fluid vapor in turbine is
        # entropy at saturated vapor at 125C. So inlet temp must provide this
        # minimum entropy.
        s_min = FluidState.getSFromTQ(125., 1, self.orcFluid)
        T_min = FluidState.getTFromPS(self.P_boil_Pa, s_min, self.orcFluid)
        if (T_in_C - self.dT_pinch) < T_min:
            raise ValueError('ORC_Cycle_Supercrit_Pboil:lowInletTemp - Inlet Temp below %.1f C for Supercritical Fluid'%(T_min+self.dT_pinch))

        T_condense_C = self.T_ambient_C + self.dT_approach

        # initiate results object
        results = Results('Water')

        # create empty list to compute cycle of 7 states
        state   = [None] * 7

        #State 1 (Condenser -> Pump)
        #saturated liquid
        state[0] = FluidState.getStateFromTQ(T_condense_C, 0, self.orcFluid)

        # State 7 (Desuperheater -> Condenser)
        # saturated vapor
        state[6] = FluidState.getStateFromTQ(state[0].T_C, 1, self.orcFluid)

        # State 2 (Pump -> Recuperator)
        h_2s = FluidState.getHFromPS(self.P_boil_Pa, state[0].S_JK, self.orcFluid)
        h2 = state[0].h_Jkg - ((state[0].h_Jkg - h_2s) / self.eta_pump)
        state[1] = FluidState.getStateFromPh(self.P_boil_Pa, h2, self.orcFluid)

        # water (assume pressure 100 kPa above saturation)
        P_water = FluidState.getPFromTQ(T_in_C, 0, 'Water') + 100e3

        # Guess orc_in fluid is state[1].T_C
        state[2] = FluidState.getStateFromPT(state[1].P_Pa, state[1].T_C, self.orcFluid)
        # Water in temp is T_in_C
        T_C_11 = T_in_C
        P_4 = state[1].P_Pa

        dT = 1
        while abs(dT) >= 1:
            # State 4 (Boiler -> Turbine)
            # Input orc/geo heat exchanger
            opt_heatExchanger_results = heatExchangerOptMdot(state[2].T_C, P_4, self.orcFluid, T_C_11, P_water, 'Water', self.dT_pinch, T_min)
            state[3] = FluidState.getStateFromPT(P_4, opt_heatExchanger_results.T_1_out, self.orcFluid)

            #State 5 (Turbine -> Recuperator)
            h_5s = FluidState.getHFromPS(state[0].P_Pa, state[3].S_JK, self.orcFluid)
            h_5 = state[3].h_Jkg - self.eta_turbine * (state[3].h_Jkg - h_5s)
            state[4] =  FluidState.getStateFromPh(state[0].P_Pa, h_5, self.orcFluid)

            # State 3 (Recuperator -> Boiler)
            # State 6 (Recuperator -> Desuperheater)
            # Assume m_dot for each fluid is 1, then output is specific heat
            # exchange
            heatExchanger_results = heatExchanger(state[1].T_C, state[1].P_Pa, 1, self.orcFluid,
                                                  state[4].T_C, state[0].P_Pa, 1, self.orcFluid, self.dT_pinch)

            state[2] = FluidState.getStateFromPT(state[2].P_Pa, state[2].T_C, self.orcFluid)
            state[5] = FluidState.getStateFromPT(state[0].P_Pa, heatExchanger_results.T_2_out, self.orcFluid)

            dT = state[2].T_C - heatExchanger_results.T_1_out
            state[2].T_C = heatExchanger_results.T_1_out

        #Calculate orc heat/work
        w_pump_orc = state[0].h_Jkg - state[1].h_Jkg
        q_boiler_orc = -1 * (state[2].h_Jkg - state[3].h_Jkg)
        w_turbine_orc = state[3].h_Jkg - state[4].h_Jkg
        q_desuperheater_orc = -1 * (state[5].h_Jkg - state[6].h_Jkg)
        q_condenser_orc = -1 * (state[6].h_Jkg - state[0].h_Jkg)

        # Cooling Tower Parasitic load
        results.dT_range_CT = state[5].T_C - state[6].T_C
        parasiticPowerFraction = parasiticPowerFractionCoolingTower(self.T_ambient_C, self.dT_approach, results.dT_range_CT, self.coolingMode)
        w_cooler_orc = q_desuperheater_orc * parasiticPowerFraction('cooling')
        w_condenser_orc = q_condenser_orc * parasiticPowerFraction('condensing')

        #Calculate water heat/work
        results.w_pump          = opt_heatExchanger_results.mdot_ratio * w_pump_orc
        results.q_boiler        = opt_heatExchanger_results.mdot_ratio * q_boiler_orc
        results.w_turbine       = opt_heatExchanger_results.mdot_ratio * w_turbine_orc
        results.q_recuperator   = opt_heatExchanger_results.mdot_ratio * heatExchanger_results.Q_exchanged
        results.q_desuperheater = opt_heatExchanger_results.mdot_ratio * q_desuperheater_orc
        results.q_condenser     = opt_heatExchanger_results.mdot_ratio * q_condenser_orc
        results.w_cooler        = opt_heatExchanger_results.mdot_ratio * w_cooler_orc
        results.w_condenser     = opt_heatExchanger_results.mdot_ratio * w_condenser_orc

        results.w_net = results.w_turbine + results.w_pump + results.w_cooler + results.w_condenser

        results.end_T_C = opt_heatExchanger_results.T_2_out
        results.dT_LMTD_boiler = opt_heatExchanger_results.dT_LMTD
        results.dT_LMTD_recuperator = heatExchanger_results.dT_LMTD

        return results