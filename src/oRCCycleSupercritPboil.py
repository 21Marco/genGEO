import os, math
import numpy as np

from src.parasiticPowerFractionCoolingTower import parasiticPowerFractionCoolingTower
from src.heatExchanger import heatExchanger
from src.heatExchangerOptMdot import heatExchangerOptMdot
from src.powerPlantOutput import PowerPlantOutput

from utils.constantsAndPaths import getPboilOptimum
from utils.fluidStates import FluidState
from utils.fluidStateFromTQ import FluidStateFromTQ
from utils.fluidStateFromPh import FluidStateFromPh
from utils.fluidStateFromPT import FluidStateFromPT

class ORCCycleSupercritPboil(object):
    """ ORCCycleSupercritPboil.
    Heat/power is output as specific heat and specific work. To find the
    actual power, multiply by the flowrate of geofluid through the system.
    """
    def __init__(self, params):

        self.params = params
        self.data = np.genfromtxt(getPboilOptimum(), delimiter=',')

    def solve(self, initialState, P_boil_Pa = False):

        T_in_C = initialState.T_C()

        if not P_boil_Pa:
            P_boil_Pa = np.interp(T_in_C, self.data[:,0], self.data[:,1])
        # Critical point of R245fa
        # if Pboil is below critical, throw error
        if P_boil_Pa < FluidState.getPcrit(self.params.orc_fluid):
            raise ValueError('GenGeo::ORC_Cycle_Supercrit_Pboil:lowBoilingPressure - Boiling Pressure Below Critical Pressure')
        # The line of minimum entropy to keep the fluid vapor in turbine is
        # entropy at saturated vapor at 125C. So inlet temp must provide this
        # minimum entropy.
        s_min = FluidState.getSFromTQ(125., 1, self.params.orc_fluid)
        T_min = FluidState.getTFromPS(P_boil_Pa, s_min, self.params.orc_fluid)
        if (T_in_C - self.params.dT_pinch) < T_min:
            raise ValueError('GenGeo::ORC_Cycle_Supercrit_Pboil:lowInletTemp - Inlet Temp below %.1f C for Supercritical Fluid'%(T_min+self.params.dT_pinch))

        T_condense_C = self.params.T_ambient_C + self.params.dT_approach

        # create empty list to compute cycle of 7 states
        state   = [None] * 7

        #State 1 (Condenser -> Pump)
        #saturated liquid
        state[0] = FluidStateFromTQ(T_condense_C, 0, self.params.orc_fluid)

        # State 7 (Desuperheater -> Condenser)
        # saturated vapor
        state[6] = FluidStateFromTQ(state[0].T_C(), 1, self.params.orc_fluid)

        # State 2 (Pump -> Recuperator)
        h_2s = FluidState.getHFromPS(P_boil_Pa, state[0].S_JK(), self.params.orc_fluid)
        h2 = state[0].h_Jkg() - ((state[0].h_Jkg() - h_2s) / self.params.eta_pump_orc)
        state[1] = FluidStateFromPh(P_boil_Pa, h2, self.params.orc_fluid)

        # water (assume pressure 100 kPa above saturation)
        P_water = FluidState.getPFromTQ(T_in_C, 0, 'Water') + 100e3

        # Guess orc_in fluid is state[1].T_C
        state[2] = FluidStateFromPT(state[1].P_Pa(), state[1].T_C(), self.params.orc_fluid)
        # Water in temp is T_in_C
        T_C_11 = T_in_C
        P_4 = state[1].P_Pa()

        # initialize state 2 with pressure state 2 and dummy temperature
        state[3] = FluidStateFromPT(P_4, 15., self.params.orc_fluid)

        # initialize state 3 with pressure state 1 and dummy enthalpy
        state[4] =  FluidStateFromPh(state[0].P_Pa(), 1e4, self.params.orc_fluid)

        # initialize state 6 with pressure state 1 and dummy temperature
        state[5] = FluidStateFromPT(state[0].P_Pa(), 15., self.params.orc_fluid)

        dT = 1
        while abs(dT) >= 1:
            # State 4 (Boiler -> Turbine)
            # Input orc/geo heat exchanger
            self.opt_heatExchanger_results = heatExchangerOptMdot(state[2].T_C(), P_4, self.params.orc_fluid, T_C_11, P_water, 'Water', self.params.dT_pinch, T_min)
            # state[3] = FluidStateFromPT(P_4, opt_heatExchanger_results.T_1_out, self.params.orc_fluid)
            state[3].T_C_in = self.opt_heatExchanger_results.T_1_out

            #State 5 (Turbine -> Recuperator)
            h_5s = FluidState.getHFromPS(state[0].P_Pa(), state[3].S_JK(), self.params.orc_fluid)
            h_5 = state[3].h_Jkg() - self.params.eta_turbine_orc * (state[3].h_Jkg() - h_5s)
            state[4].h_Jkg_in = h_5
            # state[4] =  FluidStateFromPh(state[0].P_Pa(), h_5, self.params.orc_fluid)

            # State 3 (Recuperator -> Boiler)
            # State 6 (Recuperator -> Desuperheater)
            # Assume m_dot for each fluid is 1, then output is specific heat
            # exchange
            self.heatExchanger_results = heatExchanger(state[1].T_C(), state[1].P_Pa(), 1, self.params.orc_fluid,
                                                  state[4].T_C(), state[0].P_Pa(), 1, self.params.orc_fluid, self.params.dT_pinch)

            # state[2] = FluidStateFromPT(state[2].P_Pa(), state[2].T_C(), self.params.orc_fluid)
            # state[5] = FluidStateFromPT(state[0].P_Pa(), heatExchanger_results.T_2_out, self.params.orc_fluid)
            state[5].T_C_in = self.heatExchanger_results.T_2_out

            dT = state[2].T_C() - self.heatExchanger_results.T_1_out
            state[2].T_C_in = self.heatExchanger_results.T_1_out

        # pass states to self variable
        self.state = state

        # return temperature
        self.state_out = FluidStateFromPT(initialState.P_Pa(), self.opt_heatExchanger_results.T_2_out, self.params.working_fluid)

        return self.state_out

    def gatherOutput(self):

        output = PowerPlantOutput()

        output.state_out = self.state_out

        #Calculate orc heat/work
        w_pump_orc = self.state[0].h_Jkg() - self.state[1].h_Jkg()
        q_boiler_orc = -1 * (self.state[2].h_Jkg() - self.state[3].h_Jkg())
        w_turbine_orc = self.state[3].h_Jkg() - self.state[4].h_Jkg()
        q_desuperheater_orc = -1 * (self.state[5].h_Jkg() - self.state[6].h_Jkg())
        q_condenser_orc = -1 * (self.state[6].h_Jkg() - self.state[0].h_Jkg())

        # Cooling Tower Parasitic load
        output.dT_range_CT = self.state[5].T_C() - self.state[6].T_C()
        parasiticPowerFraction = parasiticPowerFractionCoolingTower(self.params.T_ambient_C, self.params.dT_approach, output.dT_range_CT, self.params.cooling_mode)
        w_cooler_orc = q_desuperheater_orc * parasiticPowerFraction('cooling')
        w_condenser_orc = q_condenser_orc * parasiticPowerFraction('condensing')

        #Calculate water heat/work
        output.w_pump          = self.opt_heatExchanger_results.mdot_ratio * w_pump_orc
        output.q_boiler        = self.opt_heatExchanger_results.mdot_ratio * q_boiler_orc
        output.w_turbine       = self.opt_heatExchanger_results.mdot_ratio * w_turbine_orc
        output.q_recuperator   = self.opt_heatExchanger_results.mdot_ratio * self.heatExchanger_results.Q_exchanged
        output.q_desuperheater = self.opt_heatExchanger_results.mdot_ratio * q_desuperheater_orc
        output.q_condenser     = self.opt_heatExchanger_results.mdot_ratio * q_condenser_orc
        output.w_cooler        = self.opt_heatExchanger_results.mdot_ratio * w_cooler_orc
        output.w_condenser     = self.opt_heatExchanger_results.mdot_ratio * w_condenser_orc

        output.w_net = output.w_turbine + output.w_pump + output.w_cooler + output.w_condenser

        output.end_T_C = self.opt_heatExchanger_results.T_2_out
        output.dT_LMTD_boiler = self.opt_heatExchanger_results.dT_LMTD
        output.dT_LMTD_recuperator = self.heatExchanger_results.dT_LMTD

        return output
