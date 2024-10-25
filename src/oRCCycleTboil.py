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
import os, math
import numpy as np
import matplotlib as plt

from src.coolingCondensingTower import CoolingCondensingTower
from src.powerPlantOutput import PowerPlantOutput
from src.plotDiagrams import TsDischarge
from src.plotDiagrams import PlotTQHX

from utils.constantsAndPaths import getTboilOptimum
from utils.fluidState import FluidState
from utils.maxSubcritORCBoilTemp import maxSubcritORCBoilTemp
from models.simulationParameters import SimulationParameters

class ORCCycleTboil(object):
    """ ORCCycleTboil.
    Heat/power is output as specific heat and specific work. To find the
    actual power, multiply by the flowrate of geofluid through the system.
    """

    def __init__(self, params = None, **kwargs):
        self.params = params
        if self.params == None:
            self.params = SimulationParameters(**kwargs)
        self.data = getTboilOptimum()
        self.orc_fluid = self.params.orc_fluid
        self.T_boil_max = maxSubcritORCBoilTemp(self.orc_fluid)

    def update_properties(self, index):
        """Aggiorna temperatura, pressione ed entalpia in base allo stato definito."""
        if self.state[index] is not None:
            self.T[index] = self.state[index].T_C  # Temperatura in °C
            self.p[index] = self.state[index].P_Pa  # Pressione in Pa
            self.h[index] = self.state[index].h_Jkg  # Entalpia in J/kg
            self.s[index] = self.state[index].s_JK

    def solve(self, initialState, T_boil_C = False, dT_pinch = False):

        T_in_C = initialState.T_C

        if not T_boil_C:
            T_boil_C = np.interp(T_in_C, self.data[self.params.opt_mode][self.params.orc_fluid][:,0], self.data[self.params.opt_mode][self.params.orc_fluid][:,1])

        if not dT_pinch:
            dT_pinch = np.interp(T_in_C, self.data[self.params.opt_mode][self.params.orc_fluid][:,0], self.data[self.params.opt_mode][self.params.orc_fluid][:,2])

        # run some checks  if T_in_C and T_boil_C are valid
        if np.isnan(T_in_C):
            raise Exception('GenGeo::ORCCycleTboil:T_in_NaN - ORC input temperature is NaN!')

        if np.isnan(T_boil_C):
            raise Exception('GenGeo::ORCCycleTboil:T_boil_NaN - ORC boil temperature is NaN!')

        if T_boil_C > FluidState.getTcrit(self.params.orc_fluid):
            raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature above critical point')

        if dT_pinch <= 0:
            raise Exception('GenGeo::ORCCycleTboil:dT_pinch_Negative - dT_pinch is negative!')

        if T_in_C < T_boil_C + dT_pinch:
            raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature of %s is greater than input temp of %s less pinch dT of %s.'%(T_boil_C, T_in_C, dT_pinch))

        # only refresh T_boil_max if orc_fluid has changed from initial
        if self.params.orc_fluid != self.orc_fluid:
            self.T_boil_max = maxSubcritORCBoilTemp(self.params.orc_fluid)
            self.orc_fluid = self.params.orc_fluid
        if T_boil_C > self.T_boil_max:
            raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature of %s is greater than maximum allowed of %s.'%(T_boil_C, self.T_boil_max))

        T_condense_C = self.params.T_ambient_C + self.params.dT_approach

        #Creation of a list for the 10 tdn points
        self.state = [None] * 10
        self.p = np.zeros(10)
        self.T = np.zeros(10)
        self.h = np.zeros(10)
        self.s = np.zeros(10)

        out_cond, out_pump, out_rec_cold, out_eco_preheat, out_eco_subcool, out_eva, out_sh, out_turb, out_rec_hot, out_desh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        in_pump, in_rec_cold, in_eco, in_eva_preheat, in_eva_subcool, in_sh, in_turb, in_rec_hot, in_desh, in_cond = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        #State 1 (Condenser -> Pump)
        #saturated liquid
        self.state[out_cond] = FluidState.getStateFromTQ(T_condense_C, 0, self.params.orc_fluid)
        self.update_properties(out_cond)

        #State 6 (Boiler -> Superheater)
        #saturated vapor
        self.state[out_eva] = FluidState.getStateFromTQ(T_boil_C, 1, self.params.orc_fluid)
        self.update_properties(out_eva)

        #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
        #Condenser
        if self.params.dp_dT_loss['loss_cond'] > 0:
            self.p[out_desh] = self.state[out_cond].P_Pa / (1 - self.params.dp_dT_loss['loss_cond'])  # dp for the condenser
        else:
            self.T[out_desh] = self.state[out_cond].T_C + self.params.dp_dT_loss['loss_cond']
            self.p[out_desh] = FluidState.getStateFromTQ(self.T[out_desh], 1, self.params.orc_fluid).P_Pa

        #Desuperheater
        if self.params.dp_dT_loss['loss_desh'] > 0:
            self.p[in_desh] = self.p[out_desh] / (1 - self.params.dp_dT_loss['loss_desh'])  # dp for the desuperheater
        else:
            self.p[in_desh] = self.params.dp_dT_loss['loss_desh'] + self.p[out_desh]

        #Boiler/Evaporator
        if self.params.dp_dT_loss['loss_eva'] > 0:
            self.p[out_eco_subcool] = self.state[out_eco_subcool].P_Pa / (1 - self.params.dp_dT_loss['loss_eva'])  # dp for the boiler
        else:
            self.T[out_eco_subcool] = self.state[out_eva].T_C + self.params.dp_dT_loss['loss_eva']  # - dT_sc
            self.p[out_eco_subcool] = FluidState.getStateFromTQ(self.T[out_eco_subcool], 0, self.params.orc_fluid).P_Pa

        #Preheater/Economizer
        if self.params.dp_dT_loss['loss_eco'] > 0:
            self.p[in_eco] = self.p[out_eco_subcool] / (1 - self.params.dp_dT_loss['loss_eco'])  # dp for the pre-heater
        else:
            self.p[in_eco] = self.params.dp_dT_loss['loss_eco'] + self.p[out_eco_subcool]

        #State 10 (Desuperheater -> Condenser)
        #saturated vapor
        self.state[out_desh] = FluidState.getStateFromPQ(self.p[out_desh], 1, self.params.orc_fluid)
        self.update_properties(out_desh)

        #State 5 (Preheater -> Boiler)
        #saturated liquid
        self.state[out_eco_subcool] = FluidState.getStateFromPQ(self.p[out_eco_subcool], 0, self.params.orc_fluid)
        self.update_properties(out_eco_subcool)

        #State 2 (Pump -> Recuperator)
        h_out_pump_s = FluidState.getStateFromPS(self.p[in_eco], self.state[out_cond].s_JK, self.params.orc_fluid).h_Jkg
        self.h[out_pump] = self.state[out_cond].h_Jkg - ((self.state[out_cond].h_Jkg - h_out_pump_s) / self.params.eta_pump_orc)
        self.state[out_pump] = FluidState.getStateFromPh(self.p[in_eco], self.h[out_pump], self.params.orc_fluid)
        self.update_properties(out_pump)

        if self.params.orc_Saturated:  # Saturated ORC Cycle (without SH)
            if self.params.orc_no_Rec:  # without Recuperator

                # #Compute the TDN points
                #State 7 = 6 (Boiler -> Turbine)
                self.state[out_sh] = self.state[out_eva]
                self.update_properties(out_sh)

                #State 3 = 2 (Pump -> Preheater)
                self.state[in_eco] = self.state[out_pump]
                self.update_properties(in_eco)

                #State 9 = 8 (Turbine -> Desuperheater)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_desh], self.state[out_eva].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_eva].h_Jkg - self.params.eta_turbine_orc * (self.state[out_eva].h_Jkg - h_out_turb_s)
                self.state[in_desh] = FluidState.getStateFromPh(self.p[in_desh], self.h[out_turb], self.params.orc_fluid)
                self.state[out_turb] = self.state[in_desh]
                self.update_properties(out_turb)
                self.update_properties(in_desh)

            else:  # with the Recuperator

                #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                #Recuperator hot side
                if self.params.dp_dT_loss['loss_rec_hot'] > 0:
                    self.p[out_turb] = self.p[out_rec_hot] / (1 - self.params.dp_dT_loss['loss_rec_hot'])  # dp for the hot side recuperator
                else:
                    self.p[out_turb] = self.params.dp_dT_loss['loss_rec_hot'] + self.p[out_rec_hot]

                #Recuperator cold side
                if self.params.dp_dT_loss['loss_rec_cold'] > 0:
                    self.p[out_pump] = self.p[out_rec_cold] * (1 - self.params.dp_dT_loss['loss_rec_cold'])  # dp for the recuperator cold side
                else:
                    self.p[out_pump] = self.p[out_rec_cold] - self.params.dp_dT_loss['loss_rec_cold']

                # #Compute the TDN points
                #State 7 = 6 (Turbine -> Recuperator hot side)
                self.state[out_sh] = self.state[out_eva]
                self.update_properties(out_sh)
                self.update_properties(out_eva)

                #State 3 (Recuperator cold side -> Preheater)
                self.T[out_rec_cold] = self.state[out_pump].T_C + self.params.dT_pp_rec
                self.state[out_rec_cold] = FluidState.getStateFromPT(self.p[out_rec_cold], self.T[out_rec_cold], self.params.orc_fluid)
                self.update_properties(out_rec_cold)

                #State 8 (Turbine -> Recuperator)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_rec_cold], self.state[in_turb].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[in_turb].h_Jkg - self.params.eta_turbine_orc * (self.state[in_turb].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[out_turb], self.h[out_turb], self.params.orc_fluid)
                self.update_properties(out_turb)

                #State 9 (Recuperator hot -> Desuperheater)
                self.h[out_rec_hot] = self.h[out_turb] - self.state[out_rec_cold].h_Jkg + self.h[out_pump]
                self.state[out_rec_hot] = FluidState.getStateFromPh(self.p[out_rec_hot], self.h[out_rec_hot], self.params.orc_fluid)
                self.update_properties(out_rec_hot)

        else:  # ORC Cycle with SH
            if self.params.orc_no_Rec:  # without Recuperator

                #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                #Superheater
                if self.params.dp_dT_loss['loss_sh'] > 0:
                    self.p[out_sh] = self.state[out_eva].P_Pa * (1 - self.params.dp_dT_loss['loss_sh'])  # dp for the superheater
                else:
                    self.p[out_sh] = self.state[out_eva].P_Pa - self.params.dp_dT_loss['loss_sh']

                # #Compute TDN points
                #State 7 (Superheater -> Turbine)
                self.T[out_sh] = T_in_C - self.params.dT_ap_phe    #T_boil_C (T_5) + self.params.dT_sh_phe
                if T_in_C - self.params.dT_ap_phe < T_boil_C:
                    raise ValueError('GenGeo::ORCCycleTboil:Input temperature after approach difference is below boiling temperature')
                if T_boil_C + self.params.dT_sh_phe > T_in_C:
                    raise ValueError('GenGeo::ORCCycleTboil:Boiling temperature plus superheating approach difference exceeds input temperature')

                self.state[out_sh] = FluidState.getStateFromPT(self.p[out_sh], self.T[out_sh], self.params.orc_fluid)
                self.update_properties(out_sh)

                #State 9 = 8 (Turbine -> Desuperheater)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_desh], self.state[out_sh].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_sh].h_Jkg - self.params.eta_turbine_orc * (self.state[out_sh].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[in_desh], self.h[out_turb], self.params.orc_fluid)
                self.state[out_rec_hot] = self.state[out_turb]
                self.update_properties(out_rec_hot)
                self.update_properties(out_turb)

                # #State 3 = 2 (Pump -> Preheater)
                self.state[in_eco] = self.state[out_pump]
                self.update_properties(in_eco)

            else:  # ciclo orc SH con rec

                #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                #Superheater
                if self.params.dp_dT_loss['loss_sh'] > 0:
                    self.p[out_sh] = self.state[out_eva].P_Pa * (1 - self.params.dp_dT_loss['loss_sh'])  # dp for the superheater
                else:
                    self.p[out_sh] = self.state[out_eva].P_Pa - self.params.dp_dT_loss['loss_sh']

                #Recuperator hot side
                if self.params.dp_dT_loss['loss_rec_hot'] > 0:
                    self.p[out_turb] = self.p[out_rec_hot] / (1 - self.params.dp_dT_loss['loss_rec_hot'])  # dp for the hot side recuperator
                else:
                    self.p[out_turb] = self.params.dp_dT_loss['loss_rec_hot'] + self.p[out_rec_hot]

                #Recuperator cold side
                if self.params.dp_dT_loss['loss_rec_cold'] > 0:
                    self.p[out_pump] = self.p[out_rec_cold] * (1 - self.params.dp_dT_loss['loss_rec_cold'])  # dp for the cold side recuperator
                else:
                    self.p[out_pump] = self.p[out_rec_cold] - self.params.dp_dT_loss['loss_rec_cold']

                # #Compute TDN points
                #State 7 (Superheater -> Turbine)
                self.T[out_sh] = T_in_C - self.params.dT_ap_phe   #T_boil_C + self.params.dT_sh_phe
                if T_in_C - self.params.dT_ap_phe < T_boil_C:
                    raise ValueError('GenGeo::ORCCycleTboil:Input temperature after approach difference is below boiling temperature')
                if T_boil_C + self.params.dT_sh_phe > T_in_C:
                    raise ValueError('GenGeo::ORCCycleTboil:Boiling temperature plus superheating approach difference exceeds input temperature.')
                self.state[out_sh] = FluidState.getStateFromPT(self.p[out_sh], self.T[out_sh], self.params.orc_fluid)
                self.update_properties(out_sh)

                #State 8 (Turbine -> Recuperator)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_rec_hot], self.state[out_sh].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_sh].h_Jkg - self.params.eta_turbine_orc * (self.state[out_sh].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[out_turb], self.h[out_turb], self.params.orc_fluid)
                self.update_properties(out_turb)

                #State 3 (Recuperator cold -> Preheater)
                self.T[out_rec_cold] = self.state[out_pump].T_C + self.params.dT_pp_rec
                self.state[out_rec_cold] = FluidState.getStateFromPT(self.p[out_rec_cold], self.T[out_rec_cold], self.params.orc_fluid)
                self.update_properties(out_rec_cold)

                #State 9 (Recuperator hot -> Desuperheater)
                self.h[out_rec_hot] = self.h[out_turb] - self.state[out_rec_cold].h_Jkg + self.h[out_pump]
                self.state[out_rec_hot] = FluidState.getStateFromPh(self.p[out_rec_hot], self.h[out_rec_hot], self.params.orc_fluid)
                self.update_properties(out_rec_hot)

        # State 4 (Sub-cooling)
        self.T[out_eco_preheat] = self.T[out_eco_subcool] - self.params.dT_sc_phe
        self.state[out_eco_preheat] = FluidState.getStateFromPT(self.p[out_rec_cold], self.T[out_eco_preheat], self.params.orc_fluid)
        self.update_properties(out_eco_preheat)

        #Chiamata a TsDischarge
        TsDischarge(self.state, self.T, self.params.orc_fluid, self.params.orc_no_Rec)

        results = PowerPlantOutput()

        #Calculate orc heat/work
        w_pump_orc = self.state[out_cond].h_Jkg - self.state[out_pump].h_Jkg
        q_recuperator_orc = -1 * (self.state[out_pump].h_Jkg - self.state[out_rec_cold].h_Jkg)
        q_preheater_orc = -1 * (self.state[out_rec_cold].h_Jkg - self.state[out_eco_preheat].h_Jkg)
        q_boiler_orc = -1 * (self.state[out_eco_preheat].h_Jkg - self.state[out_eva].h_Jkg)
        q_superheater_orc = -1 * (self.state[out_eva].h_Jkg - self.state[out_sh].h_Jkg)
        w_turbine_orc = self.state[out_sh].h_Jkg - self.state[out_turb].h_Jkg
        q_desuperheater_orc = -1 * (self.state[out_rec_hot].h_Jkg - self.state[out_desh].h_Jkg)
        q_condenser_orc = -1 * (self.state[out_desh].h_Jkg - self.state[out_cond].h_Jkg)

        results.dP_pump_orc = self.state[out_pump].P_Pa - self.state[out_cond].P_Pa
        results.P_boil = self.state[out_eco_subcool].P_Pa

        #Points for the recuperator
        n_points = 10
        h_array_rec_hot = np.linspace(self.state[out_rec_hot].h_Jkg, self.state[in_rec_hot].h_Jkg, n_points)  # entalpia lato caldo
        h_array_rec_cold = np.linspace(self.state[in_rec_cold].h_Jkg, self.state[out_rec_cold].h_Jkg, n_points)  # entalpia lato freddo
        p_array_rec_hot = np.linspace(self.state[out_rec_hot].P_Pa, self.state[in_rec_hot].P_Pa, n_points)  # pressione lato caldo
        p_array_rec_cold = np.linspace(self.state[in_rec_cold].P_Pa, self.state[out_rec_cold].P_Pa, n_points)  # pressione lato freddo
        T_array_rec_hot = np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_rec_hot, h_array_rec_hot)]) #dalla coppia del primo elemento p e h, ottiene la T
        T_array_rec_cold = np.array([FluidState.getStateFromPh(p2, h2, self.params.orc_fluid).T_C for p2, h2 in zip(p_array_rec_cold, h_array_rec_cold)])

        # Cooling Tower Parasitic load
        dT_range = self.T[out_rec_hot] - self.T[out_desh]
        parasiticPowerFraction = CoolingCondensingTower.parasiticPowerFraction(self.params.T_ambient_C, self.params.dT_approach, dT_range, self.params.cooling_mode)
        w_cooler_orc = q_desuperheater_orc * parasiticPowerFraction('cooling')
        w_condenser_orc = q_condenser_orc * parasiticPowerFraction('condensing')

        # water (assume pressure 100 kPa above saturation)
        P_sat = FluidState.getStateFromTQ(T_in_C, 0, 'Water').P_Pa  #al posto di water, self.params.working_fuid
        cp = FluidState.getStateFromPT(P_sat + 100e3, T_in_C, 'Water').cp_JK  # al posto di P_sat + 100e3 mettere la P_out_well_geothermal_fluid, initialState.P_Pa
        # Water state a, inlet, b, mid, c, mid, d, exit
        T_a = T_in_C
        T_c = self.T[out_eco_subcool] + dT_pinch
        # mdot_ratio = mdot_orc / mdot_water
        mdot_ratio = cp * (T_a - T_c) / (q_boiler_orc + q_superheater_orc)
        T_d = T_c - mdot_ratio * q_preheater_orc / cp  #injection temperature
        #T_d = T_c - mdot_ratio / cp * (self.h[out_eco_preheat] - self.h[out_rec_cold])

        # check that T_d isn't below pinch constraint
        if T_d < (self.T[out_rec_cold] + dT_pinch):
            # pinch constraint is here, not at c
            # outlet is pump temp plus pinch
            T_d = self.T[out_rec_cold] + dT_pinch
            R = q_boiler_orc / (q_boiler_orc + q_preheater_orc)
            T_c = T_a - (T_a - T_d) * R
            mdot_ratio = cp * (T_a - T_c) / (q_boiler_orc + q_superheater_orc)

        #Calculate T_b, BE SH
        T_b = T_a - mdot_ratio * q_superheater_orc / cp

        # Calculate water heat/work
        results.q_preheater = mdot_ratio * q_preheater_orc
        results.q_boiler = mdot_ratio * q_boiler_orc
        results.q_superheater = mdot_ratio * q_superheater_orc
        results.q_desuperheater = mdot_ratio * q_desuperheater_orc
        results.q_condenser = mdot_ratio * q_condenser_orc
        results.q_recuperator = mdot_ratio * q_recuperator_orc
        results.w_turbine = mdot_ratio * w_turbine_orc
        results.w_pump = mdot_ratio * w_pump_orc
        results.w_cooler = mdot_ratio * w_cooler_orc
        results.w_condenser = mdot_ratio * w_condenser_orc
        results.w_net = results.w_turbine + results.w_pump + results.w_cooler + results.w_condenser

        # Calculate temperatures
        results.dT_range_CT = self.T[out_rec_hot] - self.T[out_desh]
        dT_A_p = T_d - self.T[out_rec_cold]
        dT_B_p = T_c - self.T[out_eco_subcool]
        if dT_A_p == dT_B_p:
            results.dT_LMTD_preheater = dT_A_p
        else:
            div = dT_A_p / dT_B_p
            results.dT_LMTD_preheater = (dT_A_p - dT_B_p) / (math.log(abs(div)) * np.sign(div))

        dT_A_b = T_c - self.T[out_eco_subcool]
        dT_B_b = T_b - self.T[out_eva]
        if dT_A_b == dT_B_b:
            results.dT_LMTD_boiler = dT_A_b
        else:
            div = dT_A_b / dT_B_b
            results.dT_LMTD_boiler = (dT_A_b - dT_B_b) / (math.log(abs(div)) * np.sign(div))

        dT_A_b = T_b - self.T[out_eva]
        dT_B_b = T_a - self.T[out_sh]
        if dT_A_b == dT_B_b:
            results.dT_LMTD_superheater = dT_A_b
        else:
            div = dT_A_b / dT_B_b
            results.dT_LMTD_superheater = (dT_A_b - dT_B_b) / (math.log(abs(div)) * np.sign(div))

        # return temperature
        results.state = FluidState.getStateFromPT(initialState.P_Pa, T_d, self.params.working_fluid)

        #Points for the PHE
        #Evaporator/Boiler
        n_points = 10
        h_array_boiler = np.linspace(self.state[in_eva_subcool].h_Jkg, self.state[out_eva].h_Jkg, n_points)
        p_array_boiler = np.linspace(self.state[in_eva_subcool].P_Pa, self.state[out_eva].P_Pa, n_points)
        T_array_boiler = np.array([self.T[in_eva_subcool], self.T[out_eva]])  #np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_boiler, h_array_boiler)])
        T_array_PHE_geo_boiler = np.array([T_c, T_b])  #np.linspace(T_c, T_b, n_points)

        #Sub-cooler
        n_points = 10
        h_array_subcool = np.linspace(self.state[in_eva_preheat].h_Jkg, self.state[in_eva_subcool].h_Jkg, n_points)
        p_array_subcool = np.linspace(self.state[in_eva_preheat].P_Pa, self.state[in_eva_subcool].P_Pa, n_points)
        T_array_subcool = np.array([self.T[in_eva_preheat], self.T[in_eva_subcool]])  # np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_subcool, h_array_subcool)])
        T_array_PHE_geo_subcool = np.array([T_c, T_c])  # np.linspace(T_c, T_c, n_points)

        #Economizer
        n_points = 10
        h_array_eco = np.linspace(self.state[in_eco].h_Jkg, self.state[out_eco_preheat].h_Jkg, n_points)
        p_array_eco = np.linspace(self.state[in_eco].P_Pa, self.state[out_eco_preheat].P_Pa, n_points)
        T_array_eco = np.array([self.T[in_eco], self.T[out_eco_preheat]])  # np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_eco, h_array_eco)])
        T_array_PHE_geo_eco = np.array([T_d, T_c])  # np.linspace(T_d, T_c, n_points)

        #Superheater
        n_points = 10
        h_array_sh = np.linspace(self.state[in_sh].h_Jkg, self.state[out_sh].h_Jkg, n_points)
        p_array_sh = np.linspace(self.state[in_sh].P_Pa, self.state[out_sh].P_Pa, n_points)
        T_array_sh = np.array([self.T[in_sh], self.T[out_sh]])  # np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_sh, h_array_sh)])
        T_array_PHE_geo_sh = np.array([T_b, T_a])  # np.linspace(T_b, T_a, n_points)

        #Water condenser
        self.state_cond = [None] * 2
        p_amb = 10e4
        cp_amb = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water').cp_JK

        if self.params.dT_cooling == 0:
            self.T_cooling_water_mid = self.T[out_desh] - self.params.dT_pp_cond
            #R = m_cooling/m_water
            self.R = mdot_ratio * abs(q_condenser_orc)/(cp_amb * (self.T_cooling_water_mid - self.params.T_cooling_water_in))
            T_cooling_water_out = self.T_cooling_water_mid + abs(q_desuperheater_orc)/(cp_amb * self.R)
            self.state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water')
            p_water_out = p_amb - (self.params.dp_water_condenser * 10e4)
            self.state_cond[1] = FluidState.getStateFromPT(p_water_out, T_cooling_water_out, 'Water')
        else:
            T_cooling_water_out = self.params.T_cooling_water_in + self.params.dT_cooling
            ratio = q_condenser_orc/q_desuperheater_orc
            self.T_b = (self.params.T_cooling_water_in + ratio * T_cooling_water_out)/(1 + ratio)
            self.params.dT_pp_cond = self.T[out_desh] - self.T_cooling_water_mid

            if self.T_cooling_water_mid > T_condense_C:
                raise ValueError('GenGeo::ORCCycleTboil:T_cooling_water_mid cannot be greater than T_condense_C')

            self.R = mdot_ratio * abs(q_condenser_orc)/(cp_amb * (self.T_cooling_water_mid - self.params.T_cooling_water_in))
            self.state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water')
            p_water_out = p_amb - (self.params.dp_water_condenser * 10e4)
            self.state_cond[1] = FluidState.getStateFromPT(p_water_out, T_cooling_water_out, 'Water')

        #Points for water the condenser
        n_points = 10
        h_array_cond = np.linspace(self.state[in_cond].h_Jkg, self.state[out_cond].h_Jkg,n_points)
        p_array_cond = np.linspace(self.state[in_cond].P_Pa, self.state[out_cond].P_Pa, n_points)
        T_array_cond = np.array([self.T[in_cond], self.T[out_cond]]) #np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_cond, h_array_cond)])
        T_array_cond_water = np.array([self.params.T_cooling_water_in, self.T_cooling_water_mid]) #np.linspace(self.params.T_cooling_water_in, self.T_cooling_water_mid, n_points)

        #Desuperheater
        n_points = 10
        h_array_desh = np.linspace(self.state[out_desh].h_Jkg, self.state[in_desh].h_Jkg, n_points)
        p_array_desh = np.linspace(self.state[out_desh].P_Pa, self.state[in_desh].P_Pa, n_points)
        T_array_desh = np.array([self.T[out_desh], self.T[in_desh]]) #np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_desh, h_array_desh)])
        T_array_desh_water = np.array([self.T_cooling_water_mid, T_cooling_water_out]) #np.linspace(self.T_cooling_water_mid, T_cooling_water_out, n_points)

        # Creazione del dizionario HXs
        HXs = {
             'condenser': {
                 'T1': [T_array_cond],
                 'T2': [T_array_cond_water],
                 'fluid1': [self.params.orc_fluid],  # Fluido orc
                 'fluid2': ['water'],  # Fluido utilizzato per il raffreddamento (ad esempio acqua)
                 'Q_sections': [abs(q_condenser_orc)],
                 'HX_parameters': {'HX_arrangement': ['counterflow']}  # Configurazione di scambio termico
             },
            'desuperheater': {
                'T1': [T_array_desh],
                'T2': [T_array_desh_water],
                'fluid1': [self.params.orc_fluid],  # Fluido orc
                'fluid2': ['water'],  # Fluido utilizzato per il raffreddamento (ad esempio acqua)
                'Q_sections': [abs(q_desuperheater_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}  # Configurazione di scambio termico
            },
            'evaporator': {
                'T1': [T_array_boiler],
                'T2': [T_array_PHE_geo_boiler],
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.working_fluid],  # Fluido geotermico
                'Q_sections': [abs(q_boiler_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            },
            'economizer': {
                'T1': [T_array_eco],
                'T2': [T_array_PHE_geo_eco],
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.working_fluid],  # Fluido geotermico
                'Q_sections': [abs(q_preheater_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            },
            'superheater': {
                'T1': [T_array_sh],
                'T2': [T_array_PHE_geo_sh],
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.working_fluid],  # Fluido geotermico
                'Q_sections': [abs(q_superheater_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            },
            'subcooler':{
                'T1': [T_array_subcool],
                'T2': [T_array_PHE_geo_subcool],
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.working_fluid],  # Fluido geotermico
                'Q_sections': [0],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            },
            'recuperator': {
                'T1': [T_array_rec_hot],  # T_rec_hot
                'T2': [T_array_rec_cold],  # T_rec_cold
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.orc_fluid],
                'Q_sections': [abs(q_recuperator_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            }
        }

        # Chiamo la funzione PlotTQHX
        HX_names = ['evaporator', 'condenser', 'recuperator'] # Elenco dei nomi dei componenti da plottare

        # Chiamo la funzione di plotting
        PlotTQHX(HXs, HX_names=HX_names)

        return results
