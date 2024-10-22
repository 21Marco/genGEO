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
        if self.params.dp_dT_loss[8] > 0:
            self.p[out_desh] = self.state[out_cond].P_Pa / (1 - self.params.dp_dT_loss[8])  # dp for the condenser
        else:
            self.T[out_desh] = self.state[out_cond].T_C + self.params.dp_dT_loss[8]
            self.p[out_desh] = FluidState.getStateFromTQ(self.T[out_desh], 1, self.params.orc_fluid).P_Pa

        #Desuperheater
        if self.params.dp_dT_loss[7] > 0:
            self.p[in_desh] = self.p[out_desh] / (1 - self.params.dp_dT_loss[7])  # dp for the desuperheater
        else:
            self.p[in_desh] = self.params.dp_dT_loss[7] + self.p[out_desh]

        #Boiler/Evaporator
        if self.params.dp_dT_loss[3] > 0:
            self.p[out_eco_subcool] = self.state[out_eco_subcool].P_Pa / (1 - self.params.dp_dT_loss[3])  # dp for the boiler
        else:
            self.T[out_eco_subcool] = self.state[out_eva].T_C + self.params.dp_dT_loss[3]  # - dT_sc
            self.p[out_eco_subcool] = FluidState.getStateFromTQ(self.T[out_eco_subcool], 0, self.params.orc_fluid).P_Pa

        #Preheater/Economizer
        if self.params.dp_dT_loss[2] > 0:
            self.p[in_eco] = self.p[out_eco_subcool] / (1 - self.params.dp_dT_loss[2])  # dp for the pre-heater
        else:
            self.p[in_eco] = self.params.dp_dT_loss[2] + self.p[out_eco_subcool]

        # State 10 (Desuperheater -> Condenser)
        # saturated vapor
        self.state[out_desh] = FluidState.getStateFromPQ(self.p[out_desh], 1, self.params.orc_fluid)
        self.update_properties(out_desh)

        if self.params.orc_Saturated:  # Saturated ORC Cycle (without SH)
            if self.params.orc_no_Rec:  # without Recuperator

                # #Compute the TDN points
                #State 7 = 6 (Boiler -> Turbine)
                self.state[out_sh] = self.state[out_eva]
                self.update_properties(out_sh)

                #State 5 (Preheater -> Boiler)
                #saturated liquid
                self.state[out_eco_subcool] = FluidState.getStateFromPQ(self.p[out_eco_subcool], 0, self.params.orc_fluid)
                self.update_properties(out_eco_subcool)

                #State 2 = 3 (Pump -> Preheater)
                h_out_pump_s= FluidState.getStateFromPS(self.p[in_eco], self.state[out_cond].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_pump]= self.state[out_cond].h_Jkg - ((self.state[out_cond].h_Jkg - h_out_pump_s) / self.params.eta_pump_orc)
                self.state[out_pump] = FluidState.getStateFromPh(self.p[in_eco], self.h[out_pump], self.params.orc_fluid)
                self.state[in_eco] = self.state[out_pump]
                self.update_properties(in_eco)
                self.update_properties(out_pump)

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
                if self.params.dp_dT_loss[6] > 0:
                    self.p[out_turb] = self.p[out_rec_hot] / (1 - self.params.dp_dT_loss[6])  # dp for the hot side recuperator
                else:
                    self.p[out_turb] = self.params.dp_dT_loss[6] + self.p[out_rec_hot]

                #Recuperator cold side
                if self.params.dp_dT_loss[1] > 0:
                    self.p[out_pump] = self.p[out_rec_cold] * (1 - self.params.dp_dT_loss[1])  # dp for the recuperator cold side
                else:
                    self.p[out_pump] = self.p[out_rec_cold] - self.params.dp_dT_loss[1]

                # #Compute the TDN points
                #State 7 = 6 (Turbine -> Recuperator hot side)
                self.state[out_sh] = self.state[out_eva]
                self.update_properties(out_sh)
                self.update_properties(out_eva)

                #State 5 (Preheater -> Boiler)
                #saturated liquid
                self.state[out_eco_subcool] = FluidState.getStateFromPQ(self.p[out_eco_subcool], 0, self.params.orc_fluid)
                self.update_properties(out_eco_subcool)

                #State 2 (Pump -> Recuperator cold side)
                h_out_pump_s = FluidState.getStateFromPS(self.p[out_pump], self.state[out_cond].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_pump] = self.state[out_cond].h_Jkg - ((self.state[out_cond].h_Jkg - h_out_pump_s) / self.params.eta_pump_orc)
                self.state[out_pump] = FluidState.getStateFromPh(self.p[out_pump], self.h[out_pump], self.params.orc_fluid)
                self.update_properties(out_pump)

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
                if self.params.dp_dT_loss[4] > 0:
                    self.p[out_sh] = self.state[out_eva].P_Pa * (1 - self.params.dp_dT_loss[4])  # dp for the superheater
                else:
                    self.p[out_sh] = self.state[out_eva].P_Pa - self.params.dp_dT_loss[4]

                # #Compute TDN points
                #State 7 (Superheater -> Turbine)
                self.T[out_sh] = T_in_C - self.params.dT_ap_phe    #T_boil_C (T_5) + self.params.dT_sh_phe
                self.state[out_sh] = FluidState.getStateFromPT(self.p[out_sh], self.T[out_sh], self.params.orc_fluid)
                self.update_properties(out_sh)

                #State 9 = 8 (Turbine -> Desuperheater)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_desh], self.state[out_sh].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_sh].h_Jkg - self.params.eta_turbine_orc * (self.state[out_sh].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[in_desh], self.h[out_turb], self.params.orc_fluid)
                self.state[out_rec_hot] = self.state[out_turb]
                self.update_properties(out_rec_hot)
                self.update_properties(out_turb)

                #State 2 = 3 (Pump -> Preheater)
                h_out_pump_s = FluidState.getStateFromPS(self.p[in_eco], self.state[out_cond].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_pump] = self.state[out_cond].h_Jkg - ((self.state[out_cond].h_Jkg - h_out_pump_s) / self.params.eta_pump_orc)
                self.state[out_pump] = FluidState.getStateFromPh(self.p[in_eco], self.h[out_pump], self.params.orc_fluid)
                self.state[in_eco] = self.state[out_pump]
                self.update_properties(in_eco)
                self.update_properties(out_pump)

                #State 5 (Preheater -> Boiler)
                #saturated liquid
                self.state[out_eco_subcool] = FluidState.getStateFromPQ(self.p[out_eco_subcool], 0, self.params.orc_fluid)
                self.update_properties(out_eco_subcool)

            else:  # ciclo orc SH con rec

                #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                #Superheater
                if self.params.dp_dT_loss[4] > 0:
                    self.p[out_sh] = self.state[out_eva].P_Pa * (1 - self.params.dp_dT_loss[4])  # dp for the superheater
                else:
                    self.p[out_sh] = self.state[out_eva].P_Pa - self.params.dp_dT_loss[4]

                #Recuperator hot side
                if self.params.dp_dT_loss[6] > 0:
                    self.p[out_turb] = self.p[out_rec_hot] / (1 - self.params.dp_dT_loss[6])  # dp for the hot side recuperator
                else:
                    self.p[out_turb] = self.params.dp_dT_loss[6] + self.p[out_rec_hot]

                #Recuperator cold side
                if self.params.dp_dT_loss[1] > 0:
                    self.p[out_pump] = self.p[out_rec_cold] * (1 - self.params.dp_dT_loss[1])  # dp for the cold side recuperator
                else:
                    self.p[out_pump] = self.p[out_rec_cold] - self.params.dp_dT_loss[1]

                # #Compute TDN points
                #State 7 (Superheater -> Turbine)
                self.T[out_sh] = T_in_C - self.params.dT_ap_phe   #T_boil_C + self.params.dT_sh_phe
                self.state[out_sh] = FluidState.getStateFromPT(self.p[out_sh], self.T[out_sh], self.params.orc_fluid)
                self.update_properties(out_sh)

                #State 8 (Turbine -> Recuperator)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_rec_hot], self.state[out_sh].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_sh].h_Jkg - self.params.eta_turbine_orc * (self.state[out_sh].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[out_turb], self.h[out_turb], self.params.orc_fluid)
                self.update_properties(out_turb)

                #State 5 (Preheater -> Boiler)
                #saturated liquid
                self.state[out_eco_subcool] = FluidState.getStateFromPQ(self.p[out_eco_subcool], 0, self.params.orc_fluid)
                self.update_properties(out_eco_subcool)

                #State 2 (Pump -> Recuperator)
                h_out_pump_s = FluidState.getStateFromPS(self.p[out_pump], self.state[out_cond].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_pump] = self.state[out_cond].h_Jkg - ((self.state[out_cond].h_Jkg - h_out_pump_s) / self.params.eta_pump_orc)
                self.state[out_pump] = FluidState.getStateFromPh(self.p[out_pump], self.h[out_pump], self.params.orc_fluid)
                self.update_properties(out_pump)

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
        T_array_rec_hot = np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_rec_hot, h_array_rec_hot)])
        T_array_rec_cold = np.array([FluidState.getStateFromPh(p2, h2, self.params.orc_fluid).T_C for p2, h2 in zip(p_array_rec_cold, h_array_rec_cold)])

        #Points for the condenser
        n_points = 10
        h_array_cond = np.linspace(self.state[in_cond].h_Jkg, self.state[out_cond].h_Jkg,n_points)
        p_array_cond = np.linspace(self.state[in_cond].P_Pa, self.state[out_cond].P_Pa, n_points)
        T_array_cond = np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_cond, h_array_cond)])
        T_in_water = self.T[out_cond] - self.params.dT_ap_cond
        T_out_water = self.T[out_desh] - self.params.dT_pp_cond
        T_array_cond_water = np.linspace(T_in_water, T_out_water, n_points)

        # Cooling Tower Parasitic load
        dT_range = self.T[out_rec_hot] - self.T[out_desh]  # 9 e 10
        parasiticPowerFraction = CoolingCondensingTower.parasiticPowerFraction(self.params.T_ambient_C, self.params.dT_approach, dT_range, self.params.cooling_mode)
        w_cooler_orc = q_desuperheater_orc * parasiticPowerFraction('cooling')
        w_condenser_orc = q_condenser_orc * parasiticPowerFraction('condensing')

        # water (assume pressure 100 kPa above saturation)
        P_sat = FluidState.getStateFromTQ(T_in_C, 0, 'Water').P_Pa
        cp = FluidState.getStateFromPT(P_sat + 100e3, T_in_C, 'Water').cp_JK
        # Water state 11, inlet, 12, mid, 13 exit
        T_C_11 = T_in_C
        T_C_12 = T_boil_C + dT_pinch
        # mdot_ratio = mdot_orc / mdot_water
        mdot_ratio = cp * (T_C_11 - T_C_12) / q_boiler_orc
        T_C_13 = T_C_12 - mdot_ratio * q_preheater_orc / cp

        # check that T_C(13) isn't below pinch constraint
        if T_C_13 < (self.T[out_pump] + dT_pinch):
            # pinch constraint is here, not at 12
            # outlet is pump temp plus pinch
            T_C_13 = self.T[out_pump] + dT_pinch
            R = q_boiler_orc / (q_boiler_orc + q_preheater_orc)
            T_C_12 = T_C_11 - (T_C_11 - T_C_13) * R
            mdot_ratio = cp * (T_C_11 - T_C_12) / q_boiler_orc

        # Calculate water heat/work
        results.q_preheater = mdot_ratio * q_preheater_orc
        results.q_boiler = mdot_ratio * q_boiler_orc
        results.q_desuperheater = mdot_ratio * q_desuperheater_orc
        results.q_condenser = mdot_ratio * q_condenser_orc
        results.w_turbine = mdot_ratio * w_turbine_orc
        results.w_pump = mdot_ratio * w_pump_orc
        results.w_cooler = mdot_ratio * w_cooler_orc
        results.w_condenser = mdot_ratio * w_condenser_orc
        results.w_net = results.w_turbine + results.w_pump + results.w_cooler + results.w_condenser

        # Calculate temperatures
        results.dT_range_CT = self.T[out_rec_hot] - self.T[out_desh]
        dT_A_p = T_C_13 - self.T[out_pump]
        dT_B_p = T_C_12 - self.T[out_eco_subcool]
        if dT_A_p == dT_B_p:
            results.dT_LMTD_preheater = dT_A_p
        else:
            div = dT_A_p / dT_B_p
            results.dT_LMTD_preheater = (dT_A_p - dT_B_p) / (math.log(abs(div)) * np.sign(div))

        dT_A_b = T_C_12 - self.T[out_eco_subcool]
        dT_B_b = T_C_11 - self.T[out_eva]
        if dT_A_b == dT_B_b:
            results.dT_LMTD_boiler = dT_A_b
        else:
            div = dT_A_b / dT_B_b
            results.dT_LMTD_boiler = (dT_A_b - dT_B_b) / (math.log(abs(div)) * np.sign(div))

        # return temperature
        results.state = FluidState.getStateFromPT(initialState.P_Pa, T_C_13, self.params.working_fluid)

        #Points for the PHE
        #Evaporator


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
            # 'evaporator': {
            #     'T1': [self.state[4].T_C],  # Tin_eva
            #     'T2': [self.state[5].T_C],  # Tout_eva
            #     'fluid1': [self.params.orc_fluid],
            #     'fluid2': [self.params.working_fluid],  # Fluido geotermico
            #     'Q_sections': [abs(q_boiler_orc)],
            #     'HX_parameters': {'HX_arrangement': ['counterflow']}
            # },
            'recuperator': {
                'T1': [T_array_rec_hot],  # T_rec_hot
                'T2': [T_array_rec_cold],  # T_rec_cold
                'fluid1': [self.params.orc_fluid],
                'fluid2': [self.params.orc_fluid],
                'Q_sections': [abs(q_recuperator_orc)],
                'HX_parameters': {'HX_arrangement': ['counterflow']}
            # },
            # 'preheater': {
            #     'T1': [self.state[2].T_C],  # Tout rec_cold_side
            #     'T2': [self.state[3].T_C],  # Tout_preheater
            #     'fluid1': [self.params.orc_fluid],
            #     'fluid2': [self.params.working_fluid],
            #     'Q_sections': [abs(q_preheater_orc)],
            #     'HX_parameters': {'HX_arrangement': ['counterflow']}
            # },
            # 'superheater': {
            #     'T1': [self.state[5].T_C],  # Tin_superheater
            #     'T2': [self.state[6].T_C],  # Tout_superheater
            #     'fluid1': [self.params.orc_fluid],
            #     'fluid2': [self.params.working_fluid],  # Fluidi coinvolti (puoi modificare in base alle tue esigenze)
            #     'Q_sections': [abs(q_superheater_orc)],  # Calore trasferito nel superheater
            #     'HX_parameters': {'HX_arrangement': ['counterflow']}
            # },
            # 'desuperheater': {
            #     'T1': [self.state[8].T_C],  # Tin_desuperheater
            #     'T2': [self.state[9].T_C],  # Tout_desuperheater
            #     'fluid1': [self.params.orc_fluid],
            #     'fluid2': ['water'],
            #     'Q_sections': [abs(q_desuperheater_orc)],  # Calore rimosso dal desuperheater
            #     'HX_parameters': {'HX_arrangement': ['counterflow']}
            }
        }

        # Chiamo la funzione PlotTQHX
        HX_names = ['condenser', 'recuperator'] #['evaporator', 'condenser', 'recuperator'] # Elenco dei nomi dei componenti da plottare

        # Chiamo la funzione di plotting
        PlotTQHX(HXs, HX_names=HX_names)

        return results
