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
import psychrolib as psy

from src.coolingCondensingTower import CoolingCondensingTower
from src.condenser import CondenserType, Condenser
from src.powerPlantOutput import PowerPlantOutput
from src.plotDiagrams import TsDischarge, PlotTQHX
from psychrolib import SetUnitSystem, SI

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
        #self.T_boil_max = maxSubcritORCBoilTemp(self.orc_fluid)
        self.T = {}
        #self.results = {}  # Inizializzazione di results come dizionario

    def update_properties(self, index):
        """Aggiorna temperatura, pressione ed entalpia in base allo stato definito."""
        if self.state[index] is not None:
            self.T[index] = self.state[index].T_C  # Temperatura in °C
            self.p[index] = self.state[index].P_Pa  # Pressione in Pa
            self.h[index] = self.state[index].h_Jkg  # Entalpia in J/kg
            self.s[index] = self.state[index].s_JK

    def calculate_wet_bulb_temperature(self):
        """
        Calcola la temperatura di bulbo umido basata sulle condizioni ambientali della classe.
        """
        SetUnitSystem(SI)  # Sistema internazionale
        T_wb = psy.GetTWetBulbFromRelHum(self.params.T_ambient_C, self.params.RH_in, 101300)
        return T_wb

    def solve(self, initialState, dT_ap_phe = False, dT_sh_phe = False):

        T_in_C = initialState.T_C

        if not dT_ap_phe:
            dT_ap_phe = self.params.dT_ap_phe
            #raise ValueError('GenGeo::ORCCycleTboil:Delta T di approach (dT_ap_phe) deve essere specificato.')

        if not dT_sh_phe:
            dT_sh_phe = self.params.dT_sh_phe
            #raise ValueError('GenGeo::ORCCycleTboil:Delta T di superheating (dT_sh_phe) deve essere specificato.')

        # Compute the maximum temperature of the ORC cycle (T_in_turb)
        T_max_orc = T_in_C - dT_ap_phe
        if np.isnan(T_max_orc):
            raise Exception('GenGeo::ORCCycleTboil:T_max_orc_NaN - Maximum ORC temperature is NaN!')

        # Compute evaporation temperature (T_out_eva)
        T_boil_C = T_max_orc - dT_sh_phe
        if np.isnan(T_boil_C):
            raise Exception('GenGeo::ORCCycleTboil:T_boil_NaN - ORC boil temperature is NaN!')

        # run some checks  if T_in_C and T_boil_C are valid
        if np.isnan(T_in_C):
            raise Exception('GenGeo::ORCCycleTboil:T_in_NaN - ORC input temperature is NaN!')

        if np.isnan(T_boil_C):
            raise Exception('GenGeo::ORCCycleTboil:T_boil_NaN - ORC boil temperature is NaN!')

        #if T_boil_C > FluidState.getTcrit(self.params.orc_fluid) * 0.95:
          #  raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature above critical point')

        # # only refresh T_boil_max if orc_fluid has changed from initial
        # if self.params.orc_fluid != self.orc_fluid:
        #     self.T_boil_max = maxSubcritORCBoilTemp(self.params.orc_fluid)
        #     self.orc_fluid = self.params.orc_fluid
        # if T_boil_C > self.T_boil_max:
        #     raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature of %s is greater than maximum allowed of %s.'%(T_boil_C, self.T_boil_max))

        #Condenser Type and initialization of condensation temperature
        if self.params.condenser_type == CondenserType.WATER:
            T_condense_C = self.params.T_ambient_C + self.params.dT_approach_cond
        elif self.params.condenser_type == CondenserType.AIR:
            T_condense_C = self.params.T_ambient_C + self.params.dT_approach_cond
        elif self.params.condenser_type == CondenserType.EVAPORATIVE_TOWER:
            T_wb = self.calculate_wet_bulb_temperature()  # Temperatura di bulbo umido dell'aria in ingresso
            T_in_water_ct = T_wb + self.params.dT_ct
            T_out_water_ct = T_in_water_ct + self.params.dT_water_ct
            T_condense_C = T_out_water_ct + self.params.dT_pp_star_ct
            T = T_out_water_ct - 15

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
        if self.params.dp_dT_loss['loss_cond'] > 0: # Perdita relativa fornita in %
            self.p[out_desh] = self.state[out_cond].P_Pa / (1 - self.params.dp_dT_loss['loss_cond'])  # dp for the condenser
        else: # Perdita assoluta fornita °C (negativo o 0)
            self.T[out_desh] = self.state[out_cond].T_C - self.params.dp_dT_loss['loss_cond']
            self.p[out_desh] = FluidState.getStateFromTQ(self.T[out_desh], 1, self.params.orc_fluid).P_Pa

        #Desuperheater
        if self.params.dp_dT_loss['loss_desh'] > 0: # Perdita relativa fornita in %
            self.p[in_desh] = self.p[out_desh] / (1 - self.params.dp_dT_loss['loss_desh'])  # dp for the desuperheater, loss_desh in %
        else: # Perdita assoluta fornita in Pascal (negativo o 0)
            self.p[in_desh] = self.params.dp_dT_loss['loss_desh'] + self.p[out_desh]

        #Boiler/Evaporator
        if self.params.dp_dT_loss['loss_eva'] > 0: # Perdita relativa fornita in %
            self.p[out_eco_subcool] = self.state[out_eva].P_Pa / (1 - self.params.dp_dT_loss['loss_eva'])  # dp for the boiler
        else: # Perdita assoluta fornita in °C (negativo o 0)
            self.T[out_eco_subcool] = self.state[out_eva].T_C - self.params.dp_dT_loss['loss_eva']
            self.p[out_eco_subcool] = FluidState.getStateFromTQ(self.T[out_eco_subcool], 0, self.params.orc_fluid).P_Pa

        # Subcooling
        if self.params.dp_dT_loss['loss_sc'] > 0: # Perdita relativa fornita in %
            self.p[out_eco_preheat] = self.p[out_eco_subcool] / (1 - self.params.dp_dT_loss['loss_sc'])  # dp for the subcooling
        else: # Perdita assoluta fornita in Pascal (negativo o 0)
            self.p[out_eco_preheat] = self.p[out_eco_subcool] - self.params.dp_dT_loss['loss_sc']

        #Preheater/Economizer
        if self.params.dp_dT_loss['loss_eco'] > 0: # Perdita relativa fornita in %
            self.p[in_eco] = self.p[out_eco_preheat] / (1 - self.params.dp_dT_loss['loss_eco'])  # dp for the pre-heater
        else: # Perdita assoluta fornita in Pascal (negativo o 0)
            self.p[in_eco] = self.p[out_eco_preheat] - self.params.dp_dT_loss['loss_eco']

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

                #State 8 (Turbine -> Recuperator)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_rec_hot], self.state[out_eva].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_eva].h_Jkg - self.params.eta_turbine_orc * (self.state[out_eva].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[out_turb], self.h[out_turb], self.params.orc_fluid)
                self.update_properties(out_turb)
                T_out_is = FluidState.getStateFromPS(self.p[out_turb], self.state[in_turb].s_JK, self.params.orc_fluid).T_C

                #State 9 (Recuperator hot -> Desuperheater)
                self.T[out_rec_hot] = self.state[out_pump].T_C + self.params.dT_pp_rec
                self.state[out_rec_hot] = FluidState.getStateFromPT(self.p[out_rec_hot], self.T[out_rec_hot], self.params.orc_fluid)
                self.update_properties(out_rec_hot)

                # State 3 (Recuperator cold side -> Preheater)
                self.h[out_rec_cold] = self.h[out_turb] - self.state[out_rec_hot].h_Jkg + self.h[out_pump]
                self.state[out_rec_cold] = FluidState.getStateFromPh(self.p[out_rec_cold], self.h[out_rec_cold], self.params.orc_fluid)
                self.update_properties(out_rec_cold)

        else:  # ORC Cycle with SH
            if self.params.orc_no_Rec:  # without Recuperator

                #Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                #Superheater
                if self.params.dp_dT_loss['loss_sh'] > 0:
                    self.p[out_sh] = self.state[out_eva].P_Pa * (1 - self.params.dp_dT_loss['loss_sh'])  # dp for the superheater
                else:
                    self.p[out_sh] = self.p[out_eva] - self.params.dp_dT_loss['loss_sh']

                # #Compute TDN points
                #State 7 (Superheater -> Turbine)
                self.T[out_sh] = T_max_orc
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
                self.T[out_sh] = T_max_orc
                self.state[out_sh] = FluidState.getStateFromPT(self.p[out_sh], self.T[out_sh], self.params.orc_fluid)
                self.update_properties(out_sh)

                #State 8 (Turbine -> Recuperator)
                h_out_turb_s = FluidState.getStateFromPS(self.p[in_rec_hot], self.state[out_sh].s_JK, self.params.orc_fluid).h_Jkg
                self.h[out_turb] = self.state[out_sh].h_Jkg - self.params.eta_turbine_orc * (self.state[out_sh].h_Jkg - h_out_turb_s)
                self.state[out_turb] = FluidState.getStateFromPh(self.p[out_turb], self.h[out_turb], self.params.orc_fluid)
                self.update_properties(out_turb)
                #H_Turb_iso = FluidState.getStateFromPS(self.p[out_turb], self.state[in_turb].s_JK, self.params.orc_fluid).h_Jkg

                # State 9 (Recuperator hot -> Desuperheater)
                self.T[out_rec_hot] = self.state[out_pump].T_C + self.params.dT_pp_rec
                self.state[out_rec_hot] = FluidState.getStateFromPT(self.p[out_rec_hot], self.T[out_rec_hot], self.params.orc_fluid)
                self.update_properties(out_rec_hot)

                #State 3 (Recuperator cold -> Preheater)
                self.h[out_rec_cold] = self.h[out_turb] - self.state[out_rec_hot].h_Jkg + self.h[out_pump]
                self.state[out_rec_cold] = FluidState.getStateFromPh(self.p[out_rec_cold], self.h[out_rec_cold], self.params.orc_fluid)
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
        #TsDischarge(self.state, self.T, self.params.orc_fluid, self.params.orc_no_Rec)

        results = PowerPlantOutput()

        #Calculate orc heat/work
        w_pump_orc = self.h[out_cond] - self.h[out_pump]
        q_recuperator_orc = -1 * (self.h[out_pump] - self.h[out_rec_cold])
        q_preheater_orc = -1 * (self.h[out_rec_cold] - self.h[out_eco_preheat])
        q_boiler_orc = -1 * (self.h[out_eco_preheat] - self.h[out_eva])
        q_superheater_orc = -1 * (self.h[out_eva] - self.h[out_sh])
        w_turbine_orc = self.h[out_sh] - self.h[out_turb]
        q_desuperheater_orc = -1 * (self.h[out_rec_hot] - self.h[out_desh])
        q_condenser_orc = -1 * (self.h[out_desh] - self.h[out_cond])

        results.dP_pump_orc = self.p[out_pump] - self.p[out_cond]
        results.P_boil = self.p[out_eco_subcool]

        # water (assume pressure 100 kPa above saturation)
        P_sat_w = FluidState.getStateFromTQ(T_in_C, 0, 'Water').P_Pa  # al posto di water, self.params.working_fuid
        cp = FluidState.getStateFromPT(P_sat_w + 100e3, T_in_C, 'Water').cp_JK  # al posto di P_sat + 100e3 mettere la P_out_well_geothermal_fluid, initialState.P_Pa
        # Water state a, inlet, b, mid, c, mid, d, exit
        T_a = T_in_C
        T_c = self.T[out_eco_subcool] + self.params.dT_pinch
        # mdot_ratio = mdot_orc / mdot_water
        mdot_ratio = cp * (T_a - T_c) / (q_boiler_orc + q_superheater_orc)
        T_d = T_c - mdot_ratio * q_preheater_orc / cp  # reinjection temperature
        results.T_d = T_d
        # T_d = T_c - mdot_ratio / cp * (self.h[out_eco_preheat] - self.h[out_rec_cold])

        # check that T_d isn't below pinch constraint
        if T_d < (self.T[out_rec_cold] + self.params.dT_pinch):
            # pinch constraint is here, not at c
            # outlet is rec cold temp plus pinch
            T_d = self.T[out_rec_cold] + self.params.dT_pinch
            R = q_boiler_orc / (q_boiler_orc + q_preheater_orc)
            T_c = T_a - (T_a - T_d) * R
            mdot_ratio = cp * (T_a - T_c) / (q_boiler_orc + q_superheater_orc)

        # Calculate T_b, BE SH
        T_b = T_a - mdot_ratio * q_superheater_orc / cp

        # Calculate temperatures
        results.dT_range_PHE = self.T[out_eva] - self.T[out_rec_cold]
        # preheater
        dT_A_p = T_d - self.T[out_rec_cold]
        dT_B_p = T_c - self.T[out_eco_subcool]
        if dT_A_p == dT_B_p:
            results.dT_LMTD_preheater = dT_A_p
        else:
            div = dT_A_p / dT_B_p
            results.dT_LMTD_preheater = (dT_A_p - dT_B_p) / (math.log(abs(div)) * np.sign(div))

        #boiler
        dT_A_b = T_c - self.T[out_eco_subcool]
        dT_B_b = T_b - self.T[out_eva]
        if dT_A_b == dT_B_b:
            results.dT_LMTD_boiler = dT_A_b
        else:
            div = dT_A_b / dT_B_b
            results.dT_LMTD_boiler = (dT_A_b - dT_B_b) / (math.log(abs(div)) * np.sign(div))

        #superheater
        dT_A_s = T_b - self.T[out_eva]
        dT_B_s = T_a - self.T[out_sh]
        if dT_A_s == dT_B_s:
            results.dT_LMTD_superheater = dT_A_s
        else:
            div = dT_A_s / dT_B_s
            results.dT_LMTD_superheater = (dT_A_s - dT_B_s) / (math.log(abs(div)) * np.sign(div))

        # return temperature
        results.state = FluidState.getStateFromPT(initialState.P_Pa, T_d, self.params.working_fluid)

        self.condenser = Condenser(self.params.condenser_type, self.params, T_condense_C)

        if self.params.condenser_type == CondenserType.WATER:
            w_pump_water_cond, T_cooling_water_mid, T_cooling_water_out = self.condenser.computeWaterCondenser(self.T[out_desh], mdot_ratio, q_condenser_orc, q_desuperheater_orc, T_condense_C)
            w_pump_cooler = w_pump_water_cond  # Solo il valore calcolato
            T_cooling_in = self.params.T_cooling_water_in
            T_cooling_mid = T_cooling_water_mid
            T_cooling_out = T_cooling_water_out
            fluid2_name = 'water'
            w_fan = 0
        elif self.params.condenser_type == CondenserType.AIR:
            w_fan_air_cond, T_cooling_air_mid, T_cooling_air_out = self.condenser.computeAirCondenser(self.T[out_desh], mdot_ratio, q_condenser_orc, q_desuperheater_orc, T_condense_C)
            w_fan = w_fan_air_cond
            T_cooling_in = self.params.T_ambient_C
            T_cooling_mid = T_cooling_air_mid
            T_cooling_out = T_cooling_air_out
            fluid2_name = 'air'
            w_pump_cooler = 0
        elif self.params.condenser_type == CondenserType.EVAPORATIVE_TOWER:
            w_pump_ct, w_fan_ct = self.condenser.computeEvaporativeTower(q_condenser_orc, q_desuperheater_orc)
            w_pump_cooler = w_pump_ct
            w_fan = w_fan_ct
            T_cooling_in = T_in_water_ct
            T_cooling_out = T_out_water_ct
            fraction = q_condenser_orc / (q_condenser_orc + q_desuperheater_orc)
            T_cooling_mid = T_cooling_in + fraction * (T_cooling_out - T_cooling_in)
            fluid2_name = 'water'

        # # Calculate water heat/work
        # results.q_recuperator = mdot_ratio * q_recuperator_orc
        # results.q_preheater = mdot_ratio * q_preheater_orc
        # results.q_boiler = mdot_ratio * q_boiler_orc
        # results.q_superheater = mdot_ratio * q_superheater_orc
        # results.q_desuperheater = mdot_ratio * q_desuperheater_orc
        # results.q_condenser = mdot_ratio * q_condenser_orc
        # results.w_turbine = mdot_ratio * w_turbine_orc
        # results.w_pump = mdot_ratio * w_pump_orc
        # # results.w_desuperheater = mdot_ratio * w_desuperheater_orc
        # # results.w_condenser = mdot_ratio * w_condenser_orc
        # results.w_pump_cooler = mdot_ratio * w_pump_cooler
        # results.w_fan = mdot_ratio * w_fan
        # results.w_net = results.w_turbine + results.w_pump #+ results.q_desuperheater + results.q_condenser + results.w_pump_cooler + results.w_fan
        # print(results.w_net)  # Stampa il valore di w_net

        # Calculate water heat/work fixing m_geo
        q_recuperator = q_recuperator_orc * mdot_ratio * self.params.m_geo
        q_preheater = q_preheater_orc * mdot_ratio * self.params.m_geo
        q_boiler = q_boiler_orc * mdot_ratio * self.params.m_geo
        q_superheater = q_superheater_orc * mdot_ratio * self.params.m_geo
        q_desuperheater = q_desuperheater_orc * mdot_ratio * self.params.m_geo
        q_condenser = q_condenser_orc * mdot_ratio * self.params.m_geo
        w_turbine = w_turbine_orc * mdot_ratio * self.params.m_geo
        w_turbine = w_turbine_orc * mdot_ratio * self.params.m_geo
        w_pump = w_pump_orc * mdot_ratio * self.params.m_geo
        results.w_net = w_turbine + w_pump

        # Turbine cost
        results.C_T_G = 1230000 * (n / 2) ** 0.5 * (SP / 0.18) ** 1.1

        # Pump cost
        results.C_pump_orc = 14000 * (w_pump / 200000) ** 0.67

        # Air Cooled Condenser cost
        U_desh = 1088.82 #W/m**2-K
        U_cond = 303.13  #W/m**2-K
        # Calculate temperatures Desuperheater
        results.dT_range_ACC = self.T[out_turb] - self.T[out_cond]
        dT_A_d = self.T[out_turb] - T_cooling_out
        dT_B_d = self.T[out_desh] - T_cooling_mid
        if dT_A_d == dT_B_d:
            results.dT_LMTD_desuperheater = dT_A_p
        else:
            div = dT_A_d / dT_B_d
            results.dT_LMTD_desuperheater = (dT_A_d - dT_B_d) / (math.log(abs(div)) * np.sign(div))
        A_desh = q_desuperheater/(U_desh * results.dT_LMTD_desuperheater)

        # Calculate temperatures Condenser
        dT_A_c = self.T[out_desh] - T_cooling_mid
        dT_B_c = self.T[out_cond] - T_cooling_in
        if dT_A_c == dT_B_c:
            results.dT_LMTD_condenser = dT_A_c
        else:
            div = dT_A_c / dT_B_c
            results.dT_LMTD_condenser = (dT_A_c - dT_B_c) / (math.log(abs(div)) * np.sign(div))
        A_cond = q_condenser / (U_cond * results.dT_LMTD_condenser)
        results.C_ACC = 530000 * ((A_desh + A_cond) / 3563) ** 0.9

        # Heat Exchanger cost
        # dT_LMTD_HX
        # U = 500/1000 #kW/m**2-K
        U = 500  # W/m**2-K
        if np.isnan(dT_LMTD_preheater) or dT_LMTD_preheater == 0:
            A_preheater = 0
        else:
            A_preheater = Q_preheater / U / dT_LMTD_preheater

        A_boiler = Q_boiler / U / dT_LMTD_boiler
        A_HX = A_preheater + A_boiler
        a = 10 ** (0.03881 - 0.11272*math.log(self.p[in_eco]) + 0.08183*(math.log(self.p[in_eco]))**2)
        results.C_HE = 1500000 * ((U*A)/(4000)) **0.9 * a

        # Recuperator cost
        if np.isnan(dT_LMTD_recuperator) or dT_LMTD_recuperator == 0:
            A_recuperator = 0
            results.C_recuperator = 0
        else:
            A_recuperator = Q_recuperator / U / dT_LMTD_recuperator
            b = 10 ** (-0.00164 - 0.00627 * math.log(self.p[out_pump]) + 0.0123 * (math.log(self.p[out_pump])) ** 2)
            results.C_recuperator = 260000 * (U*A/650) ** 0.9 * b





        #PlotTQHX
        # Points for the recuperator
        n_points = 10
        h_array_rec_hot = np.linspace(self.h[out_rec_hot], self.h[in_rec_hot], n_points)  # entalpia lato caldo
        h_array_rec_cold = np.linspace(self.h[in_rec_cold], self.h[out_rec_cold], n_points)  # entalpia lato freddo
        p_array_rec_hot = np.linspace(self.p[out_rec_hot], self.p[in_rec_hot],n_points)  # pressione lato caldo
        p_array_rec_cold = np.linspace(self.p[in_rec_cold], self.p[out_rec_cold], n_points)  # pressione lato freddo
        T_array_rec_hot = np.array([FluidState.getStateFromPh(p1, h1, self.params.orc_fluid).T_C for p1, h1 in zip(p_array_rec_hot, h_array_rec_hot)])  # dalla coppia del primo elemento p e h, ottiene la T
        T_array_rec_cold = np.array([FluidState.getStateFromPh(p2, h2, self.params.orc_fluid).T_C for p2, h2 in zip(p_array_rec_cold, h_array_rec_cold)])

        #Points for the PHE
        #Evaporator/Boiler
        T_array_boiler = np.array([self.T[in_eva_subcool], self.T[out_eva]])
        T_array_PHE_geo_boiler = np.array([T_c, T_b])

        #Sub-cooler
        T_array_subcool = np.array([self.T[in_eva_preheat], self.T[in_eva_subcool]])
        T_array_PHE_geo_subcool = np.array([T_c, T_c])

        #Economizer
        T_array_eco = np.array([self.T[in_eco], self.T[out_eco_preheat]])
        T_array_PHE_geo_eco = np.array([T_d, T_c])

        #Superheater
        T_array_sh = np.array([self.T[in_sh], self.T[out_sh]])
        T_array_PHE_geo_sh = np.array([T_b, T_a])

        #Points for water the CONDENSER
        T_array_cond = np.array([self.T[in_cond], self.T[out_cond]])
        T_array_condenser = np.array([T_cooling_in, T_cooling_mid])

        #Desuperheater
        T_array_desh = np.array([self.T[out_desh], self.T[in_desh]])
        T_array_deshuperheater = np.array([T_cooling_mid, T_cooling_out])

        # Creazione del dizionario HXs
        HXs = {
             'condenser': {
                 'T1': [T_array_cond],
                 'T2': [T_array_condenser],
                 'fluid1': [self.params.orc_fluid],  # Fluido orc
                 'fluid2': [fluid2_name],  # Fluido utilizzato per il raffreddamento nel condensatore
                 'Q_sections': [abs(q_condenser_orc)],
                 'HX_parameters': {'HX_arrangement': ['counterflow']}  # Configurazione di scambio termico
             },
            'desuperheater': {
                'T1': [T_array_desh],
                'T2': [T_array_deshuperheater],
                'fluid1': [self.params.orc_fluid],  # Fluido orc
                'fluid2': [fluid2_name],  # Fluido utilizzato per il raffreddamento
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
        #PlotTQHX(HXs, HX_names=HX_names)

        return results




