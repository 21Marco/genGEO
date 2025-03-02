from enum import Enum
from utils.fluidState import FluidState

class CondenserType(Enum):
    WATER = 'water'
    AIR = 'air'
    EVAPORATIVE_TOWER = 'evaporative_tower'

class Condenser:
    def __init__(self, condenser_type, params, T):
        self.condenser_type = condenser_type
        self.params = params
        self.T = T

    def computeWaterCondenser(self, T_out_desh, mdot_ratio, q_condenser_orc, q_desuperheater_orc, T_condense_C):
        p_amb = 101300
        cp_amb = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water').cp_JK
        state_cond = [None] * 2

        if self.params.dT_cooling == 0:
            T_cooling_water_mid = T_out_desh - self.params.dT_pp_cond
            #R = m_cooling/m_water
            R = abs(q_condenser_orc) / (cp_amb * (T_cooling_water_mid - self.params.T_cooling_water_in))
            T_cooling_water_out = T_cooling_water_mid + abs(q_desuperheater_orc) / (cp_amb * R)
            state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water')
            p_water_out = p_amb #p_amb - (self.params.dp_water_condenser * 101300)
            state_cond[1] = FluidState.getStateFromPT(p_water_out, T_cooling_water_out, 'Water')
        else:
            T_cooling_water_out = self.params.T_cooling_water_in + self.params.dT_cooling
            ratio = q_condenser_orc / q_desuperheater_orc
            T_cooling_water_mid = (self.params.T_cooling_water_in + ratio * T_cooling_water_out) / (1 + ratio)
            self.params.dT_pinch_cond = T_condense_C - T_cooling_water_mid

            if T_cooling_water_mid > T_condense_C:
                raise ValueError('GenGeo:: Condenser:T_cooling_water_mid cannot be greater than T_condense_C')

            R = mdot_ratio * abs(q_condenser_orc) / (cp_amb * (T_cooling_water_mid - self.params.T_cooling_water_in))
            state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_cooling_water_in, 'Water')
            p_water_out = p_amb - (self.params.dp_water_condenser * 10e4)
            state_cond[1] = FluidState.getStateFromPT(p_water_out, T_cooling_water_out, 'Water')

        w_pump_water_cond = -(self.params.dp_water_condenser * mdot_ratio) / (self.params.rho_water * self.params.eta_me_pump * self.params.eta_hydr_pump)
        return w_pump_water_cond, T_cooling_water_mid, T_cooling_water_out

    def computeAirCondenser(self, T_out_desh, mdot_ratio, q_condenser_orc, q_desuperheater_orc, T_condense_C):
        p_amb = 101300
        cp_amb = FluidState.getStateFromPT(p_amb, self.params.T_ambient_C, 'Air').cp_JK
        state_cond = [None] * 2

        if self.params.dT_cooling == 0:
            T_cooling_air_mid = T_out_desh - self.params.dT_pp_cond
            R = abs(q_condenser_orc) / (cp_amb * (T_cooling_air_mid - self.params.T_ambient_C))
            T_cooling_air_out = T_cooling_air_mid + abs(q_desuperheater_orc) / (cp_amb * R)
            state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_ambient_C, 'Air')
            p_air_out = p_amb #p_amb - self.params.dp_air_condenser
            state_cond[1] = FluidState.getStateFromPT(p_air_out, T_cooling_air_out, 'Air')
        else:
            T_cooling_air_out = self.params.T_ambient_C + self.params.dT_cooling
            ratio = q_condenser_orc / q_desuperheater_orc
            T_cooling_air_mid = (self.params.T_ambient_C + ratio * T_cooling_air_out) / (1 + ratio)
            self.params.dT_pinch_cond = T_condense_C - T_cooling_air_mid

            if T_cooling_air_mid > T_condense_C:
                raise ValueError('GenGeo::Condenser:T_cooling_air_mid cannot be greater than T_condense_C')

            R = mdot_ratio * abs(q_condenser_orc) / (cp_amb * (T_cooling_air_mid - self.params.T_ambient_C))
            state_cond[0] = FluidState.getStateFromPT(p_amb, self.params.T_ambient_C, 'Air')
            p_air_out = p_amb #p_amb - self.params.dp_air_condenser
            state_cond[1] = FluidState.getStateFromPT(p_air_out, T_cooling_air_out, 'Air')

        w_fan_air_cond = -(self.params.dp_air_condenser * mdot_ratio) / self.params.eta_fan
        return w_fan_air_cond, T_cooling_air_mid, T_cooling_air_out

    def computeEvaporativeTower(self, q_condenser_orc, q_desuperheater_orc):
        q_cooling_orc = -(q_condenser_orc + q_desuperheater_orc)
        mdot_ratio_water_ct = q_cooling_orc / (self.params.cp_water * self.params.dT_water_ct)
        w_pump_ct = -(self.params.dP_ct * mdot_ratio_water_ct) / (self.params.rho_water * self.params.eta_me_pump * self.params.eta_hydr_pump)
        w_fan_ct = 0.01 * q_cooling_orc
        return w_pump_ct, w_fan_ct

    def computeCondenser(self):
        if self.condenser_type == CondenserType.WATER:
            return self.computeWaterCondenser()
        elif self.condenser_type == CondenserType.AIR:
            return self.computeAirCondenser()
        elif self.condenser_type == CondenserType.EVAPORATIVE_TOWER:
            return self.computeEvaporativeTower()
        else:
            raise ValueError('GenGeo::Condenser:Unknown condenser type')



