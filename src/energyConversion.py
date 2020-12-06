

class EnergyConversionORC(object):
    """EnergyConversionORC."""

    @staticmethod
    def compute(params, input):
        N_IP_multiplier = params.well_multiplier
        ec = EnergyConversionORC()
        ec.Q_preheater = params.m_dot_IP * input.pp.q_preheater
        ec.Q_recuperator = params.m_dot_IP * input.pp.q_recuperator
        ec.Q_boiler = params.m_dot_IP * input.pp.q_boiler
        ec.Q_desuperheater = params.m_dot_IP * input.pp.q_desuperheater
        ec.Q_condenser = params.m_dot_IP * input.pp.q_condenser
        ec.W_turbine = params.m_dot_IP * input.pp.w_turbine
        ec.W_pump = params.m_dot_IP * input.pp.w_pump
        ec.W_cooler = params.m_dot_IP * input.pp.w_cooler
        ec.W_condenser = params.m_dot_IP * input.pp.w_condenser

        ec.Q_fluid = ec.Q_preheater + ec.Q_boiler

        ec.W_downhole_pump = params.m_dot_IP * input.pump.w_pump
        ec.W_net = params.m_dot_IP * input.pp.w_net + ec.W_downhole_pump

        ec.Q_preheater_total = N_IP_multiplier * ec.Q_preheater
        ec.Q_boiler_total = N_IP_multiplier * ec.Q_boiler
        ec.Q_recuperator_total = N_IP_multiplier * ec.Q_recuperator
        ec.Q_desuperheater_total = N_IP_multiplier * ec.Q_desuperheater
        ec.Q_condenser_total = N_IP_multiplier * ec.Q_condenser
        ec.W_turbine_total = N_IP_multiplier * ec.W_turbine
        ec.W_pump_orc_total = N_IP_multiplier * ec.W_pump
        ec.W_pump_prod_total = N_IP_multiplier * ec.W_downhole_pump
        ec.W_net_total = N_IP_multiplier * ec.W_net
        return ec

class EnergyConversionCPG(object):
    """EnergyConversionCPG."""

    @staticmethod
    def compute(params, input):
        N_IP_multiplier = params.well_multiplier
        ec = EnergyConversionCPG()
        ec.Q_preheater = params.m_dot_IP * input.pp.q_preheater
        ec.Q_recuperator = params.m_dot_IP * input.pp.q_recuperator
        ec.Q_boiler = params.m_dot_IP * input.pp.q_boiler
        ec.Q_desuperheater = 0. # params.m_dot_IP * input.pp.q_desuperheater
        ec.Q_condenser = params.m_dot_IP * input.pp.q_condenser
        ec.W_turbine = params.m_dot_IP * input.pp.w_turbine
        ec.W_pump = params.m_dot_IP * input.pp.w_pump
        ec.W_cooler = params.m_dot_IP * input.pp.w_cooler
        ec.W_condenser = params.m_dot_IP * input.pp.w_condenser

        ec.W_net = params.m_dot_IP * input.pp.w_net

        ec.Q_preheater_total = N_IP_multiplier * ec.Q_preheater
        ec.Q_boiler_total = N_IP_multiplier * ec.Q_boiler
        ec.Q_recuperator_total = N_IP_multiplier * ec.Q_recuperator
        ec.Q_desuperheater_total = N_IP_multiplier * ec.Q_desuperheater
        ec.Q_condenser_total = N_IP_multiplier * ec.Q_condenser
        ec.W_turbine_total = N_IP_multiplier * ec.W_turbine
        ec.W_pump_total = N_IP_multiplier * ec.W_pump
        ec.W_net_total = N_IP_multiplier * ec.W_net
        return ec
