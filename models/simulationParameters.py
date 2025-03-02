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
from models.optimizationType import OptimizationType
from models.coolingCondensingTowerMode import CoolingCondensingTowerMode
from models.wellFieldType import WellFieldType
from src.condenser import CondenserType


class SimulationParameters(object):
    """SimulationParameters provides physical properties of the system."""

    def __init__(self,
                working_fluid = 'water',   #geothermal fluid
                orc_fluid = 'R134a',      #ORC fluid
                #m_orc = 1,
                m_geo = 50,  #kg/s
                m_dot_IP = None,
                time_years = 1.,
                # subsurface model
                depth = 2500.,
                pump_depth = 500.,
                well_radius = 0.205,
                well_spacing = 707,
                monitoring_well_radius = 0.108,
                dT_dz = 0.035,
                silica_precipitation = False,
                T_surface_rock = 15,
                T_ambient_C = 20.,
                reservoir_thickness = 100.,
                permeability = 1.0e-15 * 15000 / 100., # permeability = transmissivity / thickness
                wellFieldType = WellFieldType._5Spot_SharedNeighbor,
                N_5spot = 1, #Square-root of numbe of 5spots which share a central plant in a Many_N configuration. e.g. N=2 is 4 5spots.
                has_surface_gathering_system = True,
                # power plant model
                orc_Saturated = False,  # se True, ciclo saturo
                orc_no_Rec = False,     # se True, no recuperatore
                max_pump_dP = 10.e6,
                eta_pump = 0.70,
                dp_dT_loss = {      #[0, 0, -50000, -1, 0.05, 0.01, 0.02, 0.01, 0.3]
                     'loss_rec_cold': 0,
                     'loss_eco': -0,
                     'loss_sc': -0,
                     'loss_eva': -0,
                     'loss_sh': 0,
                     'loss_rec_hot': 0,
                     'loss_desh': 0,
                     'loss_cond': -0
                },
                dT_pp_phe = 5,     # Pinch al PHE
                dT_pp_rec = 5.,  # pinch al recuperatore
                dT_ap_phe = 10,  # Approach al PHE
                dT_sh_phe = 0,  # superheater al PHE
                dT_sc_phe = 0.1,   # sub_cooling al PHE
                eta_pump_orc = 0.75,   # isoentropic efficiency
                eta_me_pump_orc = 0.95, # electrical mechanical efficiency
                eta_turbine_orc = 0.85, # isoentropic
                eta_me_turbine_orc = 0.95,  # electrical mechanical efficiency
                eta_pump_co2 = 0.9,
                eta_turbine_co2 = 0.78,
                #CONDENSER
                cooling_mode = CoolingCondensingTowerMode.Wet,
                condenser_type = CondenserType.AIR,
                dT_approach_cond = 15,   #differenza T out cond e T in fluido refrigerante
                eta_me_pump = 0.95,  # electrical mechanical efficiency
                eta_hydr_pump = 0.85,  # hydraulic efficiency
                rho_water = 1000,  # kg/m3
                dT_cooling = 0,  # dT_cond = T_cond_out - T_cond_in
                dT_pp_cond = 7.,
                dp_water_condenser = 20000,  # Pa
                #WATER Condneser
                T_cooling_water_in = 20.,  #T water in cond
                #AIR Condenser
                dp_air_condenser = 200,  #Pa
                eta_fan = 0.7,
                #EVAPORATIVE TOWER Condenser
                dT_ct = 3 ,
                dT_water_ct = 12,  # DT water in the cooling tower
                dT_pp_star_ct = 1.,    # T_out_water_ct - T_cond
                cp_water = 4186,   # J/kgK
                dP_ct = 200000,    # Pa
                RH_in = 0.55,    #relative humidity
                dT_max = 10,
                # cost model
                cost_year = 2019,
                success_rate = 0.95,
                F_OM = 0.045,
                discount_rate = 0.096,
                lifetime = 25,
                capacity_factor = 0.85,
                opt_mode = OptimizationType.MinimizeCost,
                # physical properties
                g = 9.81,                       # m/s**2
                rho_rock = 2650.,               # kg/m**3
                c_rock = 1000.,                 # J/kg-K
                k_rock = 2.0,                   # W/m-K
                useWellboreHeatLoss = True,     # bool
                well_segments = 100,            # number of well segments
                # Friction factor
                well_relative_roughness = 55 * 1e-6             # um
                    ):

        self.working_fluid = working_fluid
        self.orc_fluid = orc_fluid
        #self.m_orc = m_orc
        self.m_geo = m_geo
        self.m_dot_IP = m_dot_IP
        self.time_years = time_years
        self.depth = depth
        self.pump_depth = pump_depth
        self.well_radius = well_radius
        self.well_spacing = well_spacing
        self.monitoring_well_radius = monitoring_well_radius
        self.dT_dz = dT_dz
        self.silica_precipitation = silica_precipitation
        self.T_surface_rock = T_surface_rock
        self.T_ambient_C = T_ambient_C
        self.reservoir_thickness = reservoir_thickness
        self.permeability = permeability
        self.wellFieldType = wellFieldType
        self.N_5spot = N_5spot
        self.has_surface_gathering_system = has_surface_gathering_system
        self.orc_Saturated = orc_Saturated
        self.orc_no_Rec = orc_no_Rec
        self.max_pump_dP = max_pump_dP
        self.eta_pump = eta_pump
        self.dp_dT_loss = dp_dT_loss
        self.dT_approach_cond = dT_approach_cond
        self.dT_pp_phe = dT_pp_phe
        self.dT_pp_rec = dT_pp_rec
        self.dT_ap_phe = dT_ap_phe
        self.dT_sh_phe = dT_sh_phe
        self.dT_sc_phe = dT_sc_phe
        self.eta_pump_orc = eta_pump_orc
        self.eta_me_pump_orc = eta_me_pump_orc
        self.eta_turbine_orc = eta_turbine_orc
        self.eta_me_turbine_orc = eta_me_turbine_orc
        self.eta_pump_co2 = eta_pump_co2
        self.eta_turbine_co2 = eta_turbine_co2
        self.cooling_mode = cooling_mode
        self.condenser_type = condenser_type
        self.T_cooling_water_in = T_cooling_water_in
        self.dT_cooling = dT_cooling
        self.dT_pp_cond = dT_pp_cond
        self.dp_water_condenser = dp_water_condenser
        self.dp_air_condenser = dp_air_condenser
        self.eta_fan = eta_fan
        self.dT_ct = dT_ct
        self.dT_pp_star_ct = dT_pp_star_ct
        self.dT_water_ct = dT_water_ct
        self.cp_water = cp_water
        self.dP_ct = dP_ct
        self.eta_me_pump = eta_me_pump
        self.eta_hydr_pump = eta_hydr_pump
        self.rho_water = rho_water
        self.RH_in = RH_in
        #self.T_in_water_ct = T_in_water_ct
        self.dT_max = dT_max
        self.cost_year = cost_year
        self.success_rate = success_rate
        self.F_OM = F_OM
        self.discount_rate = discount_rate
        self.lifetime = lifetime
        self.capacity_factor = capacity_factor
        self.opt_mode = opt_mode
        self.g = g
        self.rho_rock = rho_rock
        self.c_rock = c_rock
        self.k_rock = k_rock
        self.useWellboreHeatLoss = useWellboreHeatLoss
        self.well_segments = well_segments
        self.epsilon = well_relative_roughness

    @property
    def transmissivity(self):
        return self.permeability * self.reservoir_thickness

