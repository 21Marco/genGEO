import numpy as np

from utils.fluidStateFromPT import FluidStateFromPT
from utils.solver import Solver

class FluidSystemWaterOutput(object):
    """FluidSystemWaterOutput."""
    pass

class FluidSystemWater(object):
    """FluidSystemWater provides methods to compute the water fluid cycle."""

    def __init__(self, params):

        self.params = params

        self.fluid = 'Water'
        self.injection_well = None
        self.reservoir = None
        self.production_well1 = None
        self.pump = None
        self.pp = None

    def solve(self, initial_state):

        results = FluidSystemWaterOutput()

        injection_state = FluidStateFromPT(initial_state.P_Pa(), initial_state.T_C(), initial_state.fluid)
        # Find necessary injection pressure
        dP_downhole = np.nan
        dP_solver = Solver()
        dP_loops = 1
        stop =  False

        while np.isnan(dP_downhole) or abs(dP_downhole) > 10e3:
            results.injection_well    = self.injection_well.solve(injection_state)
            results.reservoir         = self.reservoir.solve(results.injection_well.state)

            # if already at P_system_min, stop looping
            if stop:
                break

            # find downhole pressure difference (negative means overpressure)
            dP_downhole = self.params.P_reservoir() - results.reservoir.state.P_Pa()
            injection_state.P_Pa_in = dP_solver.addDataAndEstimate(injection_state.P_Pa(), dP_downhole)

            if np.isnan(injection_state.P_Pa()):
                injection_state.P_Pa_in = initial_state.P_Pa() + dP_downhole

            if dP_loops > 10:
                print('GenGeo::Warning:FluidSystemWater:dP_loops is large: %s'%dP_loops)
            dP_loops += 1

            # Set Limits
            if injection_state.P_Pa() < self.params.P_system_min():
                # can't be below this temp or fluid will flash
                injection_state.P_Pa_in = self.params.P_system_min()
                # switch stop to run injection well and reservoir once more
                stop = True

        if results.reservoir.state.P_Pa() >= self.params.P_reservoir_max():
            raise Exception('GenGeo::FluidSystemWater:ExceedsMaxReservoirPressure - '
                        'Exceeds Max Reservoir Pressure of %.3f MPa!'%(self.params.P_reservoir_max()/1e6))

        results.production_well1  = self.production_well1.solve(results.reservoir.state)
        results.pump              = self.pump.solve(results.production_well1.state, injection_state.P_Pa())
        results.pp                = self.pp.solve(results.pump.well.state)

        return results
