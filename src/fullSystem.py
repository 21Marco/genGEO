
from src.energyConversion import EnergyConversionORC
from src.energyConversion import EnergyConversionCPG

class FullSystemOutput(object):
    """FullSystemOutput."""
    pass


class FullSystemORC(object):
    """FullSystemORC."""

    def __init__(self, params, fluid_system_solver, capital_cost_model):
        self.params = params
        self.fluid_system_solver = fluid_system_solver
        self.capital_cost_model = capital_cost_model

    def solve(self, m_dot, time_years):
        self.m_dot = m_dot
        self.fluid_system_solver.solve(m_dot, time_years)
        self.energy_results = EnergyConversionORC.gatherOutput(self.params, self.m_dot, self.fluid_system_solver.fluid_system)
        self.capital_cost_model.energy_results = self.energy_results
        self.capital_cost_model.fluid_system = self.fluid_system_solver.fluid_system
        self.capital_cost_model.solve()

    def gatherOutput(self):
        output = FullSystemOutput()
        output.fluid_system_solver = self.fluid_system_solver.gatherOutput()
        output.energy_results = self.energy_results
        output.capital_cost_model = self.capital_cost_model.gatherOutput()
        return output


class FullSystemCPG(object):
    """FullSystemCPG."""

    def __init__(self, params, fluid_system_solver, capital_cost_model):
        self.params = params
        self.fluid_system_solver = fluid_system_solver
        self.capital_cost_model = capital_cost_model

    def solve(self, m_dot, time_years):
        self.m_dot = m_dot
        self.fluid_system_solver.solve(m_dot, time_years)
        self.energy_results = EnergyConversionCPG.gatherOutput(self.params, self.m_dot, self.fluid_system_solver)
        self.capital_cost_model.energy_results = self.energy_results
        self.capital_cost_model.fluid_system = self.fluid_system_solver
        self.capital_cost_model.solve()

    def gatherOutput(self):
        output = FullSystemOutput()
        output.fluid_system_solver = self.fluid_system_solver.gatherOutput()
        output.energy_results = self.energy_results
        output.capital_cost_model = self.capital_cost_model.gatherOutput()
        return output
