import sys
import unittest

from tests.semiAnalyticalWellTest import *
from tests.reservoirDepletionTest import *
from tests.oRCCycleTboilTest import *
from tests.oRCCycleSupercritPboilTest import *
from tests.heatExchangerTest import *
from tests.fluidSystemWaterTest import *
from tests.fluidSystemCO2Test import *

def testSuite(full=False):
    suite = unittest.TestSuite()
    # semi-analytical well
    suite.addTest(SemiAnalyticalWellTest('testProductionWell'))
    suite.addTest(SemiAnalyticalWellTest('testInjectionWellWater'))
    suite.addTest(SemiAnalyticalWellTest('testInjectionWellCO2'))
    # reservoir
    suite.addTest(ReservoirDepletionTest('testDepletionCurve'))
    suite.addTest(ReservoirDepletionTest('testNoTransient'))
    suite.addTest(ReservoirDepletionTest('testTransientPNoT'))
    suite.addTest(ReservoirDepletionTest('testTransientPT'))
    suite.addTest(ReservoirDepletionTest('testTransientTNoP'))
    # ORC cycles
    suite.addTest(ORCCycleTboilTest('testParasiticPowerFraction'))
    suite.addTest(ORCCycleTboilTest('testORCCycleTboil'))
    suite.addTest(ORCCycleSupercritPboilTest('testORCCycleSupercritPboil'))
    suite.addTest(HeatExchangerTest('testHeatExchanger'))
    suite.addTest(HeatExchangerTest('testHeatExchangerOptMdot'))
    # fluidsystem Water
    suite.addTest(FluidSystemWaterTest('testFluidSystemWaterMdot1'))
    suite.addTest(FluidSystemWaterTest('testFluidSystemWaterSolverMdot1'))
    suite.addTest(FluidSystemWaterTest('testFluidSystemWaterMdot40'))
    suite.addTest(FluidSystemWaterTest('testFluidSystemWaterSolverMdot40'))
    # fluidsystem CO2
    suite.addTest(FluidSystemCO2Test('testFluidSystemCO2Mdot10'))
    suite.addTest(FluidSystemCO2Test('testFluidSystemCO2Mdot80'))
    suite.addTest(FluidSystemCO2Test('testFluidSystemCO2Mdot200'))
    suite.addTest(FluidSystemCO2Test('testFluidSystemCO2Mdot57_4'))
    # heavy tests only if full test is run
    if full:
        suite.addTest(FluidSystemWaterTest('testFluidSystemWaterSolverOptMdot'))
        suite.addTest(FluidSystemCO2Test('testFluidSystemCO2SolverOptMdot'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    if sys.argv[-1] == 'FULL':
        runner.run(testSuite(full=True))
    else:
        runner.run(testSuite())
