import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
#from utils.fluidState import FluidState

# Inizializzo i parametri con il fluido desiderato e creo il ciclo ORC passando i parametri configurati
params = SimulationParameters(orc_fluid='R245fa')
cycle = ORCCycleTboil(params=params)

# Definisco il problema di ottimizzazione
class ORCOptimizationProblem(Problem):   # è il mio problema di ottimizzazione
    def __init__(self, cycle):
        self.cycle = cycle  # Passo un'istanza di ORCCycleTboil
        super().__init__(
            n_var=2,  # Numero di variabili (T_boil_C e dT_sh_phe)
            n_obj=1,  # Numero di obiettivi
            n_constr=0,  # Numero di vincoli
            xl=np.array([80, 5]),  # Limiti inferiori delle variabili
            xu=np.array([120, 20])  # Limiti superiori delle variabili
        )

    def _evaluate(self, x, out, *args, **kwargs):
        T_boil_C = x[:, 0]  # Prima variabile: temperatura di evaporazione
        dT_sh_phe = x[:, 1]  # Seconda variabile: delta T di surriscaldamento
        print(dT_sh_phe)
        print(T_boil_C)

        # Calcolo la potenza netta usando il metodo del ciclo ORC
        w_net = np.array([self.cycle.calculate_w_net(t, dt) for t, dt in zip(T_boil_C, dT_sh_phe)])

        # Minimizzare il negativo della potenza netta
        out["F"] = -w_net

# Istanziare il problema con il ciclo configurato
problem = ORCOptimizationProblem(cycle=cycle)

# Configurare l'algoritmo PatternSearch
algorithm = PatternSearch()

# Risolvere il problema
res = minimize(problem,
               algorithm,
               verbose=True,
               seed=1)

# Stampare i risultati
print("Migliore soluzione trovata:")
print("Temperatura di evaporazione (T_boil_C):", res.X[0], "°C")
print("Delta T di surriscaldamento (dT_sh_phe):", res.X[1], "°C")
print("Massima potenza netta:", -res.F[0])














