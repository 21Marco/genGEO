import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
from utils.fluidState import FluidState

# Definizione del problema di ottimizzazione
class ORCProblem(ElementwiseProblem):
    def __init__(self, orc_cycle, initialState):
        self.orc_cycle = orc_cycle
        self.initialState = initialState
        super().__init__(n_var=2,  # 2 variabili: dT_ap_phe, dT_sh_phe
                         n_obj=1,  # 1 obiettivo: w_net
                         n_constr=0,  # Nessun vincolo di disuguaglianza
                         xl=np.array([5, 0]),  # Limiti inferiori per dT_ap_phe, dT_sh_phe
                         xu=np.array([40, 20]))  # Limiti superiori per dT_ap_phe, dT_sh_phe

    def _evaluate(self, x, out, *args, **kwargs):
        dT_ap_phe, dT_sh_phe = x

        # Risolvo il ciclo ORC con i parametri forniti
        results = self.orc_cycle.solve(initialState=self.initialState, dT_ap_phe=dT_ap_phe, dT_sh_phe=dT_sh_phe)

        w_net = results.w_net

        out["F"] = -w_net  # Minimizzare il negativo per massimizzare w_net

# Creazione dell'oggetto ORCCycleTboil
params = SimulationParameters(orc_fluid='R245fa')
orc_cycle = ORCCycleTboil(params=params)
initialState = FluidState.getStateFromPT(1.e6, 160., 'water')

# Creazione del problema
problem = ORCProblem(orc_cycle, initialState)

# Definire l'algoritmo Pattern Search
algorithm = PatternSearch()

# Ottimizzazione con Pattern Search
res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1,
               x0=np.array([10, 15]))  # Imposto un punto iniziale esplicito per dT_ap_phe e dT_sh_phe, altrimenti prende un valore casuale tra i limiti imposti

# Stampa dei risultati
print("Best solution found: ")
print("dT_ap_phe: ", res.X[0], "°C")
print("dT_sh_phe: ", res.X[1], "°C")
print("Net Power:", -res.F, "W")  # Restituiamo la potenza netta positiva