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

# Definizione del problema di ottimizzazione ORC mono-obiettivo
class ORCProblem(ElementwiseProblem):
    def __init__(self, orc_cycle, initialState):
        self.orc_cycle = orc_cycle
        self.initialState = initialState
        super().__init__(n_var=2,  # 2 variabili: T_boil_C, dT_ap_phe
                         n_obj=1,  # 1 obiettivo: w_net
                         n_constr=0,  # Nessun vincolo di disuguaglianza
                         xl=np.array([80, 5]),  # Limiti inferiori per T_boil_C, dT_ap_phe
                         xu=np.array([120, 20]))  # Limiti superiori per T_boil_C, dT_ap_phe

    def _evaluate(self, x, out, *args, **kwargs):
        T_boil_C, dT_ap_phe = x

        # Aggiorna il parametro dT_ap_phe dentro orc_cycle
        self.orc_cycle.params.dT_ap_phe = dT_ap_phe

        # Risolvi il ciclo ORC con i parametri forniti
        results = self.orc_cycle.solve(initialState=self.initialState, T_boil_C=T_boil_C)

        # Potenza netta (w_net) come obiettivo
        w_net = results['w_net']

        # Restituisci l'obiettivo (potenza netta, massimizzata moltiplicando per -1 per minimizzare)
        out["F"] = -w_net  # Minimizzare il negativo per massimizzare w_net

# Creazione dell'oggetto ORCCycleTboil
params = SimulationParameters(orc_fluid='R245fa')
orc_cycle = ORCCycleTboil(params=params)
initialState = FluidState.getStateFromPT(1.e6, 150., 'water')

# Creazione del problema
problem = ORCProblem(orc_cycle, initialState)

# Definire l'algoritmo Pattern Search per mono-obiettivo
algorithm = PatternSearch()

# Esegui l'ottimizzazione con Pattern Search
res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1,
               x0=np.array([100, 10]))  # Imposta un punto iniziale esplicito per T_boil_C e dT_ap_phe, altrimenti prende un valore casuale tra i limiti imposti

# Stampa dei risultati
print("Best solution found: ")
print("T_boil_C: ", res.X[0], "°C")
print("dT_ap_phe: ", res.X[1], "°C")
print("Potenza netta:", -res.F, "W")  # Restituiamo la potenza netta positiva



