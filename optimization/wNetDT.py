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
# class ORCProblem(ElementwiseProblem):
#     def __init__(self, orc_cycle, initialState):
#         self.orc_cycle = orc_cycle
#         self.initialState = initialState
#         super().__init__(n_var=2,  # 2 variabili: dT_ap_phe, dT_sh_phe
#                          n_obj=1,  # 1 obiettivo: w_net
#                          n_constr=0,  # Nessun vincolo di disuguaglianza
#                          xl=np.array([5, 5]),  # Limiti inferiori per dT_ap_phe, dT_sh_phe
#                          xu=np.array([20, 30]))  # Limiti superiori per dT_ap_phe, dT_sh_phe
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         dT_ap_phe, dT_sh_phe = x
#
#         # Aggiorna i parametri dT_ap_phe e dT_sh_phe dentro orc_cycle
#         self.orc_cycle.params.dT_ap_phe = dT_ap_phe
#         self.orc_cycle.params.dT_sh_phe = dT_sh_phe
#
#         # Ottieni le temperature aggiornate dal ciclo ORC
#         temperatures = self.orc_cycle.get_temperatures()
#
#         # Calcola la temperatura massima del ciclo, cioè la temperatura di ingresso in turbina
#         T_max_orc = temperatures['T_in_turb']  # Temperatura di ingresso in turbina
#
#         # Calcolo della temperatura di evaporazione
#         T_boil_C = T_max_orc - dT_sh_phe
#
#         # Risolvo il ciclo ORC con i parametri forniti
#         results = self.orc_cycle.solve(initialState=self.initialState, T_boil_C=T_boil_C)
#
#         # Estrai il lavoro netto
#         w_net = results['w_net']
#         out["F"] = -w_net  # Minimizzare il negativo per massimizzare w_net

class ORCProblem(ElementwiseProblem):
    def __init__(self, orc_cycle, initialState):
        self.orc_cycle = orc_cycle
        self.initialState = initialState
        super().__init__(n_var=2,  # 2 variabili: dT_ap_phe, dT_sh_phe
                         n_obj=1,  # 1 obiettivo: w_net
                         n_constr=0,  # Nessun vincolo di disuguaglianza
                         xl=np.array([5, 5]),  # Limiti inferiori per dT_ap_phe, dT_sh_phe
                         xu=np.array([20, 30]))  # Limiti superiori per dT_ap_phe, dT_sh_phe

    def _evaluate(self, x, out, *args, **kwargs):
        dT_ap_phe, dT_sh_phe = x

        # Aggiorno i parametri dT_ap_phe e dT_sh_phe dentro orc_cycle
        self.orc_cycle.params.dT_ap_phe = dT_ap_phe
        self.orc_cycle.params.dT_sh_phe = dT_sh_phe

        # Iterazione per calcolare T_boil_C
        T_boil_C = 100  # Stima iniziale
        delta = 1  # Differenza iniziale per la convergenza
        tolerance = 0.01  # Tolleranza per la convergenza
        max_iterations = 50  # Numero massimo di iterazioni
        iteration = 0

        while delta > tolerance and iteration < max_iterations:
            # Risolvo il ciclo ORC con il valore corrente di T_boil_C
            results = self.orc_cycle.solve(initialState=self.initialState, T_boil_C=T_boil_C)

            # Ottengo le temperature aggiornate (inclusa T_in_turb)
            temperatures = self.orc_cycle.get_temperatures()

            # Calcolo il nuovo T_boil_C
            T_max_orc = temperatures['T_in_turb']
            #T_max_orc = self.state[6].T_C
            T_boil_C_new = T_max_orc - dT_sh_phe

            print(f"T_max_orc: {T_max_orc} °C")
            print(f"dT_ap_phe: {self.orc_cycle.params.dT_ap_phe} °C")
            print(f"dT_sh_phe: {self.orc_cycle.params.dT_sh_phe} °C")

            # Controllo la variazione
            delta = abs(T_boil_C_new - T_boil_C)
            T_boil_C = T_boil_C_new
            iteration += 1

        if iteration == max_iterations:
            raise RuntimeError("Iterazione per T_boil_C non convergente")

        # Estrai il lavoro netto
        w_net = results['w_net']
        out["F"] = -w_net  # Minimizzare il negativo per massimizzare w_net


# Creazione dell'oggetto ORCCycleTboil
params = SimulationParameters(orc_fluid='R245fa')
orc_cycle = ORCCycleTboil(params=params)
initialState = FluidState.getStateFromPT(1.e6, 150., 'water')

# Creazione del problema
problem = ORCProblem(orc_cycle, initialState)

# Definire l'algoritmo Pattern Search
algorithm = PatternSearch()

# Ottimizzazione con Pattern Search
res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1,
               x0=np.array([10, 15]))  # Imposto un punto iniziale esplicito

# Risolvo il ciclo ORC per i parametri ottimizzati (test dei risultati)
test_results = orc_cycle.solve(initialState=initialState, T_boil_C=orc_cycle.T[6] - res.X[0])
print(test_results)

# Stampa dei risultati
print("Best solution found: ")
print("dT_ap_phe: ", res.X[0], "°C")
print("dT_sh_phe: ", res.X[1], "°C")
print("Net Power:", -res.F, "W")  # Restituiamo la potenza netta positiva
