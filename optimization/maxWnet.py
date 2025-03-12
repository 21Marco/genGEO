import numpy as np
import sys
import os
import traceback
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
from utils.fluidState import FluidState

# Lista delle temperature geotermiche (in °C)
geothermal_temperatures = list(range(100, 171, 10))  # Modifica qui l'intervallo delle temperature geotermiche

# Lista dei fluidi di lavoro
fluids = ['Isopentane', 'n-Pentane', 'Isobutane', 'n-Butane']

# Dizionario per memorizzare i risultati
results_dict = {}

# Definizione del problema di ottimizzazione
class ORCProblem(ElementwiseProblem):
    def __init__(self, orc_cycle, initialState):
        self.orc_cycle = orc_cycle
        self.initialState = initialState
        super().__init__(n_var=2,  # 2 variabili: dT_ap_phe, dT_sh_phe
                         n_obj=1,  # 1 obiettivo: solo la potenza netta w_net
                         n_constr=0,  # Nessun vincolo di disuguaglianza
                         xl=np.array([5, 0.1]),  # Limiti inferiori per dT_ap_phe, dT_sh_phe
                         xu=np.array([65, 50]))  # Limiti superiori per dT_ap_phe, dT_sh_phe

    def _evaluate(self, x, out, *args, **kwargs):
        dT_ap_phe, dT_sh_phe = x

        try:
            # Calcola i risultati per il ciclo ORC
            results = self.orc_cycle.solve(initialState=self.initialState, dT_ap_phe=dT_ap_phe, dT_sh_phe=dT_sh_phe)
            T_reinj = results.T_d

            # Verifica che la temperatura di reiniezione non sia troppo bassa
            if T_reinj < 70:
                out["F"] = np.inf  # Assegna un valore infinito se la temperatura di reiniezione è troppo bassa
                return

            W_net = results.W_net

            # Controlla se la potenza netta è NaN o inf
            if np.isnan(W_net) or np.isinf(W_net):
                out["F"] = np.inf  # Se w_net è NaN o inf, non considerare questa soluzione
                return

            out["F"] = -W_net  # Minimizzare il negativo per massimizzare la potenza netta

        except Exception as e:
            # Gestisce qualsiasi eccezione che potrebbe verificarsi
            print(f"Errore durante il calcolo con dT_ap_phe={dT_ap_phe} e dT_sh_phe={dT_sh_phe}: {e}")
            traceback.print_exc()
            out["F"] = np.inf  # In caso di errore, assegna np.inf per evitare che la soluzione venga presa in considerazione


# Iterazione sui fluidi
for fluido in fluids:
    print(f"\n=== Ottimizzazione per fluido {fluido} ===")

    # Creazione dell'oggetto ORCCycleTboil con il fluido corrente
    params = SimulationParameters(orc_fluid=fluido)
    orc_cycle = ORCCycleTboil(params=params)

    # Iterazione sulle diverse temperature del fluido geotermico
    for T_geo in geothermal_temperatures:
        print(f"\n=== Ottimizzazione per temperatura geotermica {T_geo}°C ===")

        try:
            # Stato iniziale del fluido geotermico (acqua) alla temperatura T_geo
            initialState = FluidState.getStateFromPT(1.e5, T_geo, 'water')  # Modifica qui il PT se necessario

            # Creazione del problema di ottimizzazione
            problem = ORCProblem(orc_cycle, initialState)

            # Definire l'algoritmo Pattern Search
            algorithm = PatternSearch()

            # Eseguire l'ottimizzazione con Pattern Search
            res = minimize(problem,
                           algorithm,
                           verbose=False,
                           seed=1,
                           x0=np.array([20, 10]))  # Imposto un punto iniziale esplicito per dT_ap_phe e dT_sh_phe

            # Memorizzazione dei risultati
            results_dict[(fluido, T_geo)] = {
                "dT_ap_phe": float(res.X[0]),
                "dT_sh_phe": float(res.X[1]),
                "W_net": -float(res.F[0]),
            }

            # Stampa dei risultati
            print(f"Temperatura geotermica: {T_geo}°C")
            print("Migliore soluzione trovata:")
            print(f"dT_ap_phe: {float(res.X[0]):.2f} °C")
            print(f"dT_sh_phe: {float(res.X[1]):.2f} °C")
            print(f"Potenza netta massimizzata: {-float(res.F[0]):.2f} W")

        except Exception as e:
            print(f"Errore nell'ottimizzazione per T_geo={T_geo} con fluido {fluido}: {e}")
            traceback.print_exc()
            continue

# Salvataggio in un file CSV
gengeo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Risali di un livello
results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'

# Percorso completo per il file CSV
output_csv_file = os.path.join(results_folder, 'orc_maxWnet_results.csv')  # Cambiato il nome del file

# Salvataggio in un file CSV (sovrascrive ogni volta)
with open(output_csv_file, mode='w', newline='') as file:  # newline='' serve per evitare che vengano aggiunte righe vuote tra le righe scritte
    writer = csv.writer(file)   # Creo un oggetto (writer) per scrivere i dati
    writer.writerow(['fluido', 'T_geo (°C)', 'dT_ap_phe (°C)', 'dT_sh_phe (°C)', 'W_net (W)'])  # Intestazione
    for (fluido, T_geo), result in results_dict.items():  # Viene riempito il file
        writer.writerow([fluido, T_geo, result["dT_ap_phe"], result["dT_sh_phe"], result["W_net"]])

print(f"File CSV saved in: {output_csv_file}")
