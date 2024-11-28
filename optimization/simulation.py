import os
import csv
import json
import numpy as np
import sys

# Aggiungo il percorso della cartella genGEO per importare i moduli correttamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
from utils.fluidState import FluidState

# Configurazione parametri simulazione
params = SimulationParameters(orc_fluid = 'R245fa')

initialState = FluidState.getStateFromPT(1.e6, 150., 'water')

# Creo l'oggetto ORCCycleTboil
cycle = ORCCycleTboil(params = params)

def simulation(initialState, params):
    # Intervalli di T_boil_C e dT_ap_phe
    T_boil_values = np.arange(100, 121, 5)  # Valori da 100 a 120 (incluso), con step di 5
    dT_approach_values = np.arange(5, 21, 5)  # Valori da 5 a 20 (incluso), con step di 5

    # Dizionario per salvare i risultati
    results_dict = {}

    # Itero sugli intervalli
    for T_boil_C in T_boil_values:
        for dT_ap_phe in dT_approach_values:

            # Aggiorno il parametro dT_ap_phe nel ciclo, perchè in solve ho self.params.dT_ap_phe e qui l'ho chiamatpo semplicemente dT_ap_phe
            params.dT_ap_phe = dT_ap_phe

            # Simulazione per questi valori specifici di T_boil_C e dT_ap_phe
            results = cycle.solve(initialState, T_boil_C = T_boil_C, dT_ap_phe = dT_ap_phe)

            # Salvo il risultato di w_net per questa combinazione di T_boil_C e dT_ap_phe
            results_dict[(T_boil_C, dT_ap_phe)] = results['w_net']

    # Metto i risultati in 'results'
    gengeo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Risali di un livello
    results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'

    # Percorso completo per il file CSV
    output_csv_file = os.path.join(results_folder, 'orc_cycle_results.csv')

    # Salvataggio in un file CSV (sovrascrive ogni volta)
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['T_boil_C', 'dT_ap_phe', 'w_net'])  # Intestazione
        for (T_boil_C, dT_ap_phe), w_net in results_dict.items():
            writer.writerow([T_boil_C, dT_ap_phe, w_net])

    print(f"File CSV salvato in: {output_csv_file}")

    # Salvataggio in formato JSON (sovrascrive ogni volta)
    output_json_file = os.path.join(results_folder, 'orc_cycle_results.json')

    # Crea una lista di dizionari (convertendo i valori a tipi nativi di Python)
    json_results = [
        {"T_boil_C": float(T_boil_C), "dT_ap_phe": float(dT_ap_phe), "w_net": float(w_net)}  # Converti a float o int se necessario
        for (T_boil_C, dT_ap_phe), w_net in results_dict.items()
    ]

    # Scrivi i dati in formato JSON
    with open(output_json_file, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    print(f"File JSON salvato in: {output_json_file}")

# Esegui la simulazione
simulation(initialState, params)


















