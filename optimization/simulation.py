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

initialState = FluidState.getStateFromPT(1.e6, 120., 'water')

# Creo l'oggetto ORCCycleTboil
cycle = ORCCycleTboil(params = params)

def simulation(initialState, params):
    # Intervalli di T_boil_C e dT_ap_phe
    dT_approach_values = np.arange(5, 45, 5)  # Valori con step di 5
    dT_superheating_values = np.arange(0, 20, 5)

    # Dizionario per salvare i risultati
    results_dict = {}

    # Itero sugli intervalli
    for dT_ap_phe in dT_approach_values:
        for dT_sh_phe in dT_superheating_values:

            # Simulazione per questi valori specifici di T_boil_C e dT_ap_phe
            results = cycle.solve(initialState, dT_ap_phe = dT_ap_phe, dT_sh_phe = dT_sh_phe)

            # Salvo il risultato di w_net per questa combinazione di T_boil_C e dT_ap_phe
            results_dict[(dT_ap_phe, dT_sh_phe)] = results['w_net']

    # Trova il massimo w_net e la relativa combinazione di parametri
    max_w_net = max(results_dict.values())
    max_params = [key for key, value in results_dict.items() if value == max_w_net]

    print(f"Maximum w_net: {max_w_net} obtained for parameters: {max_params}")

    # Metto i risultati in 'results'
    gengeo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Risali di un livello
    results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'

    # Percorso completo per il file CSV
    output_csv_file = os.path.join(results_folder, 'orc_cycle_results.csv')

    # Salvataggio in un file CSV (sovrascrive ogni volta)
    with open(output_csv_file, mode='w', newline='') as file:  # newline='' serve per evitare che vengano aggiunte righe vuote tra le righe scritte
        writer = csv.writer(file)   #creo un oggetto (writer) per scrivere i dati
        writer.writerow(['dT_ap_phe', 'dT_sh_phe', 'w_net'])  # Intestazione
        for (dT_ap_phe, dT_sh_phe), w_net in results_dict.items():  # Viene riempito il file
            writer.writerow([dT_ap_phe, dT_sh_phe, w_net])

    print(f"File CSV saved in: {output_csv_file}")

    # Salvataggio in formato JSON (sovrascrive ogni volta)
    output_json_file = os.path.join(results_folder, 'orc_cycle_results.json')

    # Crea una lista di dizionari (convertendo i valori a tipi nativi di Python)
    json_results = [
        {"dT_ap_phe": float(dT_ap_phe), "dT_sh_phe": float(dT_sh_phe), "w_net": float(w_net)}  # Converti a float o int se necessario
        for (dT_ap_phe, dT_sh_phe), w_net in results_dict.items()        # Per ogni coppia di dati, mi da la potenza
    ]

    # Scrivi i dati in formato JSON
    with open(output_json_file, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    print(f"File JSON saved in: {output_json_file}")

# Esegui la simulazione
simulation(initialState, params)