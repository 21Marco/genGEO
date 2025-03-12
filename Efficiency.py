import os
import csv
from src.oRCCycleTboil1 import ORCCycleTboil1
from utils.fluidState import FluidState
from models.simulationParameters import SimulationParameters

# Definisci i parametri per il ciclo ORC, con il fluido scelto
params = SimulationParameters(orc_fluid='Isobutane')

# Crea un'istanza del ciclo ORC
orc_cycle = ORCCycleTboil1(params=params)

def efficiency():
    # Stato iniziale del fluido geotermico (richiesto da solve)
    initialState = FluidState.getStateFromPT(1e5, 150., 'water')  # Pressione 1 MPa, temperatura 150°C
    print("Initial State:", initialState)

    # Definisci il valore fisso di m_geo e il range di T_boil_C
    m_geo = 50  # Portata di massa fissata a 50 kg/s
    T_boil_C_values = range(85, 126, 5)  # Da 60°C a 120°C con passo di 10

    # Risali dal percorso corrente per raggiungere la cartella 'results'
    gengeo_path = os.path.dirname(__file__)  # Ottieni la cartella corrente
    results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'
    output_csv_file = os.path.join(results_folder, 'orc_efficiency_results.csv')  # Percorso completo per il file CSV

    # Crea il file CSV
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m_geo (kg/s)', 'T_boil_C (°C)', 'W_net (W)', 'Q_in (W)', 'cicle_efficiency (%)', 'Q_in_max (W)', 'plant_efficiency (%)'])  # Intestazione

        # Itera solo su T_boil_C con m_geo fisso
        for T_boil_C in T_boil_C_values:
            # Risolvi il ciclo ORC con i parametri richiesti
            results = orc_cycle.solve(initialState, m_geo=m_geo, T_boil_C=T_boil_C)

            # Scrivi i risultati nel file CSV
            writer.writerow([m_geo, T_boil_C, results.W_net, results.Q_in, results.cicle_efficiency, results.Q_in_max, results.plant_efficiency])
            print(f"m_geo = {m_geo} kg/s, T_boil_C = {T_boil_C}°C, W_net = {results.W_net:.2f} W, "
                  f"Q_in = {results.Q_in:.2f} W, cicle_efficiency = {results.cicle_efficiency:.2f} % "
                  f"Q_in_max = {results.Q_in_max:.2f} W, plant_efficiency = {results.plant_efficiency:.2f} % ")

    print(f"File CSV salvato in: {output_csv_file}")

# Esegui la funzione Powers solo se questo file è eseguito come script principale
if __name__ == "__main__":
    efficiency()