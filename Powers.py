import os
import csv
from src.oRCCycleTboil1 import ORCCycleTboil1
from utils.fluidState import FluidState
from models.simulationParameters import SimulationParameters

# Definisci i parametri per il ciclo ORC, con il fluido scelto
params = SimulationParameters(orc_fluid='Isobutane')

# Crea un'istanza del ciclo ORC
orc_cycle = ORCCycleTboil1(params=params)

def Powers():
    # Stato iniziale del fluido geotermico (richiesto da solve)
    initialState = FluidState.getStateFromPT(1e6, 150., 'water')  # Pressione 1 MPa, temperatura 150°C
    print("Initial State:", initialState)

    # Definisci il valore fisso di m_geo e il range di T_boil_C
    m_geo = 100  # Portata di massa fissata a 50 kg/s
    T_boil_C_values = range(60, 121, 10)  # Da 60°C a 120°C con passo di 10

    # Risali dal percorso corrente per raggiungere la cartella 'results'
    gengeo_path = os.path.dirname(__file__)  # Ottieni la cartella corrente
    results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'
    output_csv_file = os.path.join(results_folder, 'orc_Powers_results.csv')  # Percorso completo per il file CSV

    # Crea il file CSV
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m_geo (kg/s)', 'T_boil_C (°C)', 'W_net (W)', 'W_turb (W)', 'W_pump (W)', 'Q_cond (W)', 'm_orc_fluid (kg/s)'])  # Intestazione

        # Itera solo su T_boil_C con m_geo fisso
        for T_boil_C in T_boil_C_values:
            # Risolvi il ciclo ORC con i parametri richiesti
            results = orc_cycle.solve(initialState, m_geo=m_geo, T_boil_C=T_boil_C)

            # Scrivi i risultati nel file CSV
            writer.writerow([m_geo, T_boil_C, results.W_net, results.W_turbine, abs(results.W_pump),
                             abs(results.Q_condenser), results. m_orc_fluid])
            print(f"m_geo = {m_geo} kg/s, T_boil_C = {T_boil_C}°C, W_net = {results.W_net:.2f} W, "
                  f"W_turb = {results.W_turbine:.2f} W, W_pump = {abs(results.W_pump):.2f} W, Q_cond = {abs(results.Q_condenser):.2f} W, "
                  f"m_orc_fluid = {results.m_orc_fluid:.2f} kg/s ")

    print(f"File CSV salvato in: {output_csv_file}")

# Esegui la funzione Powers solo se questo file è eseguito come script principale
if __name__ == "__main__":
    Powers()

