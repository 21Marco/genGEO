import os
import csv
from src.oRCCycleTboil1 import ORCCycleTboil1
from utils.fluidState import FluidState
from models.simulationParameters import SimulationParameters

# Definisci i parametri per il ciclo ORC, con il fluido scelto
params = SimulationParameters(orc_fluid='Isobutane')

# Crea un'istanza del ciclo ORC
orc_cycle = ORCCycleTboil1(params=params)

def Teva():
    # Stato iniziale del fluido geotermico (richiesto da solve)
    initialState = FluidState.getStateFromPT(1e6, 150., 'water')  # Pressione 1 MPa, temperatura 150°C
    print("Initial State:", initialState)

    # Definisci i range di m_geo (portata di massa) e T_boil_C (temperatura di ebollizione)
    m_geo_values = range(100, 121, 5)  # Da 30 a 80 kg/s con passo di 10
    T_boil_C_values = range(70, 121, 10)  # Da 60°C a 120°C con passo di 10

    # Risali dal percorso corrente per raggiungere la cartella 'results'
    gengeo_path = os.path.dirname(__file__)  # Ottieni la cartella corrente
    results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'
    output_csv_file = os.path.join(results_folder, 'orc_Teva_results.csv')  # Percorso completo per il file CSV

    # Crea il file CSV
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m_geo (kg/s)', 'T_boil_C (°C)', 'W_net (W)', 'w_turbine_orc (J/kg)', 'w_pump_orc (J/kg)'
                         'q_desuperheater_orc (J/kg)', 'm_orc_fluid (kg/s)'])  # Intestazione

        # Ciclo annidato per calcolare W_net per ogni combinazione di m_geo e T_boil_C
        for m_geo in m_geo_values:
            for T_boil_C in T_boil_C_values:
                # Risolvi il ciclo ORC con i parametri richiesti
                results = orc_cycle.solve(initialState, m_geo=m_geo, T_boil_C=T_boil_C)

                # Scrivi i risultati nel file CSV
                writer.writerow([m_geo, T_boil_C, results.W_net, results.w_turbine_orc,
                                 abs(results.q_desuperheater_orc), results.m_orc_fluid])
                print(f"m_geo = {m_geo} kg/s, T_boil_C = {T_boil_C}°C, W_net = {results.W_net:.2f} W, "
                      f"w_turbine_orc = {results.w_turbine_orc:.2f} J/kg, w_pump_orc = {abs(results.w_pump_orc):.2f} J/kg, q_desuperheater_orc = {abs(results.q_desuperheater_orc):.2f} J/kg, "
                      f"m_orc_fluid = {results.m_orc_fluid:.2f} kg/s")

    print(f"File CSV salvato in: {output_csv_file}")

# Esegui la funzione Teva solo se questo file è eseguito come script principale
if __name__ == "__main__":
    Teva()

