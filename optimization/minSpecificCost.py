import numpy as np
import sys
import os
import traceback
import csv
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
from utils.fluidState import FluidState

# Lista delle temperature geotermiche (in °C)
geothermal_temperatures = list(range(90, 171, 10))  # Modifica qui l'intervallo delle temperature geotermiche

# Lista dei fluidi di lavoro
fluids = ['Isopentane', 'n-Pentane', 'Isobutane', 'n-Butane']

# Lista dei valori di m_geo (portata massica) da considerare
m_geo_values = list(range(20, 121, 10))  # Intervallo da 30 a 60 con passo di 10

# Dizionario per memorizzare i risultati
results_dict = {}

# Definizione del problema di ottimizzazione
class ORCProblem(ElementwiseProblem):
    def __init__(self, orc_cycle, initialState, m_geo):
        self.orc_cycle = orc_cycle
        self.initialState = initialState
        self.m_geo = m_geo  # Aggiungi m_geo come input non ottimizzabile
        super().__init__(n_var=4,  # 4 variabili ottimizzabili
                         n_obj=1,  # 1 obiettivo: il costo specifico da minimizzare
                         n_constr=0,  # Nessun vincolo di disuguaglianza
                         xl=np.array([5, 0.1, 2, 2]),  # Limiti inferiori per dT_ap_phe, dT_sh_phe, dT_pp_phe, dT_pp_rec
                         xu=np.array([65, 50, 20, 20]))  # Limiti superiori per dT_ap_phe, dT_sh_phe, dT_pp_phe, dT_pp_rec

    def _evaluate(self, x, out, *args, **kwargs):
        dT_ap_phe, dT_sh_phe, dT_pp_phe, dT_pp_rec = x

        try:
            # Calcola i risultati per il ciclo ORC includendo m_geo come parametro
            results = self.orc_cycle.solve(initialState=self.initialState,
                                           dT_ap_phe=dT_ap_phe,
                                           dT_sh_phe=dT_sh_phe,
                                           dT_pp_phe=dT_pp_phe,
                                           dT_pp_rec=dT_pp_rec,
                                           m_geo=self.m_geo)  # Passa m_geo al ciclo ORC
            T_reinj = results.T_d

            # Verifica che la temperatura di reiniezione non sia troppo bassa
            if T_reinj < 70:
                out["F"] = np.inf  # Assegna un valore infinito se la temperatura di reiniezione è troppo bassa
                return

            specific_cost = results.Specific_cost

            # Controlla se il costo specifico è NaN o inf
            if np.isnan(specific_cost) or np.isinf(specific_cost):
                out["F"] = np.inf  # Se specific_cost è NaN o inf, non considerare questa soluzione
                return

            out["F"] = specific_cost  # Minimizzare il costo specifico

        except Exception as e:
            # Gestisce qualsiasi eccezione che potrebbe verificarsi
            print(f"Errore durante il calcolo con dT_ap_phe={dT_ap_phe}, dT_sh_phe={dT_sh_phe}, dT_pp_phe={dT_pp_phe}, dT_pp_rec={dT_pp_rec}: {e}")
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

        # Iterazione su ogni valore di m_geo
        for m_geo in m_geo_values:
            print(f"  - Ottimizzazione per m_geo={m_geo} kg/s")

            try:
                # Stato iniziale del fluido geotermico (acqua) alla temperatura T_geo
                initialState = FluidState.getStateFromPT(1.e5, T_geo, 'water')  # Modifica qui il PT se necessario

                # Creazione del problema di ottimizzazione
                problem = ORCProblem(orc_cycle, initialState, m_geo)

                # Definire l'algoritmo Pattern Search
                algorithm = PatternSearch()

                # Eseguire l'ottimizzazione con Pattern Search
                res = minimize(problem,
                               algorithm,
                               verbose=False,
                               seed=1,
                               x0=np.array([30, 25, 10, 10]))  # Imposto un punto iniziale esplicito per le 4 variabili

                # Otteniamo i risultati dal ciclo ORC
                results = orc_cycle.solve(initialState=initialState,
                                          dT_ap_phe=res.X[0],
                                          dT_sh_phe=res.X[1],
                                          dT_pp_phe=res.X[2],
                                          dT_pp_rec=res.X[3],
                                          m_geo=m_geo)

                # Memorizzazione dei risultati
                results_dict[(fluido, T_geo, m_geo)] = {
                    "dT_ap_phe": float(res.X[0]),
                    "dT_sh_phe": float(res.X[1]),
                    "dT_pp_phe": float(res.X[2]),
                    "dT_pp_rec": float(res.X[3]),
                    "Specific_cost": float(res.F[0]),
                    "W_net": float(results.W_net),  # Potenza netta
                    "C_plant": float(results.C_plant)  # Costi totali
                }

                # Stampa dei risultati
                print(f"Temperatura geotermica: {T_geo}°C, m_geo: {m_geo} kg/s")
                print("Migliore soluzione trovata:")
                print(f"Potenza netta: {float(results.W_net):.2f} kW")  # Stampa della potenza netta
                print(f"Costi totali ORC: {float(results.C_plant):.2f} €")  # Stampa dei costi totali
                print(f"dT_ap_phe: {float(res.X[0]):.2f} °C")
                print(f"dT_sh_phe: {float(res.X[1]):.2f} °C")
                print(f"dT_pp_phe: {float(res.X[2]):.2f} °C")
                print(f"dT_pp_rec: {float(res.X[3]):.2f} °C")
                print(f"Costo specifico minimizzato: {float(res.F[0]):.2f} €/kW")

            except Exception as e:
                print(f"Errore nell'ottimizzazione per T_geo={T_geo}, m_geo={m_geo} con fluido {fluido}: {e}")
                traceback.print_exc()
                continue

# Salvataggio in un file CSV
gengeo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Risali di un livello
results_folder = os.path.join(gengeo_path, 'results')  # Aggiungi la cartella 'results'

# Percorso completo per il file CSV
output_csv_file = os.path.join(results_folder, 'orc_minSpecificCost_results.csv')  # Cambiato il nome del file

# Salvataggio in un file CSV (sovrascrive ogni volta)
with open(output_csv_file, mode='w', newline='') as file:  # newline='' serve per evitare che vengano aggiunte righe vuote tra le righe scritte
    writer = csv.writer(file)   # Creo un oggetto (writer) per scrivere i dati
    writer.writerow(['fluido', 'T_geo (°C)', 'm_geo (kg/s)', 'dT_ap_phe (°C)', 'dT_sh_phe (°C)', 'dT_pp_phe (°C)', 'dT_pp_rec (°C)', 'specific_cost (€/kW)', 'W_net (kW)', 'C_plant (€)'])  # Intestazione
    for (fluido, T_geo, m_geo), result in results_dict.items():  # Viene riempito il file
        writer.writerow([fluido, T_geo, m_geo, result["dT_ap_phe"], result["dT_sh_phe"], result["dT_pp_phe"], result["dT_pp_rec"], result["Specific_cost"], result["W_net"], result["C_plant"]])

print(f"File CSV saved in: {output_csv_file}")


# Inizializzazione delle matrici Xr, Yr e flux_st
Xr = np.array(geothermal_temperatures)  # Valori di temperatura geotermica (T_geo)
Yr = np.array(m_geo_values)  # Valori di portata massica (m_geo)

# Ciclo sui fluidi per generare un grafico per ciascuno
for fluido in fluids:
    print(f"\nGenerando grafico per fluido {fluido}...")

    # Crea una nuova figura per il fluido corrente
    plt.figure(figsize=(8, 6))

    # Creazione delle matrici per W_net, C_plant e Specific_cost
    flux_st_specific_cost = np.zeros((len(Xr), len(Yr)))  # Matrice per i costi specifici
    flux_st_W_net = np.zeros((len(Xr), len(Yr)))  # Matrice per la potenza netta
    flux_st_C_plant = np.zeros((len(Xr), len(Yr)))  # Matrice per i costi totali

    # Popolamento delle matrici con i valori ottenuti in precedenza
    for i, T_geo in enumerate(Xr):
        for j, m_geo in enumerate(Yr):
            # Recupera i risultati per ogni combinazione (T_geo, m_geo)
            if (fluido, T_geo, m_geo) in results_dict:
                flux_st_specific_cost[i, j] = results_dict[(fluido, T_geo, m_geo)]["Specific_cost"]
                flux_st_W_net[i, j] = results_dict[(fluido, T_geo, m_geo)]["W_net"]
                flux_st_C_plant[i, j] = results_dict[(fluido, T_geo, m_geo)]["C_plant"]
            else:
                flux_st_specific_cost[i, j] = np.nan  # Se non ci sono risultati, assegna NaN
                flux_st_W_net[i, j] = np.nan  # Se non ci sono risultati, assegna NaN
                flux_st_C_plant[i, j] = np.nan  # Se non ci sono risultati, assegna NaN

    # --- Contour plot per il costo specifico ---
    cp_specific_cost = plt.contourf(Yr, Xr, flux_st_specific_cost, levels=20, cmap='viridis', alpha=0.5)  # Contour plot
    contours_specific_cost = plt.contour(Yr, Xr, flux_st_specific_cost, levels=20, colors='black', linewidths=1)  # Linee nere per i contorni
    plt.clabel(contours_specific_cost, inline=True, fontsize=8, fmt="%.0f", inline_spacing=5)  # Etichetta dei contorni senza decimali
    plt.colorbar(cp_specific_cost, label='Costo specifico (€/kW)')  # Barra dei colori
    plt.title(f"Specific cost for {fluido}")
    plt.xlabel("(m_geo) [kg/s]")
    plt.ylabel("T(T_geo) [°C]")
    output_plot_specific_cost = os.path.join(results_folder, f'contour_plot_specific_cost_{fluido}.png')
    plt.savefig(output_plot_specific_cost)
    plt.show()
    print(f"Contour plot per il costo specifico salvato in: {output_plot_specific_cost}")

    # --- Contour plot per la potenza netta ---
    cp_W_net = plt.contourf(Yr, Xr, flux_st_W_net, levels=20, cmap='viridis', alpha=0.5)  # Contour plot
    contours_W_net = plt.contour(Yr, Xr, flux_st_W_net, levels=20, colors='black', linewidths=1)  # Linee nere per i contorni
    plt.clabel(contours_W_net, inline=True, fontsize=8, fmt="%.0f", inline_spacing=5)  # Etichetta dei contorni senza decimali
    plt.colorbar(cp_W_net, label='Potenza Netta (kW)')  # Barra dei colori
    plt.title(f"Net power for {fluido}")
    plt.xlabel("(m_geo) [kg/s]")
    plt.ylabel("(T_geo) [°C]")
    output_plot_W_net = os.path.join(results_folder, f'contour_plot_W_net_{fluido}.png')
    plt.savefig(output_plot_W_net)
    plt.show()
    print(f"Contour plot per la potenza netta salvato in: {output_plot_W_net}")

    # --- Contour plot per i costi totali ---
    cp_C_plant = plt.contourf(Yr, Xr, flux_st_C_plant, levels=20, cmap='viridis', alpha=0.5)  # Contour plot
    contours_C_plant = plt.contour(Yr, Xr, flux_st_C_plant, levels=20, colors='black', linewidths=1)  # Linee nere per i contorni
    plt.clabel(contours_C_plant, inline=True, fontsize=8, fmt="%.0f", inline_spacing=5)  # Etichetta dei contorni senza decimali
    plt.colorbar(cp_C_plant, label='Costi Totali ORC (k€)')  # Barra dei colori
    plt.title(f"Contorno dei Costi Totali ORC per {fluido}")
    plt.xlabel("Portata massica (m_geo) [kg/s]")
    plt.ylabel("Temperatura geotermica (T_geo) [°C]")
    output_plot_C_plant = os.path.join(results_folder, f'contour_plot_C_plant_{fluido}.png')
    plt.savefig(output_plot_C_plant)
    plt.show()
    print(f"Contour plot per i costi totali ORC salvato in: {output_plot_C_plant}")




