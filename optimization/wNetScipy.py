import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.optimize import minimize

from src.oRCCycleTboil import ORCCycleTboil
from models.simulationParameters import SimulationParameters
from utils.fluidState import FluidState


# Parametri di simulazione
params = SimulationParameters(orc_fluid='R245fa')

# Stato iniziale del fluido geotermico (acqua)
initialState = FluidState.getStateFromPT(1.e6, 150., 'water')


def optimization_function(params, orc_cycle, initialState):
    """
    Funzione obiettivo da ottimizzare: calcola la potenza netta negativa
    per un dato T_boil_C e dT_ap_phe.

    Parametri:
    - params: lista con T_boil_C e dT_ap_phe [T_boil_C, dT_ap_phe]
    - orc_cycle: oggetto della classe ORCCycleTboil
    - initialState: stato iniziale da passare alla funzione solve

    Ritorna:
    - Potenza netta negativa (perché si sta minimizzando)
    """
    T_boil_C, dT_ap_phe = params

    # Aggiorna il parametro dT_ap_phe dentro orc_cycle
    orc_cycle.params.dT_ap_phe = dT_ap_phe

    # Risolvi il ciclo ORC con i parametri forniti
    results = orc_cycle.solve(initialState=initialState, T_boil_C=T_boil_C)

    # Restituire la potenza netta negativa (per la minimizzazione)
    return -results['w_net']


def optimize_orc_cycle(orc_cycle, initialState):
    """
    Ottimizzazione del ciclo ORC per massimizzare la potenza netta (w_net).

    Parametri:
    - orc_cycle: oggetto della classe ORCCycleTboil
    - initialState: stato iniziale da passare alla funzione solve

    Ritorna:
    - Risultati ottimizzati (T_boil_C, dT_ap_phe)
    """

    bounds = [(70, 120), (5, 20)]  # Limiti per T_boil_C e dT_ap_phe

    initial_guess = [100, 10]  # Valori iniziali per T_boil_C e dT_ap_phe

    # Ottimizzazione usando il metodo di 'L-BFGS-B'
    result = minimize(optimization_function, initial_guess, args=(orc_cycle, initialState), bounds=bounds, method='L-BFGS-B')

    # Valori ottimizzati per T_boil_C e dT_ap_phe
    if result.success:
        optimal_params = result.x
        # Calcolo la potenza netta ottimizzata usando i parametri ottimizzati
        optimal_power = -result.fun
        print(f"Best solution found:")
        print(f"T_boil_C : {optimal_params[0]} °C")
        print(f"dT_ap_phe : {optimal_params[1]} °C")
        print(f"Net Power : {optimal_power:} W")
    else:
        print("Optimization failed")
        return None


# Creazione dell'oggetto ORCCycleTboil
orc_cycle = ORCCycleTboil(params=params)

# Esecuzione dell'ottimizzazione
optimized_params = optimize_orc_cycle(orc_cycle, initialState)
