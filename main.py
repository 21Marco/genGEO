from src.oRCCycleTboil import ORCCycleTboil
from utils.fluidState import FluidState
from models.simulationParameters import SimulationParameters

# Definisci i parametri per il ciclo ORC, con il fluido scelto
params = SimulationParameters(orc_fluid='n-Pentane')

# Crea un'istanza del ciclo ORC
orc_cycle = ORCCycleTboil(params=params)

def main():
    # Stato iniziale del fluido
    initialState = FluidState.getStateFromPT(1e6, 130., 'water')  # Utilizza 'water' come fluido geotermico
    print("Initial State:", initialState)

    # Itera su m_geo
    for m_geo in range(30, 61, 10):  # Itera m_geo da 30 a 60 con passo 10
        print(f"Eseguendo oRCCycleTboil con m_geo = {m_geo}")

        # Ora puoi chiamare il metodo solve sull'istanza
        results = orc_cycle.solve(initialState, dT_ap_phe=16.25, dT_sh_phe=3.84, m_geo=m_geo)

        print(f"Risultati per m_geo = {m_geo}:")
        print(f"Lavoro netto: {results.W_net} W")
        print(f"Costi specifici: {results.Specific_cost} W")

if __name__ == "__main__":
    main()

