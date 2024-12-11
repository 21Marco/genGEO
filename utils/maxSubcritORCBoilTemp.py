# Licensed under LGPL 2.1, please see LICENSE for details
# https://www.gnu.org/licenses/lgpl-2.1.html
#
# The work on this project has been performed at the GEG Group at ETH Zurich:
# --> https://geg.ethz.ch
#
# The initial version of this file has been implemented by:
#
#     Philipp Schaedle (https://github.com/philippschaedle)
#     Benjamin M. Adams
#
# Further changes are done by:
#

############################
import numpy as np

from utils.fluidState import FluidState

def maxSubcritORCBoilTemp(orc_fluid):    #Funzione che calcola la temperatura massima di ebollizione subcritica
    P_crit = FluidState.getPcrit(orc_fluid)  #calcola Pcrit

    # assume min pressure is condensing at 0C
    P_min = FluidState.getStateFromTQ(0, 1, orc_fluid).P_Pa #calcola Pmin he è la P di condensazione (supposta) a 0°C e in condiz di vap saturo
    dP = (P_crit-P_min)/1000  #crea un passo

    # find pressure with maximum entropy
    P = np.arange(P_min, P_crit, dP)   #creazione array di pressione dalla minima alla critica con passo dP
    s = FluidState.getStateFromPQ(P, 1, orc_fluid).s_JK   #calcolo dell'entropia per ogni pressione nelle condiz di vap saturo

    #Trova la pressione corrispondente all'entropia massima
    max_s = np.max(s)
    row_index = np.argmax(s, axis=0)
    max_P = P[row_index]

    return FluidState.getStateFromPS(max_P, max_s, orc_fluid).T_C   #Restituisce la temperatura corrispondente alla pressione ed entropia max
