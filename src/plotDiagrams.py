import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from utils.fluidState import FluidState
import sys


def TsDischarge(state, array_T, array_s, orc_fluid, orc_no_Rec, DH_water=None, arrangement_cond="counterflow"): #, info_sim=None
    fig, ax = plt.subplots(1, 1) # creazione della figura e degli assi
    dT_min = 10
    T_min = min(array_T) - dT_min
    color_fluid = 'red'
    color_rec = 'green'
    color_sat = 'gray'
    color_water = 'blue'

    # Plot saturation curve
    sat_points = 100
    T_crit = FluidState.getTcrit(orc_fluid)
    tol = 0.001
    T_array_sat = np.linspace(T_min, T_crit - tol, sat_points)

    # Inizializzo gli array per le entropie
    s_array_liq = np.zeros(sat_points)
    s_array_vap = np.zeros(sat_points)
    # Calcolo l'entropia per ogni punto di saturazione
    for i, T in enumerate(T_array_sat):
        # Stato del liquido saturo
        state_liq = FluidState.getStateFromTQ(T, 0, orc_fluid)  # Q=0 per liquido
        s_array_liq[i] = state_liq.s_JK  # Entropia del liquido

        # Stato del vapore saturo
        state_vap = FluidState.getStateFromTQ(T, 1, orc_fluid)  # Q=1 per vapore
        s_array_vap[i] = state_vap.s_JK  # Entropia del vapore

    ax.plot(s_array_liq, T_array_sat, linewidth=1, color=color_sat)
    ax.plot(s_array_vap, T_array_sat, linewidth=1, color=color_sat)

    # Plot discharging cycle
    cycle_points = 50
    seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    in_pump, in_rec_cold, in_eco, in_eva, in_sh, in_turb, in_rec_hot, in_desh, in_cond = seq
    streams_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for ii in seq:
        if ii == in_turb:
            h_array = np.linspace(state[ii].h_Jkg, state[ii + 1].h_Jkg, cycle_points)
            p_array = np.linspace(state[ii].P_Pa, state[ii + 1].P_Pa, cycle_points)
            fluid_state = FluidState.getStateFromPh(p_array, h_array, orc_fluid)
            T_array = fluid_state.T_C
            s_array = fluid_state.s_JK
        elif ii == in_pump:
            T_array = np.array([state[ii].T_C, state[ii + 1].T_C])   #creo un array di due valori, ingresso e uscita
            s_array = np.array([state[ii].s_JK, state[ii + 1].s_JK])
        elif ii == in_cond:
            h_array = np.linspace(state[ii].h_Jkg, state[8].h_Jkg, cycle_points)
            p_array = np.linspace(state[ii].P_Pa, state[8].P_Pa, cycle_points)
            fluid_state = FluidState.getStateFromPh(p_array, h_array, orc_fluid)
            T_array = fluid_state.T_C
            s_array = fluid_state.s_JK
        else:
            h_array = np.linspace(state[ii].h_Jkg, state[ii + 1].h_Jkg, cycle_points)
            p_array = np.linspace(state[ii].P_Pa, state[ii + 1].P_Pa, cycle_points)
            fluid_state = FluidState.getStateFromPh(p_array, h_array, orc_fluid)
            T_array = fluid_state.T_C
            s_array = fluid_state.s_JK

        ax.plot(s_array, T_array, linewidth=2, color=color_fluid)
        ax.scatter(s_array[0], T_array[0], 12, color=color_fluid)
        ax.text(s_array[0] - 0.01, T_array[0], streams_labels[ii], color=color_fluid, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')

    if not orc_no_Rec:
        # Recuperator hot side
        T_array_rec_hot = np.array([state[in_rec_hot].T_C, state[in_eco].T_C])
        s_array_rec_hot = np.array([state[in_rec_hot].s_JK, state[in_eco].s_JK])
        ax.plot(s_array_rec_hot, T_array_rec_hot, linewidth=1, linestyle='--', color=color_rec)

        # Recuperator cold side
        T_array_rec_cold = np.array([state[in_desh].T_C, state[in_rec_cold].T_C])
        s_array_rec_cold = np.array([state[in_desh].s_JK, state[in_rec_cold].s_JK])
        ax.plot(s_array_rec_cold, T_array_rec_cold, linewidth=1, linestyle='--', color=color_rec)


    if DH_water is not None:
        # District heating water
        if arrangement_cond == "parallel":
            T_water_array = np.array([DH_water["T_DH_out"][0], DH_water["T_DH_in"][0]])
            labels_water = ["DH,out", "DH,in"]
        elif arrangement_cond == "counterflow":
            T_water_array = np.array([DH_water["T_DH_in"][0], DH_water["T_DH_out"][0]])
            labels_water = ["DH,in", "DH,out"]
        else:
            sys.exit("condenser HX arrangement is not defined")
        s_water_array = np.array([state[0].s_JK, state[in_desh].s_JK])
        # s_water_array = np.array([tdn_streams[0, 4], tdn_streams[in_desh, 4]])
        ax.plot(s_water_array, T_water_array, linewidth=1, color=color_water)
        ax.scatter(s_water_array, T_water_array, 10, color=color_water)
        ax.text(s_water_array[0] - 0.02, T_water_array[0], labels_water[0], color=color_water, fontsize=7,
                verticalalignment='bottom', horizontalalignment='right')
        ax.text(s_water_array[1] + 0.02, T_water_array[1], labels_water[1], color=color_water, fontsize=7,
                verticalalignment='top', horizontalalignment='left')

    # set axis and title
    ax.set_xlabel("Specific entropy [kJ/kgK]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title('T-s diagram - Discharge') #+ ' - n°sim: ' + str(info_sim))
    plt.grid(linestyle="--")
    plt.show()
    return fig

# noinspection PyDefaultArgument
def PlotTQHX(HXs, HX_names=['evaporator', 'condenser', 'recuperator'], mode_op='discharge', info_sim=None):
    fig, ax = plt.subplots(1, 3)
    nn = 0

    for ii in HX_names:
        T1 = HXs[ii]['T1'][0]
        T2 = HXs[ii]['T2'][0]
        fld1 = HXs[ii]['fluid1'][0]
        fld2 = HXs[ii]['fluid2'][0]
        Q = HXs[ii]['Q_sections'][0]

        # Aggiungo il preriscaldatore (preheater) se sto gestendo l'evaporatore
        if 'preheater' in HXs.keys() and ii == "evaporator":
            T1 = np.array([T1, HXs['preheater']['T1'][0]])
            T2 = np.array([T2, HXs['preheater']['T2'][0]])
            Q = np.append(Q, HXs['preheater']['Q_sections'][0])

        # Aggiungo il super riscaldatore se sto gestendo l'evaporatore
        if 'superheater' in HXs.keys() and ii == "evaporator":
            T1 = np.append(T1, HXs['superheater']['T1'][0])
            T2 = np.append(T2, HXs['superheater']['T2'][0])
            Q = np.append(Q, HXs['superheater']['Q_sections'][0])

        # Aggiungo il desuperheater se sto gestendo il condensatore
        if 'desuperheater' in HXs.keys() and ii == "condenser":
            T1 = np.append(T1, HXs['desuperheater']['T1'][0])
            T2 = np.append(T2, HXs['desuperheater']['T2'][0])
            Q = np.append(Q, HXs['desuperheater']['Q_sections'][0])

        PlotTQSingle(T1, T2, Q, fld1, fld2, HX_names[nn], mode_op, ax[nn], fig, info_sim)
        nn += 1
        plt.show()
    return fig

def PlotTQSingle(T1, T2, Q, fld1, fld2, HX_names, arrangement_HX, mode_op, curr_ax, fig,
                 info_sim=None):  # Plot T-q fro HXs

    str_RP = 'REFPROP::'

    if str_RP in fld1:
        fld1 = fld1.replace(str_RP, '')
    if str_RP in fld2:
        fld2 = fld2.replace(str_RP, '')
    if len(fld1) > 10:
        fld1 = fld1[:9] + "."
    if len(fld2) > 10:
        fld2 = fld2[:9] + "."

    if mode_op == "discharge" and HX_names == "recuperator":
        T1 = T1[::-1]
        T2 = T2[::-1]

    if all(T1 > T2):
        TH = T1
        fldH = fld1
        TC = T2
        fldC = fld2
    else:
        TH = T2
        fldH = fld2
        TC = T1
        fldC = fld1

    if arrangement_HX == 'counterflow':
        arrHX = 'cf.'  # Counterflow
    else:
        arrHX = 'cc.'  # Co-current (parallel)

    n_sections = len(TH)
    Q_cumul = np.concatenate(([0], np.cumsum(Q * np.ones(n_sections)))) #Q_cumul = np.concatenate(([0], np.cumsum(Q * np.ones(n_sections - 1))))
    if mode_op == "discharge":
        fig.suptitle("T-Q chart of Heat Exchangers in Discharge Mode" + ' - n°sim: ' + str(info_sim))
    elif mode_op == "charge":
        fig.suptitle("T-Q chart of Heat Exchangers in Charge Mode" + ' - n°sim: ' + str(info_sim)) # si possono togliere?

    lH, = curr_ax.plot(Q_cumul, TH)
    lC, = curr_ax.plot(Q_cumul, TC)
    lH.set_color('red')
    lC.set_color('Blue')
    curr_ax.set_title(HX_names + " - " + arrHX)
    curr_ax.legend([lH, lC], [fldH, fldC])
    curr_ax.set_xlabel('Thermal power [kW]')
    curr_ax.set_ylabel('Temperature [°C]')
    curr_ax.set_xlim(left=0)
    curr_ax.grid(linestyle="--")
    fig.tight_layout()
    plt.show()
    return

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.show()
    return


def save_figs(foldername, filename_list, fig_handle_list, ext='svg', close_saved_fig=True):
    ii=0

    for fig_handle in fig_handle_list:
        fig_handle.set_size_inches(7, 4)
        fig_handle.savefig(foldername + filename_list[ii] + "." + ext, bbox_inches='tight')
        if close_saved_fig==True:
            plt.close(fig_handle)
        ii+=1
    plt.show()
    return