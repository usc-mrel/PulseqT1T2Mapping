import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pypulseq.calc_duration import calc_duration
from numpy import interp
from numpy.typing import ArrayLike

def slice_profile_small_tip(rf, gz, dt: float, z_off: float = 0):

    rf_ft = np.abs(np.fft.fftshift(np.fft.fft(rf.signal, 4*rf.signal.shape[0])))
    freqz = np.fft.fftshift(np.fft.fftfreq(rf_ft.shape[0], dt))

    zz = freqz/gz.amplitude + z_off

    slc_prfl = rf_ft/rf_ft[int(rf_ft.shape[0]/2)]

    return (zz, slc_prfl)


def plot_slice_profile(zz, slc_prfl, ylabelstr: str = 'M [au]', markers: dict = None, zlimarr: ArrayLike = None):
    color_list = ['red', 'orange', 'green', 'blue', 'black']
    fig, ax = plt.subplots()
    ax.plot(zz*1e3, slc_prfl)

    lgd_str = ['Slice Profile']
    if markers is not None:

        idx = 0
        for marker_name, marker_pos in markers.items():
            ax.vlines(
                x=marker_pos*1e3, 
                ymin=[0, 0], ymax=[1, 1], 
                linestyles='dashed', colors=[color_list[int(idx%len(color_list))]])

            lgd_str.append(marker_name)
            idx += 1

    ax.set_xlabel('z [mm]')
    ax.set_ylabel(ylabelstr)
    ax.set_title('Slice Profile')
    ax.legend(lgd_str)

    if zlimarr is not None:
        ax.set_xlim(zlimarr)

def slice_profile_bloch(rf, gz, gzr, dp: ArrayLike, dt: float):
    try:
        from bloch.bloch import bloch
    except ImportError:
        print("Bloch package is not installed, returning empty arrays.")
        return ([], [])

    b1_orig = rf.signal*np.exp(1j*2*pi*rf.freq_offset*rf.t) # [Hz]
    b1_t    = rf.t + rf.delay # [s]
    t_end   = calc_duration(rf, gz) + calc_duration(gzr) # [s]

    tt = np.arange(0, t_end, dt) # [s]
    b1 = 10e3*interp(tt, b1_t, b1_orig)/42.576e6 # [Hz] -> [G]

    gzamps  = 100*np.array([0, gz.amplitude, gz.amplitude, 0, gzr.amplitude, gzr.amplitude, 0])/42.576e6 # [Hz/m] -> [G/cm]
    gztimes = np.cumsum([0, gz.rise_time, gz.flat_time, gz.fall_time, gzr.rise_time, gzr.flat_time, gzr.fall_time]) + gz.delay # [s]

    gg = interp(tt, gztimes, gzamps) # [G]

    t1 = 1 # [s]
    t2 = 100e-3 # [s]
    df = 0

    mode = 2

    mx_0 = 0
    my_0 = 0
    mz_0 = 1

    mx, my, mz = bloch(b1, gg, dt, t1, t2, df, dp, mode, mx_0, my_0, mz_0)

    return (mx, my, mz, tt, b1, gg)


def sat_profile_bloch(rf, gspoil, dp: ArrayLike, dt: float):
    try:
        from bloch.bloch import bloch
    except ImportError:
        print("Bloch package is not installed, returning empty arrays.")
        return ([], [])

    b1_orig = rf.signal*np.exp(1j*2*pi*rf.freq_offset*rf.t) # [Hz]
    b1_t    = rf.t + rf.delay # [s]
    t_end   = calc_duration(rf, gspoil) # [s]

    tt = np.arange(0, t_end, dt) # [s]
    b1 = 10e3*interp(tt, b1_t, b1_orig)/42.576e6 # [Hz] -> [G]

    gzamps  = 100*np.array([0, gspoil.amplitude, gspoil.amplitude, 0])/42.576e6 # [Hz/m] -> [G/cm]
    gztimes = np.cumsum([gspoil.delay, gspoil.rise_time, gspoil.flat_time, gspoil.fall_time]) # [s]

    gg = interp(tt, gztimes, gzamps) # [G]

    t1 = 1 # [s]
    t2 = 100e-3 # [s]
    df = 0

    mode = 2

    mx_0 = 0 #np.ones((1, dp.shape[0])).T
    my_0 = 0 #np.zeros((1, dp.shape[0])).T
    mz_0 = 1 #np.zeros((1, dp.shape[0])).T

    mx, my, mz = bloch(b1, gg, dt, t1, t2, df, dp, mode, mx_0, my_0, mz_0)

    return (mx, my, mz, tt, b1, gg)

def design_bir4(T_seg, b1_max, dw_max, zeta, kappa, beta, dt) -> ArrayLike:
    '''design_bir4 -- Calculate a BIR-4 module
    
      Usage
        rf_bir4 = calculate_pulseq_BIR4_module(T_seg, b1_max, dw_max, zeta, kappa, beta, dt)
      Inputs
        T_seg       duration of one pulse segment [sec]
        b1_max      maximum RF amplitude [uT]
        dw_max      maximum frequency sweep [Hz]
        zeta        constant in the amplitude function [rad]
        kappa       constant in the frequency/phase function [rad]
        beta        flip angle [degree]
        dt          rf raster time
      Outputs
        rf_bir4     Pulseq RF event
    
      Description
        BIR-4 based B1 and B0 insensitive velocity selective pulse train described
        by Wong and Guo 2010 ISMRM
    
     Written by Nam Gyun Lee
     Adapted by Bilal Tasdelen
     Email: namgyunl@usc.edu, ggang56@gmail.com (preferred)
     Started: 09/08/2022, Last modified: 09/08/2022

     BIR-4 module
    --------------------------------------------------------------------------
    
              RF       RF       RF       RF 
           segment1 segment2 segment3 segment4
          |<------>|<------>|<------>|<------>|
           _______   _______|_______   _______         
          |       \ /       |       \ /       |
          |        |        |        |        |
          |        |        |        |        |
     o----+--------+--------+--------+--------+-> t
     |<-->|<------>|<------>|<------>|<------>|
     deadTime T_seg| T_seg    T_seg  | T_seg
    --------------------------------------------------------------------------
    '''

    ## Recalculate T_seg and Tp
    #--------------------------------------------------------------------------
    # Set the number of samples in one segment as an even number
    #--------------------------------------------------------------------------

    N_seg = np.floor(T_seg/dt)
    if N_seg % 2 == 1: # odd
        N_seg += 1

    T_seg = N_seg * dt

    #--------------------------------------------------------------------------
    # Calculate the duration of all segments [sec]
    #--------------------------------------------------------------------------
    Tp = 4 * T_seg

    ## Calculate a BIR-4 module
    rf_samples = N_seg * 4 # number of RF samples
    t = (np.arange(0, rf_samples) + 0.5) * dt # RRT: RF RASTER TIME

    #--------------------------------------------------------------------------
    # Calculate the maximum RF amplitude in [Hz]
    #--------------------------------------------------------------------------
    # [uT] * [Hz/T] * [T/1e6uT] => * 1e-6 [Hz]
    w1_max = b1_max * 42.57e6 * 1e-6 # [Hz]

    #--------------------------------------------------------------------------
    # Define dphi1 and dphi2
    #--------------------------------------------------------------------------
    dphi1 =  (180 + beta / 2) * pi / 180 # [rad]
    dphi2 = -(180 + beta / 2) * pi / 180 # [rad]

    #--------------------------------------------------------------------------
    # Calculate phi_max
    #--------------------------------------------------------------------------
    tan_kappa = np.tan(kappa)
    # [Hz] * [2pi rad/cycle] * [sec] / ([rad] * [unitless]) => [rad]
    phi_max = -(dw_max * 2 * pi) * Tp / (kappa * tan_kappa) * np.log(np.cos(kappa)) # [rad]

    #--------------------------------------------------------------------------
    # Define function handles for amplitude and phase functions
    #--------------------------------------------------------------------------
    w1  = lambda t: w1_max * np.tanh(zeta * (1 - 4 * t / Tp)) # [Hz]
    phi = lambda t: phi_max / 4 - (dw_max * 2 * pi) * Tp / (4 * kappa * tan_kappa) * np.log(np.cos(4 * kappa * t / Tp) / np.cos(kappa)) # [rad]

    #--------------------------------------------------------------------------
    # Segment 1 (0 < t <= 0.25 * Tp)
    #--------------------------------------------------------------------------
    segment1_range = (t <= 0.25 * Tp)
    t1 = t[segment1_range]
    am_segment1 = w1(t1)
    pm_segment1 = phi(t1)

    #--------------------------------------------------------------------------
    # Segment 2 (0.25 * Tp < t <= 0.5 * Tp)
    #--------------------------------------------------------------------------
    segment2_range = np.logical_and(t > 0.25 * Tp, t <= 0.5 * Tp)
    t2 = t[segment2_range]
    am_segment2 = w1(0.5 * Tp - t2)
    pm_segment2 = phi(0.5 * Tp - t2) + dphi1

    #--------------------------------------------------------------------------
    # Segment 3 (0.5 * Tp < t <= 0.75 * Tp)
    #--------------------------------------------------------------------------
    segment3_range = np.logical_and(t > 0.5 * Tp, t <= 0.75 * Tp)
    t3 = t[segment3_range]
    am_segment3 = w1(t3 - 0.5 * Tp)
    pm_segment3 = phi(t3 - 0.5 * Tp) + dphi1

    #--------------------------------------------------------------------------
    # Segment 4 (0.75 * Tp < t <= Tp)
    #--------------------------------------------------------------------------
    segment4_range = np.logical_and(t > 0.75 * Tp, t <= Tp)
    t4 = t[segment4_range]
    am_segment4 = w1(Tp - t4)
    pm_segment4 = phi(Tp - t4) + dphi1 + dphi2

    #--------------------------------------------------------------------------
    # Combine all segments to form BIR-4
    #--------------------------------------------------------------------------
    am = np.concatenate((am_segment1, am_segment2, am_segment3, am_segment4))
    pm = np.concatenate((pm_segment1, pm_segment2, pm_segment3, pm_segment4))
    rf_shape = am * np.exp(1j * pm)

    return rf_shape

def amfm2cplxsignal(am: ArrayLike, fm: ArrayLike, dwell: float) -> ArrayLike:
    # Adapted from pypulseq make_adiabatic_pulse.py
    pm = np.cumsum(fm) * dwell

    ifm = np.argmin(np.abs(fm))
    dfm = np.abs(fm)[ifm]

    if dfm == 0:
        pm0 = pm[ifm]
        am0 = am[ifm]
        roc_fm0 = np.abs(fm[ifm + 1] - fm[ifm - 1]) / 2 / dwell
    else:  # We need to bracket the zero-crossing
        if fm[ifm] * fm[ifm + 1] < 0:
            b = 1
        else:
            b = -1

        pm0 = (pm[ifm] * fm[ifm + b] - pm[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])
        am0 = (am[ifm] * fm[ifm + b] - am[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])
        roc_fm0 = np.abs(fm[ifm] - fm[ifm + b]) / dwell

    pm -= pm0
    a = (roc_fm0 * 4) ** 0.5 / 2 / np.pi / am0

    signal = a * am * np.exp(1j * pm)

    return signal
