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
