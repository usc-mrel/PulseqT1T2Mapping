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

    
