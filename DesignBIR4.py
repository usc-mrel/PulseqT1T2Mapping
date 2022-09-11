from pypulseq import Sequence
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.opts import Opts
from utils.rftools import design_bir4
import numpy as np
from matplotlib import pyplot as plt
from math import pi, atan
from numpy import interp
from bloch.bloch import bloch


# Set system limits
system = Opts(
    max_grad = 15, grad_unit="mT/m",
    max_slew = 40, slew_unit="T/m/s",
    grad_raster_time =  10e-6, # [s] ( 10 us)
    rf_raster_time   =   1e-6, # [s] (  1 us)
    rf_ringdown_time =  10e-6, # [s] ( 10 us)
    rf_dead_time     = 100e-6, # [s] (100 us)
    adc_dead_time    =  10e-6, # [s] ( 10 us)
)


dt = system.rf_raster_time # [s]

# --------------------------------------------------------------------------
# Define parameters for a BIR-4 preparation
# --------------------------------------------------------------------------
T_seg     = 2e-3          # duration of one pulse segment [sec]
b1_max    = 15            # maximum RF amplitude [uT]
# dw_max    = 39.8e3        # maximum frequency sweep [Hz]
dw_max    = 39.8e3        # maximum frequency sweep [Hz]
zeta      = 15.2          # constant in the amplitude function [rad]
kappa     = atan(63.6)    # constant in the frequency/phase function [rad]
beta      = 90            # flip angle [degree]

signal = design_bir4(T_seg, b1_max, dw_max, zeta, kappa, beta, dt)

# Create an RF event
rf_res = make_arbitrary_rf(signal=signal, flip_angle=2*pi, system=system, return_gz=False, use="saturation")
rf_res.signal = signal


b1frac = np.linspace(0, 2, 200)
df     = np.linspace(-500, 500, 200) # [Hz]


gx_res = make_trapezoid(channel='x', area=4 * 128 * 1/0.3, system=system, delay=calc_duration(rf_res))
t_reset = calc_duration(rf_res, gx_res)

seq = Sequence(system)
seq.add_block(rf_res, gx_res)

# seq.plot(time_disp="ms")

# Get the time waveform
waves,_,_,_,_ = seq.waveforms_and_times(True)
rf_waves_t = np.real(waves[3][0,:])
rf_waves = waves[3][1,:]
g_waves_t = waves[0][0,:]
g_waves = waves[0][1,:]

t_end = max(g_waves_t[-1], rf_waves_t[-1])
tt = np.arange(0, t_end, system.rf_raster_time) # [s]
b1 = 10e3*interp(tt, rf_waves_t, rf_waves)/system.gamma # [Hz] -> [G]
gg = 100*interp(tt, g_waves_t, g_waves)/system.gamma # [Hz/m] -> [G/cm]

# ---------------------------------------------------------------
# Bloch simulation of the saturation performance
# ---------------------------------------------------------------
dp = 0 #np.arange(-10, 10, 0.1)

t1 = 100 # [s]
t2 = 100 # [s]

mode = 0

mx_0 = np.zeros((1, df.shape[0]))
my_0 = np.zeros((1, df.shape[0]))
mz_0 = np.ones((1, df.shape[0]))

mx = np.zeros((df.shape[0], 1, b1frac.shape[0]))
my = np.zeros((df.shape[0], 1, b1frac.shape[0]))
mz = np.zeros((df.shape[0], 1, b1frac.shape[0]))

for bi, b1f in enumerate(b1frac):
    mx[:,:,bi], my[:,:,bi], mz[:,:,bi] = bloch(b1f*b1, gg.T, dt, t1, t2, df.T, dp, mode, mx_0, my_0, mz_0)

print("Value at 0,0 = " + str(mz[np.argmin(np.abs(df)), 0, np.argmin(np.abs(b1frac-1))]))

plt.figure()
plt.pcolor(df, (b1frac), (np.abs(np.squeeze(mz).T)), cmap = 'gray', vmin=0, vmax=1)
plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel(r'$\Delta f$ [Hz]')
plt.ylabel(r'Relative $B_1$')

plt.figure()
ax1 = plt.subplot(3,1,1)
plt.plot(tt, np.abs(b1))
plt.ylabel(r'$|B_1|$ [G]')
plt.subplot(3,1,2, sharex=ax1)
plt.plot(tt, np.angle(b1))
plt.ylabel(r'$\angle B_1$ [rad]')
plt.subplot(3,1,3, sharex=ax1)
plt.plot(tt, gg)
plt.ylabel('G [G/cm]')
plt.xlabel('Time [s]')

plt.show()
