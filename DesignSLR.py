from pypulseq import Sequence
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.opts import Opts
from utils.rftools import slice_profile_bloch, plot_slice_profile
import numpy as np
from matplotlib import pyplot as plt
from math import pi, atan, ceil
from numpy import interp
from bloch.bloch import bloch
from mplcursors import cursor
from sigpy.mri.rf import slr

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
Trfmin       = 2e-3          # duration of one pulse segment [sec]
slice_thickness = 4e-3 # [m]
Nz = 20
Nkz = 26
presc_slab = Nz*slice_thickness
encoded_slab_thk = Nkz*slice_thickness


n = ceil((Trfmin/dt)/4)*4
Trf = n*dt
tb = 8
bw = tb/Trf

signal = slr.dzrf(n=n, tb=tb, ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False)

# Create an RF event
rf_res, gz = make_arbitrary_rf(
    signal=signal, slice_thickness=slice_thickness*Nz,
    bandwidth=bw, flip_angle=pi/3, 
    system=system, return_gz=True, use="excitation"
    )


seq = Sequence(system)
seq.add_block(rf_res, gz)


# seq.plot(time_disp="ms")

# Get the time waveform
waves,_,_,_,_ = seq.waveforms_and_times(True)
rf_waves_t = np.real(waves[3][0,:])
rf_waves = waves[3][1,:]
g_waves_t = waves[2][0,:]
g_waves = waves[2][1,:]

t_end = max(g_waves_t[-1], rf_waves_t[-1])
tt = np.arange(0, t_end, system.rf_raster_time) # [s]
b1 = 10e3*interp(tt, rf_waves_t, rf_waves)/system.gamma # [Hz] -> [G]
gg = 100*interp(tt, g_waves_t, g_waves)/system.gamma # [Hz/m] -> [G/cm]

# ---------------------------------------------------------------
# Bloch simulation of the saturation performance
# ---------------------------------------------------------------
dp = np.linspace(-slice_thickness*(Nz+5), slice_thickness*(Nz+5), 100)*100 # [cm]
df = np.array([0]) # np.linspace(-500, 500, 200) # [Hz]

t1 = 100 # [s]
t2 = 100 # [s]

mode = 0

mx_0 = np.zeros((dp.shape[0], df.shape[0]))
my_0 = np.zeros((dp.shape[0], df.shape[0]))
mz_0 = np.ones((dp.shape[0], df.shape[0]))

mx, my, mz = bloch(b1, gg.T, dt, t1, t2, df.T, dp, mode, mx_0, my_0, mz_0)

vialpos = np.array([-24, 24])*1e-3 # [m]

markers = {
    'Prescribed Slab': np.array([-presc_slab/2 ,presc_slab/2]), 
    'Encoded FoV': np.array([-encoded_slab_thk/2 ,encoded_slab_thk/2]), 
    'Approx. Vial Positions': vialpos
    }


plot_slice_profile(dp/100, np.sqrt(mx[:]**2 + my[:]**2), ylabelstr='|Mx, My|', markers=markers, zlimarr=np.array([-100, 100]))

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
cursor()

plt.show()
