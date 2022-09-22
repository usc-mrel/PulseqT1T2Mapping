
from utils.load_params import load_params
from utils.rftools import plot_slice_profile, slice_profile_bloch, slice_profile_small_tip
from pypulseq.opts import Opts
from math import ceil, pi
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_sinc_pulse import make_sinc_pulse
from sigpy.mri.rf import slr
import numpy as np
import matplotlib.pyplot as plt
from pypulseq.make_trapezoid import make_trapezoid



param_filename = "b1map_vfa"

slice_profile_sim = "Bloch" # 'Small Tip', 'Bloch', None

exc_pulse_type = 'slr' # 'slr', 'sinc'


# Load parameter file
params = load_params(param_filename)

Nx, Ny, Nz = params['matrix_size']
slice_thickness = params['slice_thickness']  # Slice thickness
Nkz = Nz + params['kz_os_steps']
z = params['slice_pos'][0]
alpha = params['flip_angle']
fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution

encoded_slab_thk = Nkz*slice_thickness

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

# Create alpha-degree slice selection pulse and gradient
tbwp = 8 # Time-BW product of sinc
Trf = 2e-3 # [s] RF duration
dt = system.rf_raster_time


# Create the excitation RF events
rf = []

if exc_pulse_type == 'slr':
    raster_ratio = int(system.grad_raster_time/system.rf_raster_time)
    n = ceil((Trf/dt)/(4*raster_ratio))*4*raster_ratio
    Trf = n*dt
    bw = tbwp/Trf
    signal = slr.dzrf(n=n, tb=tbwp, ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False)

    rf, gz = make_arbitrary_rf(
        signal=signal, slice_thickness=fov[2],
        bandwidth=bw, flip_angle=alpha[0] * np.pi / 180, 
        system=system, return_gz=True, use="excitation"
        )

    gzr = make_trapezoid(channel="z", area=-gz.area/2, system=system)

elif exc_pulse_type == 'sinc':
    # Calculate and store rf pulses
    rf, gz, gzr = make_sinc_pulse(
        flip_angle=alpha[0] * np.pi / 180,
        duration=Trf,
        slice_thickness=fov[2],
        apodization=0.5,
        time_bw_product=tbwp,
        system=system,
        return_gz=True,
        use="excitation"
    )

else:
    print('Wrong RF pulse.')
    exit()

rf.freq_offset = gz.amplitude*z

vialpos = (z - np.array([encoded_slab_thk/2-20*slice_thickness, encoded_slab_thk/2-40*slice_thickness])) # [m]

markers = {
    'Prescribed Slab': z+np.array([-fov[2]/2 ,fov[2]/2]), 
    'Encoded FoV': z+np.array([-encoded_slab_thk/2 ,encoded_slab_thk/2]), 
    'Approx. Vial Positions': vialpos
    }

if slice_profile_sim == 'Small Tip':

    # ------------------------
    # DEBUG: RF slice profile
    # Small-tip approx.
    # ------------------------

    zz,slc_prfl_sta = slice_profile_small_tip(rf, gz, system.rf_raster_time, z)

    plot_slice_profile(zz, slc_prfl_sta, markers=markers, zlimarr=z*1e3+np.array([-100, 100]))
    plt.show()
    
elif slice_profile_sim == 'Bloch':
    # ------------------------
    # DEBUG: RF slice profile
    # Bloch sim
    # ------------------------

    dp = (np.arange(-Nkz/2, Nkz/2)*slice_thickness + z)*100 # [cm]

    mx, my, mz, tt, b1, gg = slice_profile_bloch(rf, gz, gzr, dp, system.rf_raster_time)

    plt.figure()
    plt.subplot(211)
    plt.plot(tt*1e3, np.abs(b1))
    plt.ylabel("|RF| [uT]")
    plt.subplot(212)
    plt.plot(tt*1e3, gg*10)
    plt.ylabel("G [mT/m]")
    plt.xlabel("t [ms]")


    plot_slice_profile(dp/100, np.sqrt(mx[:,-1]**2 + my[:,-1]**2), ylabelstr='|Mx, My|', markers=markers, zlimarr=z*1e3+np.array([-100, 100]))
    try:
        import mplcursors
        mplcursors.cursor()
    except ImportError:
        pass 
    plt.show()