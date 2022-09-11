# -------------------------------------------------------------------
# Title: 3D Variable Flip Angle GRE, based on 3D GRE from pypulseq library.
# Author: Bilal Tasdelen <tasdelen at usc dot edu>
# -------------------------------------------------------------------

## TODO:

from math import pi
import sys
import numpy as np
import pypulseq as pp
import os
import matplotlib.pyplot as plt
from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.make_label import make_label
from pypulseq.make_delay import make_delay
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.convert import convert
from pypulseq.points_to_waveform import points_to_waveform
from utils.grad_timing import rnd2GRT
import json
from utils import seqplot
from utils.rftools import amfm2cplxsignal, slice_profile_small_tip, plot_slice_profile, slice_profile_bloch, sat_profile_bloch

# =============
# USER OPTIONS
# =============
plot = False 
write_seq = True
detailed_rep = False
slice_profile_sim = None # 'Small Tip', 'Bloch', None
reset_block = True

# param_filename = "t1map_both_vfagre"
# param_filename = "t1map_debug"
# param_filename = "t1map_MnCl2_vfagre"
# param_filename = "t1map_NiCl2_vfagre"
param_filename = "b1map_vfa"

# Load parameter file
params = {}

param_dir = "protocols"
with open(os.path.join(param_dir,param_filename + ".json"), "r") as fj:
    jsonstr = fj.read()
    params = json.loads(jsonstr)

# ======
# SETUP
# ======
Nx, Ny, Nz = params['matrix_size']
alpha = params['flip_angle']

Ndummy = params['Ndummy']

slice_thickness = params['slice_thickness']  # Slice thickness
fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
Nkz = Nz + params['kz_os_steps']
TE  = params['TE'][0]  # Echo time
TR  = params['TR']     # Repetition time

slice_orientation = params['slice_orientation'] # "TRA", "COR", "SAG" 
z = params['slice_pos'][0]

ro_duration = params['readout_duration']  # ADC duration

seq_filename = params['file_name']
seq_folder   = params['output_folder']

phi0 = 117  # RF spoiling increment


# Grad axes
if slice_orientation == "TRA":
    dir_ro, dir_pe, dir_ss = ("x", "y", "z")
elif slice_orientation == "COR":
    dir_ro, dir_pe, dir_ss = ("x", "z", "y")
elif slice_orientation == "SAG":
    dir_ro, dir_pe, dir_ss = ("z", "y", "x")

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

# ==============
# CREATE EVENTS
# ==============
# Create alpha-degree slice selection pulse and gradient
tbwp = 6 # Time-BW product of sinc
Trf = 2e-3 # [s] RF duration

# Calculate and store rf pulses
rf = []
rf_, gz, gzr = pp.make_sinc_pulse(
    flip_angle=alpha[0] * np.pi / 180,
    duration=Trf,
    slice_thickness=fov[2],
    apodization=0.5,
    time_bw_product=tbwp,
    system=system,
    return_gz=True,
    use="excitation"
)
rf_.freq_offset = gz.amplitude*z

rf.append(rf_)

for fa_i in range(1, len(alpha)):
    rf_ = pp.make_sinc_pulse(
        flip_angle=alpha[fa_i] * np.pi / 180,
        duration=Trf,
        slice_thickness=fov[2],
        apodization=0.5,
        time_bw_product=tbwp,
        system=system,
        return_gz=False,
        use="excitation"
    )
    rf_.freq_offset = gz.amplitude*z
    rf.append(rf_)

if slice_profile_sim is not None:
    encoded_slab_thk = Nkz*slice_thickness

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

    zz,slc_prfl_sta = slice_profile_small_tip(rf[0], gz, system.rf_raster_time, z)

    plot_slice_profile(zz, slc_prfl_sta, markers=markers, zlimarr=z*1e3+np.array([-100, 100]))
    plt.show()
    
elif slice_profile_sim == 'Bloch':
    # ------------------------
    # DEBUG: RF slice profile
    # Bloch sim
    # ------------------------

    dp = (np.arange(-Nkz/2, Nkz/2)*slice_thickness + z)*100 # [cm]

    mx, my, mz, tt, b1, gg = slice_profile_bloch(rf[-1], gz, gzr, dp, system.rf_raster_time)

    plt.figure()
    plt.subplot(211)
    plt.plot(tt*1e3, np.abs(b1))
    plt.subplot(212)
    plt.plot(tt*1e3, gg)
    plt.xlabel("t [ms]")
    try:
        import mplcursors
        mplcursors.cursor()
    except ImportError:
        pass 

    plot_slice_profile(dp/100, np.sqrt(mx[:,-1]**2 + my[:,-1]**2), ylabelstr='|Mx, My|', markers=markers, zlimarr=z*1e3+np.array([-100, 100]))

    plt.show()

# -----------------------------------------
# Define other gradients and ADC events
# -----------------------------------------

delta_k = 1 / fov[0]
gx = pp.make_trapezoid(
    channel=dir_ro, flat_area=Nx * delta_k, flat_time=ro_duration, system=system
)

# TODO: Oversampling.
adc = pp.make_adc(
    num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
)
gx_pre = pp.make_trapezoid(
    channel=dir_ro, area=-gx.area / 2, system=system
)

phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

# 3D Phase encode
if Nz == 1:
    areaZ = [-gz.area/2]
else:
    areaZ = -(np.arange(0,Nkz) - Nkz/2)/(Nkz*slice_thickness) - gz.area/2 # Combine kz encoding and rephasing

gz_enc_largest = pp.make_trapezoid(channel=dir_ss, area=np.max(np.abs(areaZ)), system=system)
Tz_enc = pp.calc_duration(gz_enc_largest)

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel=dir_ro, area=2 * Nx * delta_k, system=system)
# gx_spoil = pp.make_trapezoid(channel=dir_ro, area=-gx.area/2, system=system)

# gz_spoil = pp.make_trapezoid(channel=dir_ss, area=4 / slice_thickness, system=system)

# RESET Block
if reset_block:

    from pypulseq.make_arbitrary_rf import make_arbitrary_rf
    from utils.rftools import design_bir4
    from math import atan

    # --------------------------------------------------------------------------
    # Define parameters for a BIR-4 preparation
    # --------------------------------------------------------------------------
    T_seg     = 2e-3          # duration of one pulse segment [sec]
    b1_max    = 10            # maximum RF amplitude [uT]
    dw_max    = 39.8e3        # maximum frequency sweep [Hz]
    zeta      = 15.2          # constant in the amplitude function [rad]
    kappa     = atan(63.6)    # constant in the frequency/phase function [rad]
    beta      = 90            # flip angle [degree]

    signal = design_bir4(T_seg, b1_max, dw_max, zeta, kappa, beta, system.rf_raster_time)

    # Create an RF event
    rf_res = make_arbitrary_rf(signal=signal, flip_angle=2*pi, system=system, return_gz=False, use="saturation")
    rf_res.signal = signal

    gx_res = make_trapezoid(channel=dir_ro, area=4 * Nx * delta_k, system=system, delay=calc_duration(rf_res))
    t_reset = calc_duration(rf_res, gx_res)

else:
    t_reset = 0


# Calculate timing
TEd = rnd2GRT(
            TE
            - calc_duration(gz)/2
            - calc_duration(gx_pre, gz_enc_largest)
            - calc_duration(gx)/2
)
TRd = rnd2GRT(
            TR
            - TE
            - calc_duration(gz)/2
            - calc_duration(gx)/2
            - calc_duration(gx_spoil, gz_enc_largest, gx_pre)
            - t_reset
)

assert np.all(TEd >= 0), "Required TE can not be achieved."
assert np.all(TRd >= 0), "Required TR can not be achieved."

delay_TE = make_delay(TEd)
delay_TR = make_delay(TRd)

rf_phase = 0
rf_inc = 0

seq = pp.Sequence(system)  # Create a new sequence object

seq.add_block(pp.make_label(label="REV", type="SET", value=1))

# Register fixed grad events for speedup
gx_pre.id = seq.register_grad_event(gx_pre)
gx.id = seq.register_grad_event(gx)

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over slices
for fa_i in range(len(alpha)):
    seq.add_block(pp.make_label(type="SET", label="SET", value=fa_i))

    # Run several TRs for driving the magnetization to steady-state
    gz_enc = pp.make_trapezoid(channel=dir_ss,area=areaZ[int(Nkz/2)],duration=Tz_enc, system=system)
    gz_rew = pp.make_trapezoid(channel=dir_ss,area=-(areaZ[int(Nkz/2)] + gz.area/2),duration=Tz_enc, system=system)

    gy_pre = pp.make_trapezoid(channel=dir_pe,area=phase_areas[int(Ny/2)],duration=pp.calc_duration(gx_pre),system=system)

    for dm_i in range(Ndummy):

        rf[fa_i].phase_offset = 0.5*phi0*(dm_i*dm_i + dm_i + 2)*np.pi/180
        adc.phase_offset = rf[fa_i].phase_offset

        seq.add_block(rf[fa_i], gz)
        
        seq.add_block(delay_TE)

        seq.add_block(gx_pre, gy_pre, gz_enc)


        seq.add_block(gx)
        gy_pre.amplitude = -gy_pre.amplitude


        # seq.add_block(delay_TR, gx_spoil, gy_pre, gz_spoil)
        seq.add_block(gx_spoil, gy_pre, gz_rew)
        seq.add_block(delay_TR)


    for kz_i in range(Nkz):
        # Loop over phase encodes and define sequence blocks
        seq.add_block(make_label(type="SET", label="PAR", value=Nkz-kz_i-1))
        kz_phase_offset = kz_i*2*np.pi*z/(Nkz*slice_thickness) - 2*np.pi*rf[fa_i].freq_offset*calc_rf_center(rf[fa_i])[0] # align the phase for off-center slices

        gz_enc = pp.make_trapezoid(channel=dir_ss,area=areaZ[kz_i],duration=Tz_enc, system=system)
        gz_rew = pp.make_trapezoid(channel=dir_ss,area=-(areaZ[kz_i] + gz.area/2),duration=Tz_enc, system=system)

        gz_enc.id = seq.register_grad_event(gz_enc)
        gz_rew.id = seq.register_grad_event(gz_rew)
        n = 0

        for ky_i in range(Ny):
            seq.add_block(make_label(type="SET", label="LIN", value=ky_i))
            # n = ky_i#ky_i+kz_i*Ny
            n+=1
            rf[fa_i].phase_offset = 0.5*phi0*(n*n + n + 2)*np.pi/180 + kz_phase_offset
            adc.phase_offset = 0.5*phi0*(n*n + n + 2)*np.pi/180

            seq.add_block(rf[fa_i], gz)
            
            gy_pre = pp.make_trapezoid(
                channel=dir_pe,
                area=phase_areas[ky_i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            # seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(delay_TE)

            seq.add_block(gx_pre, gy_pre, gz_enc)

            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude

            # seq.add_block(delay_TR, gx_spoil, gy_pre, gz_spoil)
            seq.add_block(gx_spoil, gy_pre, gz_rew)
            
            if reset_block:
                rf_res.phase_offset = 0.5*phi0*(n*n + n + 2)*np.pi/180
                seq.add_block(rf_res, gx_res)

            seq.add_block(delay_TR)
            



ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# ================
# VISUALIZATION
# ===============
if plot:
    seqplot.plot(seq, time_range=((Ndummy+1)*TR, (Ndummy+21)*TR), time_disp="ms", grad_disp="mT/m", plot_now=True)
    # (k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc) = seq.calculate_kspace()
    # pass

    # seq.plot(time_range=(0, 256*TR), time_disp="ms", grad_disp="mT/m", plot_now=False)
    # seq.plot_logical(label="lin", time_range=(0, 4*TR), time_disp="ms", grad_disp="mT/m", plot_now=False)

    # mplcursors.cursor()
    # plt.show()

# from utils.export_waveforms import export_waveforms

# waveform_tr,_,_,_,_ = export_waveforms(seq,time_range=(50*TR, 51*TR), append_RF=True)

if detailed_rep:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)


# ===============
# WRITE .SEQ
# ===============
if write_seq:
    # Prepare the sequence output for the scanner
    seq.set_definition(key="FOV", value=fov)
    seq.set_definition(key="SliceThickness", value=fov[2])
    seq.set_definition(key="Name", value="vfagre3d")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=alpha)
    # seq.set_definition(key="OrientationMapping", value=slice_orientation)
    seq.set_definition(key="ReconMatrixSize", value=[Nx, Ny, Nz])

    seq_path = os.path.join(seq_folder, f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    from utils.seqinstall import seqinstall
    seqinstall(seq_path)

