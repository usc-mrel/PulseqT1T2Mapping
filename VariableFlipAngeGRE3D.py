# -------------------------------------------------------------------
# Title: 3D Variable Flip Angle GRE, based on 3D GRE from pypulseq library.
# Author: Bilal Tasdelen <tasdelen at usc dot edu>
# -------------------------------------------------------------------

## TODO: AFI proper steady-state handling.

import os
from math import ceil, pi

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_delay import make_delay
from pypulseq.make_label import make_label
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from sigpy.mri.rf import slr

from utils import seqplot
from utils.grad_timing import rnd2GRT
from utils.load_params import load_params

# =============
# USER OPTIONS
# =============

# Load user options
user_opts = load_params("user_opts_vfagre", "./")

show_diag    = user_opts['show_diag']
write_seq    = user_opts['write_seq']
detailed_rep = user_opts['detailed_rep']
param_filename = user_opts["param_filename"]

# Load parameter file
params = load_params(param_filename)

# ======
# SETUP
# ======
Nx, Ny, Nz = params['matrix_size']
alpha = params['flip_angle']
exc_pulse_type = params['exc_pulse_type'] # 'slr', 'sinc'

Ndummy = params['Ndummy']

slice_thickness = params['slice_thickness']  # Slice thickness
fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
Nkz = Nz + params['kz_os_steps']
TE  = params['TE'][0]  # Echo time
TR  = params['TR']     # Repetition time
is_afi = params['afi']
if is_afi:
    TR2 = params['TR2']

slice_orientation = params['slice_orientation'] # "TRA", "COR", "SAG" 
z = params['slice_pos'][0]

ro_duration = params['readout_duration']  # ADC duration

seq_filename = params['file_name']
seq_folder   = params['output_folder']

reset_block = params['reset_block']

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

seq = pp.Sequence(system)  # Create a new sequence object

# ==============
# CREATE EVENTS
# ==============
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

    rf_, gz = make_arbitrary_rf(
        signal=signal, slice_thickness=fov[2],
        bandwidth=bw, flip_angle=alpha[0] * np.pi / 180, 
        system=system, return_gz=True, use="excitation"
        )

    gzr = make_trapezoid(channel=dir_ss, area=-gz.area/2, system=system)

elif exc_pulse_type == 'sinc':
    # Calculate and store rf pulses
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

else:
    print('Wrong RF pulse.')
    exit()

rf_.freq_offset = gz.amplitude*z
# rf_.id, _ = seq.register_rf_event(rf_)

rf.append(rf_)


for fa_i in range(1, len(alpha)):
    

    if exc_pulse_type == 'slr':
        rf_ = make_arbitrary_rf(
            signal=signal, 
            slice_thickness=fov[2],
            bandwidth=bw, 
            flip_angle=alpha[fa_i] * np.pi / 180, 
            system=system, 
            return_gz=False, 
            use="excitation"
        )

    elif exc_pulse_type == 'sinc':
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
    # rf_.id, _ = seq.register_rf_event(rf_)

    rf.append(rf_)

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

# ky Phase Encode
phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

gy_pre = []
gy_rew = []

for ky_i in range(Ny):
    gy_pre.append(make_trapezoid(
        channel=dir_pe,
        area=phase_areas[ky_i],
        duration=calc_duration(gx_pre),
        system=system,
    ))

    gy_rew.append(make_trapezoid(
        channel=dir_pe,
        area=-phase_areas[ky_i],
        duration=calc_duration(gx_pre),
        system=system,
    ))

    gy_pre[ky_i].id = seq.register_grad_event(gy_pre[ky_i])
    gy_rew[ky_i].id = seq.register_grad_event(gy_rew[ky_i])


# 3D Phase encode
if Nz == 1: # Basically 2D
    areaZ = [-gz.area/2]
else:
    areaZ = -(np.arange(0,Nkz) - Nkz/2)/(Nkz*slice_thickness) - gz.area/2 # Combine kz encoding and rephasing

gz_enc_largest = pp.make_trapezoid(channel=dir_ss, area=np.max(np.abs(areaZ)), system=system)
Tz_enc = pp.calc_duration(gz_enc_largest)

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel=dir_ro, area=gx.area, system=system)

# RESET Block
if reset_block:

    if params['reset_type'] == 'bir4':
        from math import atan

        from pypulseq.make_arbitrary_rf import make_arbitrary_rf

        from utils.rftools import design_bir4

        # --------------------------------------------------------------------------
        # Define parameters for a BIR-4 preparation
        # --------------------------------------------------------------------------
        T_seg     = 2e-3          # duration of one pulse segment [sec]
        b1_max    = 20            # maximum RF amplitude [uT]
        dw_max    = 20e3          # maximum frequency sweep [Hz]
        zeta      = 15.2          # constant in the amplitude function [rad]
        kappa     = atan(63.6)    # constant in the frequency/phase function [rad]
        beta      = 90            # flip angle [degree]

        signal = design_bir4(T_seg, b1_max, dw_max, zeta, kappa, beta, system.rf_raster_time)

        # Create an RF event
        rf_res = make_arbitrary_rf(signal=signal, flip_angle=2*pi, system=system, return_gz=False, use="saturation")
        rf_res.signal = signal

        gx_res = make_trapezoid(channel=dir_ro, area=4 * Nx * delta_k, system=system, delay=calc_duration(rf_res))
        gy_res = make_trapezoid(channel=dir_pe, area=4 * Nx * delta_k, system=system, delay=calc_duration(rf_res))
        gz_res = make_trapezoid(channel=dir_ss, area=4 * Nx * delta_k, system=system, delay=calc_duration(rf_res))
        
        gx_res.id = seq.register_grad_event(gx_res)
        gy_res.id = seq.register_grad_event(gy_res)
        gz_res.id = seq.register_grad_event(gz_res)

        t_reset = calc_duration(rf_res, gx_res)

    elif params['reset_type'] == 'composite':
        from pypulseq.make_block_pulse import make_block_pulse
        # Create an RF event
        rf45x = make_block_pulse(duration=0.3e-3, flip_angle=pi/4, phase_offset=0, system=system, use="saturation")
        rf90y = make_block_pulse(duration=0.6e-3, flip_angle=pi/2, phase_offset=pi/2, system=system, use="saturation")
        rf90x = make_block_pulse(duration=0.6e-3, flip_angle=pi/2, phase_offset=0, system=system, use="saturation")
        rf45y = make_block_pulse(duration=0.3e-3, flip_angle=pi/4, phase_offset=-pi/2, system=system, use="saturation")
        rf_ringdown_delay = make_delay(system.rf_ringdown_time + system.rf_dead_time)
        
        gx_res = make_trapezoid(channel=dir_ro, area=4 * Nx * delta_k, system=system)
        gy_res = make_trapezoid(channel=dir_pe, area=4 * Nx * delta_k, system=system)
        gz_res = make_trapezoid(channel=dir_ss, area=4 * Nx * delta_k, system=system)
        
        gx_res.id = seq.register_grad_event(gx_res)
        gy_res.id = seq.register_grad_event(gy_res)
        gz_res.id = seq.register_grad_event(gz_res)
        t_reset = (
              calc_duration(rf45x) + calc_duration(rf_ringdown_delay)
            + calc_duration(rf90y) + calc_duration(rf_ringdown_delay)
            + calc_duration(rf90x) + calc_duration(rf_ringdown_delay)
            + calc_duration(rf45y)
            + calc_duration(gx_res, gy_res, gz_res)
            )
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
            - calc_duration(gx_spoil, gz_enc_largest, gy_pre[0])
            - t_reset
)

assert np.all(TEd >= 0), "Required TE can not be achieved."
assert np.all(TRd >= 0), "Required TR can not be achieved."

delay_TE = make_delay(TEd)
delay_TR = [make_delay(TRd)]

if is_afi:
    TR2d = rnd2GRT(
            TR2
            - TE
            - calc_duration(gz)/2
            - calc_duration(gx)/2
            - calc_duration(gx_spoil, gz_enc_largest, gy_pre[0])
            - t_reset
    )
    delay_TR.append(make_delay(TR2d))

seq.add_block(pp.make_label(label="REV", type="SET", value=1))

# Register fixed grad events for speedup
gx_pre.id = seq.register_grad_event(gx_pre)
gx.id = seq.register_grad_event(gx)

nTR = 1
if is_afi:
    nTR = 2

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over slices
for fa_i in range(len(alpha)):
    if not is_afi:
        seq.add_block(pp.make_label(type="SET", label="SET", value=fa_i))

    for kz_i in range(Nkz):
        # Loop over phase encodes and define sequence blocks
        seq.add_block(make_label(type="SET", label="PAR", value=Nkz-kz_i-1))
        kz_phase_offset = kz_i*2*np.pi*z/(Nkz*slice_thickness) - 2*np.pi*rf[fa_i].freq_offset*calc_rf_center(rf[fa_i])[0] # align the phase for off-center slices

        gz_enc = pp.make_trapezoid(channel=dir_ss,area=areaZ[kz_i],duration=Tz_enc, system=system)
        gz_rew = pp.make_trapezoid(channel=dir_ss,area=-(areaZ[kz_i] + gz.area/2),duration=Tz_enc, system=system)

        gz_enc.id = seq.register_grad_event(gz_enc)
        gz_rew.id = seq.register_grad_event(gz_rew)

        # Only run dummy TRs for the first partition
        pe_rng = None
        if kz_i == 0:
            pe_rng = range(-Ndummy, Ny)
        else:
            pe_rng = range(Ny)

        n_i = 0
        for ky_i in pe_rng:
            for tr_i in range(nTR):
                if is_afi:
                    seq.add_block(pp.make_label(type="SET", label="SET", value=tr_i))

                # RF/ADC Phase update
                rf_spoil_upd = 0.5*phi0*(n_i*n_i + n_i + 2)*np.pi/180
                rf[fa_i].phase_offset = rf_spoil_upd + kz_phase_offset
                adc.phase_offset = rf_spoil_upd
                n_i+=1

                seq.add_block(rf[fa_i], gz)
                
                seq.add_block(delay_TE)

                if ky_i < 0: # Dummy TRs
                    seq.add_block(gx_pre, gy_pre[int(Ny/2)], gz_enc)
                    seq.add_block(gx)
                    seq.add_block(gx_spoil, gy_rew[int(Ny/2)], gz_rew)


                else:
                    seq.add_block(make_label(type="SET", label="LIN", value=ky_i))
                    seq.add_block(gx_pre, gy_pre[ky_i], gz_enc)
                    seq.add_block(gx, adc)
                    seq.add_block(gx_spoil, gy_rew[ky_i], gz_rew)


                if reset_block:
                    if params['reset_type'] == 'bir4':
                        rf_res.phase_offset = rf[fa_i].phase_offset
                        seq.add_block(rf_res, gx_res, gy_res, gz_res)
                    elif params['reset_type'] == 'composite':
                        # Set additional phases, align them 90 diff with excitation
                        rf45x.phase_offset += rf[fa_i].phase_offset
                        rf90y.phase_offset += rf[fa_i].phase_offset
                        rf90x.phase_offset += rf[fa_i].phase_offset
                        rf45y.phase_offset += rf[fa_i].phase_offset

                        seq.add_block(rf45x)
                        seq.add_block(rf_ringdown_delay)
                        seq.add_block(rf90y)
                        seq.add_block(rf_ringdown_delay)
                        seq.add_block(rf90x)
                        seq.add_block(rf_ringdown_delay)
                        seq.add_block(rf45y)
                        seq.add_block(gx_res, gy_res, gz_res)

                seq.add_block(delay_TR[tr_i])
            


ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# ================
# VISUALIZATION
# ===============
if show_diag:
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

