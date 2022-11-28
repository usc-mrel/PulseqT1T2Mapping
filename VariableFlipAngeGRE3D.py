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
from Kernels.FISPKernel import FISPKernel
from Kernels.MagSatKernel import MagSatKernel

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

if is_afi:
    gx_afi_spoil = pp.make_trapezoid(channel=dir_ro, area=gx.area*(TR2/TR), system=system)

params_gre = params
params_gre['flip_angle'] = params['flip_angle'][0]
GREKernel = FISPKernel(seq, params)

# RESET Block
ResBlock = None
if reset_block:
    ResBlock = MagSatKernel(seq, params['reset_type'], crasher_area=4*Nx*delta_k)

t_reset = ResBlock.duration() if reset_block else 0

# Calculate timing
TEd = rnd2GRT(
            TE
            - calc_duration(gz)/2
            - calc_duration(gx_pre, gz_enc_largest)
            - calc_duration(gx)/2
)
TRd = rnd2GRT(
            TR
            # - TE
            # - calc_duration(gz)/2
            # - calc_duration(gx)/2
            # - calc_duration(gx_spoil, gz_enc_largest, gy_pre[0])
            - GREKernel.duration()
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
    phi0 = 129.3 # AFI RF spoiling increment
else:
    phi0 = 117  # conventional RF spoiling increment


# =============================================================================
# CONSTRUCT SEQUENCE
# =============================================================================
# Loop over slices
for fa_i in range(len(alpha)):
    GREKernel.update_flip_angle(alpha[fa_i])
    if not is_afi:
        seq.add_block(pp.make_label(type="SET", label="SET", value=fa_i))

    for kz_i in range(Nkz):
        # Loop over phase encodes and define sequence blocks
        seq.add_block(make_label(type="SET", label="PAR", value=Nkz-kz_i-1))
        # kz_phase_offset = kz_i*2*np.pi*z/(Nkz*slice_thickness) - 2*np.pi*rf[fa_i].freq_offset*calc_rf_center(rf[fa_i])[0] # align the phase for off-center slices

        # gz_enc = pp.make_trapezoid(channel=dir_ss,area=areaZ[kz_i],duration=Tz_enc, system=system)
        # gz_rew = pp.make_trapezoid(channel=dir_ss,area=-(areaZ[kz_i] + gz.area/2),duration=Tz_enc, system=system)

        # gz_enc.id = seq.register_grad_event(gz_enc)
        # gz_rew.id = seq.register_grad_event(gz_rew)

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
                # rf[fa_i].phase_offset = rf_spoil_upd + kz_phase_offset
                # adc.phase_offset = rf_spoil_upd
                
                n_i+=1
                # if tr_i==1:
                #     n_i+=(TR2//TR-1)

                # seq.add_block(rf[fa_i], gz)
                
                # seq.add_block(delay_TE)

                if ky_i < 0: # Dummy TRs
                    GREKernel.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False)
                #     seq.add_block(gx_pre, gy_pre[Ny//2], gz_enc)
                #     seq.add_block(gx)
                #     if tr_i == 0:
                #         seq.add_block(gx_spoil, gy_rew[Ny//2], gz_rew)
                #     elif tr_i == 1:
                #         seq.add_block(gx_afi_spoil, gy_rew[Ny//2], gz_rew)

                else:
                    GREKernel.add_kernel(ky_i, kz_i, rf_spoil_upd)

                #     seq.add_block(make_label(type="SET", label="LIN", value=ky_i))
                #     seq.add_block(gx_pre, gy_pre[ky_i], gz_enc)
                #     seq.add_block(gx, adc)
                #     if tr_i == 0:
                #         seq.add_block(gx_spoil, gy_rew[ky_i], gz_rew)
                #     elif tr_i == 1:
                #         seq.add_block(gx_afi_spoil, gy_rew[ky_i], gz_rew)


                if reset_block:
                    MagSatKernel.add_kernel(rf[fa_i].phase_offset)

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

