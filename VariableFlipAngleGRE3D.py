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

z = params['slice_pos'][0]

ro_duration = params['readout_duration']  # ADC duration

seq_filename = params['file_name']
seq_folder   = params['output_folder']

reset_block = params['reset_block']
is_afi      = params['afi']


# Sanity check the input params
if params['afi'] and len(alpha) > 1:
    print("AFI is on, and multiple flip-angle supplied. Only using the first flip angle, discarding the rest.")
    alpha = [alpha[0]]

# Set system limits
sys_params = load_params("lospecs", "./systems")
system = Opts(
    max_grad = sys_params['max_grad'], grad_unit="mT/m",
    max_slew = sys_params['max_slew'], slew_unit="T/m/s",
    grad_raster_time = sys_params['grad_raster_time'],  # [s] ( 10 us)
    rf_raster_time   = sys_params['rf_raster_time'],    # [s] (  1 us)
    rf_ringdown_time = sys_params['rf_ringdown_time'],  # [s] ( 10 us)
    rf_dead_time     = sys_params['rf_dead_time'],      # [s] (100 us)
    adc_dead_time    = sys_params['adc_dead_time'],     # [s] ( 10 us)
)

seq = pp.Sequence(system)  # Create a new sequence object

# Crasher area's for VFA and AFI are set according to:
# 1. Yarnykh VL. Optimal radiofrequency and gradient spoiling for improved accuracy of T1 and B1 measurements 
# using fast steady-state techniques. Magnetic Resonance in Medicine. 2010;63(6):1610-1626. doi:10.1002/mrm.22394

if is_afi:
    A = (450)*1e-6*system.gamma # Spoiler area in Hz [mT.ms/m] -> [Hz.s/m]
else:
    A = (280)*1e-6*system.gamma # Spoiler area in Hz [mT.ms/m] -> [Hz.s/m]

delta_k = 1 / fov[0]

params_gre = params.copy()
params_gre['flip_angle'] = params['flip_angle'][0]
GREKernel = FISPKernel(seq, params_gre, spoiler_area=A, spoiler_axes='xyz')

# RESET Block
ResBlock = None
if reset_block:
    ResBlock = MagSatKernel(seq, params['reset_type'], crasher_area=A)

t_reset = ResBlock.duration() if reset_block else 0

# Calculate timing
TRd = rnd2GRT(
            TR
            - GREKernel.duration()
            - t_reset
)

assert np.all(TRd >= 0), "Required TR can not be achieved."

delay_TR = make_delay(TRd)


if is_afi:
    phi0 = 39 # AFI RF spoiling increment
    Ntr = int(params['TR2']/params['TR'])

else:
    phi0 = 169  # 117 is conventional RF spoiling increment, 169 is for VFA T1 mapping

seq.add_block(pp.make_label(label="REV", type="SET", value=1))



# =============================================================================
# CONSTRUCT SEQUENCE
# =============================================================================
# Loop over slices
for fa_i in range(len(alpha)):
    GREKernel.update_flip_angle(alpha[fa_i])

    seq.add_block(pp.make_label(type="SET", label="SET", value=fa_i))

    for kz_i in range(Nkz):
        # Loop over phase encodes and define sequence blocks
        seq.add_block(make_label(type="SET", label="PAR", value=Nkz-kz_i-1))
        # kz_phase_offset = kz_i*2*np.pi*z/(Nkz*slice_thickness) - 2*np.pi*rf[fa_i].freq_offset*calc_rf_center(rf[fa_i])[0] # align the phase for off-center slices

        # Only run dummy TRs for the first partition
        pe_rng = None
        if kz_i == 0:
            pe_rng = range(-Ndummy, Ny)
        else:
            pe_rng = range(Ny)

        n_i = 0
        for ky_i in pe_rng:

                # RF/ADC Phase update
                rf_spoil_upd = 0.5*phi0*(n_i*n_i + n_i + 2)*np.pi/180
                n_i+=1

                if ky_i < 0: # Dummy TRs
                    GREKernel.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False)
                else:
                    if is_afi:
                        seq.add_block(make_label(type="SET", label="SET", value=0))

                    seq.add_block(make_label(type="SET", label="LIN", value=ky_i))
                    GREKernel.add_kernel(ky_i, kz_i, rf_spoil_upd)

                if reset_block:
                    ResBlock.add_kernel(phase_offset=rf_spoil_upd+pi/2)

                seq.add_block(delay_TR)

                if is_afi:
                    rf_spoil_upd = 0.5*phi0*(n_i*n_i + n_i + 2)*np.pi/180

                    n_i+=1
                    if ky_i < 0: # Dummy TRs
                        GREKernel.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False)
                    else:
                        seq.add_block(make_label(type="SET", label="SET", value=1))
                        GREKernel.add_kernel(ky_i, kz_i, rf_spoil_upd)
                
                    seq.add_block(delay_TR)

                    # Virtual TRs
                    for tr_i in range(Ntr-1):
                        if ky_i < 0: # Dummy TRs
                            GREKernel.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False, play_rf=False)
                        else:
                            GREKernel.add_kernel(ky_i, kz_i, rf_spoil_upd, is_acq=False, play_rf=False)

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
if show_diag:
    seqplot.plot_bokeh(seq, time_range=((150)*TR, (200)*TR), time_disp="ms", grad_disp="mT/m", plot_now=True)


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
    seq.set_definition(key="ReconMatrixSize", value=[Nx, Ny, Nz])

    seq_filename+=f"_TR{int(TR*1e3)}"
    if is_afi:
        seq_filename+=f"_N{int(Ntr)}"

    seq_path = os.path.join(seq_folder, f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    from utils.seqinstall import seqinstall
    seqinstall(seq_path)

