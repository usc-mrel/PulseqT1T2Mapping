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

delta_k = 1 / fov[0]

params_gre = params.copy()
params_gre['flip_angle'] = params['flip_angle'][0]
GREKernel = FISPKernel(seq, params_gre)

# RESET Block
ResBlock = None
if reset_block:
    ResBlock = MagSatKernel(seq, params['reset_type'], crasher_area=4*Nx*delta_k)

t_reset = ResBlock.duration() if reset_block else 0

# Calculate timing
TRd = rnd2GRT(
            TR
            - GREKernel.duration()
            - t_reset
)

assert np.all(TRd >= 0), "Required TR can not be achieved."

delay_TR = make_delay(TRd)


is_afi = params['afi']
if is_afi:
    phi0 = 129.3 # AFI RF spoiling increment

    params_gre2 = params_gre.copy()
    params_gre2['TR'] = params['TR2']
    GREKernel2 = FISPKernel(seq, params_gre2, params['TR2']/params['TR'])

    TR2 = params['TR2']
    TR2d = rnd2GRT(
            TR2
            - TE
            - GREKernel2.duration()
    )
    delay_TR2 = make_delay(TR2d)

else:
    phi0 = 117  # conventional RF spoiling increment

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
                # if tr_i==1:
                #     n_i+=(TR2//TR-1)                

                if ky_i < 0: # Dummy TRs
                    GREKernel.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False)
                else:
                    seq.add_block(make_label(type="SET", label="SEG", value=0))
                    GREKernel.add_kernel(ky_i, kz_i, rf_spoil_upd) # TODO: RF spoil update

                if is_afi:
                    rf_spoil_upd = 0.5*phi0*(n_i*n_i + n_i + 2)*np.pi/180
                    n_i+=1
                    if ky_i < 0: # Dummy TRs
                        GREKernel2.add_kernel(Ny//2, Nz//2, rf_spoil_upd, is_acq=False)
                    else:
                        seq.add_block(make_label(type="SET", label="SEG", value=1))
                        GREKernel2.add_kernel(ky_i, kz_i, rf_spoil_upd)

                if reset_block:
                    MagSatKernel.add_kernel(rf_spoil_upd)

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
    seq.set_definition(key="ReconMatrixSize", value=[Nx, Ny, Nz])

    seq_path = os.path.join(seq_folder, f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    from utils.seqinstall import seqinstall
    seqinstall(seq_path)

