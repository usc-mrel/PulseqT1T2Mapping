# -------------------------------------------------------------------
# Title: 2D Multi-echo spin-echo, based on SE from pypulseq library.
# Author: Bilal Tasdelen <tasdelen at usc dot edu>
# -------------------------------------------------------------------

## TODO:
# 1. Test IRSE
# 2. Multislice

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import os
from pypulseq.make_delay import make_delay
from pypulseq.make_label import make_label
from pypulseq.opts import Opts
from pypulseq.Sequence.sequence import Sequence
from Kernels.IRKernel import IRKernel
from Kernels.SEKernel import SEKernel
from utils.grad_timing import rnd2GRT

import json

from utils.load_params import load_params

# ## **USER INPUTS**
# 
# These parameters are typically on the user interface of the scanner computer console 

# Load user options
user_opts = load_params("user_opts_mese", "./")

show_diag    = user_opts['show_diag']
write_seq    = user_opts['write_seq']
detailed_rep = user_opts['detailed_rep']
param_filename = user_opts["param_filename"]


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

slice_thickness = params['slice_thickness']  # Slice thickness
fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
TE  = params['TE']     # Echo time [s]
ETL = params['ETL']    # Echo Train Length
TR  = params['TR']     # Repetition time [s]
TI  = params['TI']     # Inversion time [s]
single_echo  = params['single_echo']     # Is single echo multi contrast?

slice_orientation = params['slice_orientation'] # "TRA", "COR", "SAG" 
z = params['slice_pos']

readout_time = params['readout_duration']  # ADC duration
Ndummy = params['Ndummy'] # Number 

seq_filename = params['file_name']
seq_folder   = params['output_folder']

nsa = 1  # Number of averages

TE = TE*np.arange(1, ETL+1)#[11e-3, 22e-3, 33e-3]#  # [s]

n_slices = len(z)

# Sanity check parameters

# Can not have single echo multi contrast SE with IRSE
if single_echo and TI > 0:
    raise ValueError("Can not have single echo multi contrast SE with IRSE.")

# ## **SYSTEM LIMITS**
# Set the hardware limits and initialize sequence object

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

seq = Sequence(system)

# Create IR kernel if needed

irk = None

if TI > 0:
    irk = IRKernel(seq, 1.25*slice_thickness, TI)

TI_dur = irk.duration() if TI > 0 else 0

# ## **DELAYS**
delay_TR = []
se_kernels = []

if ETL == 1 or not single_echo:
    se_kernels.append(SEKernel(seq, params))

    TRd = (TR 
    - TI_dur 
    - se_kernels[-1].duration()
    )

    delay_TR.append(make_delay(rnd2GRT(TRd)))

else:

    dummy_params = params.copy()
    dummy_params['ETL'] = 1

    for TE_ in TE:
        dummy_params['TE'] = TE_
        se_kernels.append(SEKernel(seq, dummy_params))

        TRd = (TR 
            - TI_dur 
            - se_kernels[-1].duration()
        )

        delay_TR.append(make_delay(rnd2GRT(TRd)))



# ## **CONSTRUCT SEQUENCE**

seq.add_block(make_label(label="REV", type="SET", value=1))

seq.add_block(make_label(label="PAR", type="SET", value=0))

Nocon = ETL if single_echo else 1

for ocon_i in range(Nocon): # Contrast, outer loop. Either single echo TE dimension or TI dimension
    se_kernel = se_kernels[ocon_i]

    if single_echo:
        seq.add_block(make_label(type="SET", label="SET", value=ocon_i))

    for avg_i in range(nsa):  # Averages

        seq.add_block(make_label(type="SET", label="AVG", value=avg_i))

        for slc_i in range(n_slices):  # Slices
            seq.add_block(make_label(type="SET", label="SLC", value=slc_i))


            for ky_i in range(-Ndummy, Ny):  # Phase encodes

                if TI > 0:
                    irk.add_kernel()

                if ky_i < 0:
                    se_kernel.add_kernel(Ny//2, slc_i, False)
                else:
                    seq.add_block(make_label(type="SET", label="LIN", value=ky_i))
                    se_kernel.add_kernel(ky_i, slc_i, True)

                seq.add_block(delay_TR[ocon_i])



ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# ## **PLOTTING TIMNG DIAGRAM**

if show_diag:
    # seq.plot(time_range=(0, TR*20), time_disp='ms', grad_disp='mT/m', plot_now=False)
    from utils import seqplot
    seqplot.plot_bokeh(seq, time_range=(0, TR*2), time_disp='ms', grad_disp='mT/m', plot_now=True)


## Generate Sequence Report

if detailed_rep:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# ## **GENERATING `.SEQ` FILE**
# Uncomment the code in the cell below to generate a `.seq` file and download locally.

if write_seq:
    seq.set_definition(key="FOV", value=fov)
    if TI > 0:
        seq.set_definition(key="Name", value="irse2d")
    else:
        seq.set_definition(key="Name", value="mcse2d")

    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=90)
    seq.set_definition(key="ReconMatrixSize", value=[Nx, Ny, 1])
    seq.set_definition(key="SliceOrientation", value=slice_orientation)
    
    if TI > 0:
        seq_filename += f"_TI{int(TI*1e3)}"

    seq_path = os.path.join(seq_folder, f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    from utils.seqinstall import seqinstall
    seqinstall(seq_path)


