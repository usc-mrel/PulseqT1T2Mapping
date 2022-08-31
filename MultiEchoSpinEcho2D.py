# -------------------------------------------------------------------
# Title: 2D Multi-echo spin-echo, based on SE from pypulseq library.
# Author: Bilal Tasdelen <tasdelen at usc dot edu>
# -------------------------------------------------------------------

## TODO:
# 1. **Optional:** Make 180 an adiabatic pulse.

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import os
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_label import make_label
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.convert import convert
from pypulseq.opts import Opts
from pypulseq.Sequence.sequence import Sequence
from utils.grad_timing import rnd2GRT
from scipy.io import savemat

import json

# ## **USER INPUTS**
# 
# These parameters are typically on the user interface of the scanner computer console 

show_diag = True
detailed_rep = False
write_seq = False

param_filename = "mese_debug"
# param_filename = "t2map_MnCl2_mese"
# param_filename = "t2map_NiCl2_mese"

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
TE  = params['TE'][0]  # Echo time
TR  = params['TR']     # Repetition time

slice_orientation = params['slice_orientation'] # "TRA", "COR", "SAG" 
z = params['slice_pos']

readout_time = params['readout_duration']  # ADC duration

seq_filename = params['file_name']
seq_folder   = params['output_folder']

nsa = 1  # Number of averages

TE = 15e-3*np.arange(1, 33)#[11e-3, 22e-3, 33e-3]#  # [s]

n_slices = len(z)


# ## **SYSTEM LIMITS**
# Set the hardware limits and initialize sequence object

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

seq = Sequence(system)

# Grad axes
if slice_orientation == "TRA":
    dir_ro, dir_pe, dir_ss = ("x", "y", "z")
elif slice_orientation == "COR":
    dir_ro, dir_pe, dir_ss = ("x", "z", "y")
elif slice_orientation == "SAG":
    dir_ro, dir_pe, dir_ss = ("z", "y", "x")

Neco = len(TE) # Echo Train Length (ETL)

# ## **RF**

rf_flip = 90  # degrees
rf_offset = 0

flip90 = round(rf_flip * pi / 180, 3)
flip180 = 180 * pi / 180
rf90, gz90, gz_reph = make_sinc_pulse(flip_angle=flip90, system=system, duration=2.5e-3, 
                                slice_thickness=slice_thickness, apodization=0.5, 
                                time_bw_product=4, return_gz=True, use="excitation")


gz90.channel = dir_ss
gz_reph.channel = dir_ss
                                 
rf180, gz180, _ = make_sinc_pulse(flip_angle=flip180, system=system, 
                                  duration=2.5e-3, 
                                  slice_thickness=2*slice_thickness, 
                                  apodization=0.5, 
                                time_bw_product=4, phase_offset=pi/2, 
                                return_gz=True, use="refocusing")

gz180.channel = dir_ss

savemat('t2_mese_rfinfo.mat', {'rfe': rf90.signal, 't_rfe': rf90.t, 'rfr': rf180.signal, 't_rfr': rf180.t, 'Ge': (100*gz90.amplitude)/(system.gamma), 'Gr': (100*gz180.amplitude)/(system.gamma)})

# ## **READOUT**
# Readout gradients and related events
delta_k = 1 / fov[0]
k_width = (Nx) * delta_k
gx = make_trapezoid(channel=dir_ro, system=system, flat_area=k_width, 
                    flat_time=readout_time)
adc = make_adc(num_samples=Nx, duration=readout_time, delay=gx.rise_time)

# ## **PREPHASE AND REPHASE**

# phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
phase_areas = np.linspace(-0.5*delta_k*Ny, 0.5*delta_k*Ny, Ny) # Stagger ky axis by delta_ky/2 to avoid ky shift at odd numbered echoes.

# gz_reph2 = make_trapezoid(channel='z', system=system, area=-gz90.area / 2,
#                          duration=2.5e-3)

# gy_shift = make_trapezoid(channel=dir_pe, system=system, area=-delta_k) # Provides 1 sample shift along k_y to fix the next echo's phase encode

gy_pe_max = make_trapezoid(channel=dir_pe, system=system, area=np.max(np.abs(phase_areas)))
pe_duration = calc_duration(gy_pe_max)

gx_pre = make_trapezoid(channel=dir_ro, system=system, area=(gx.area)/2)
gx_post = make_trapezoid(channel=dir_ro, system=system, area=3*gx.area/2)

# ## **SPOILER**
gss_spoil_6piarea = 4/slice_thickness -(gz180.area-gz180.flat_area)/2
gss_spoil_4piarea = 2/slice_thickness
gss_spoil_6pi = make_trapezoid(channel=dir_ss, system=system, area=gss_spoil_6piarea)
gss_spoil_4pi = make_trapezoid(channel=dir_ss, system=system, area=gss_spoil_4piarea, rise_time=gss_spoil_6pi.rise_time, flat_time=gss_spoil_6pi.flat_time)

# gss_spoil_neg = make_trapezoid(channel='z', system=system, area=-gss_spoil_area)

# Bridge the spoilers
# We will have 4 spoilers with alternating amplitude to reduce the number of Stimulated Echo (STE) paths, as we are likely to have large ETL.
gss_times = np.cumsum([0, gss_spoil_6pi.rise_time, gss_spoil_6pi.flat_time, 
                    gss_spoil_6pi.fall_time-gz180.rise_time, gz180.flat_time, gss_spoil_6pi.rise_time-gz180.fall_time, 
                    gss_spoil_6pi.flat_time, gss_spoil_6pi.fall_time])

gss_amps_6pi = np.array([0, gss_spoil_6pi.amplitude, gss_spoil_6pi.amplitude, gz180.amplitude, gz180.amplitude, gss_spoil_6pi.amplitude, gss_spoil_6pi.amplitude, 0])
gss_amps_4pi = np.array([0, gss_spoil_4pi.amplitude, gss_spoil_4pi.amplitude, gz180.amplitude, gz180.amplitude, gss_spoil_4pi.amplitude, gss_spoil_4pi.amplitude, 0])

gss_bridges = []
gss_bridges.append(make_extended_trapezoid(channel=dir_ss, amplitudes=gss_amps_6pi, times=gss_times, system=system))
gss_bridges.append(make_extended_trapezoid(channel=dir_ss, amplitudes=-gss_amps_6pi, times=gss_times, system=system))
gss_bridges.append(make_extended_trapezoid(channel=dir_ss, amplitudes=gss_amps_4pi, times=gss_times, system=system))
gss_bridges.append(make_extended_trapezoid(channel=dir_ss, amplitudes=-gss_amps_4pi, times=gss_times, system=system))

gss_bridge_duration = calc_duration(gss_bridges[0])

# gss_bridge     = make_extended_trapezoid(channel=dir_ss, amplitudes=gss_amps, times=gss_times, system=system)
# gss_bridge_neg = make_extended_trapezoid(channel=dir_ss, amplitudes=-gss_amps, times=gss_times, system=system)

rf180.delay = gss_spoil_6pi.rise_time + gss_spoil_6pi.flat_time + gss_spoil_6pi.fall_time - gz180.rise_time

# End of TR spoiler
gz_spoil = make_trapezoid(channel=dir_ss, system=system, area=2/slice_thickness)


# ## **DELAYS**
# Echo time (TE) and repetition time (TR).

pre180delay = []
post180delay = []

for eco_i in range(0, Neco):
    if eco_i == 0:
        pre180d = (TE[0]/2 
                - calc_duration(gz90) / 2 
                - calc_duration(gx_pre, gz_reph) 
                - gss_bridge_duration/2
                )
        post180d = (TE[0]/2 
                - gss_bridge_duration/2
                - pe_duration
                - (gx.rise_time + readout_time/2)
                )
    else:
        pre180d = ((TE[eco_i] - TE[eco_i-1])/2 
                - calc_duration(gx)/2 
                - gss_bridge_duration/2
                )
        post180d = ((TE[eco_i] - TE[eco_i-1])/2 
                - gss_bridge_duration/2
                - pe_duration
                - (gx.rise_time + readout_time/2)
                )

    assert (pre180d > 0) and (post180d > 0), f"TE for echo {eco_i} can not be satisfied."

    pre180delay.append(make_delay(rnd2GRT(pre180d)))
    post180delay.append(make_delay(rnd2GRT(post180d)))

delay_TR = TR - TE[-1] - calc_duration(gx) / 2 - max(pe_duration, calc_duration(gz_spoil), calc_duration(gx_post))
delay_TR = make_delay(rnd2GRT(delay_TR))

# ## **CONSTRUCT SEQUENCE**
# Construct sequence for one phase encode and multiple slices

# Prepare RF offsets. This is required for multi-slice acquisition
# delta_z = n_slices * slice_gap
# z = [0] #np.linspace((-delta_z / 2), (delta_z / 2), n_slices) + rf_offset

seq.add_block(make_label(label="REV", type="SET", value=1))

seq.add_block(make_label(label="PAR", type="SET", value=0))

gz180amp = gz180.amplitude
# gss_spoil_amp = gss_spoil.amplitude

for avg_i in range(nsa):  # Averages

    if avg_i == 0:
        seq.add_block(make_label(type="SET", label="AVG", value=0))
    else:
        seq.add_block(make_label(type="INC", label="AVG", value=1))

    for slc_i in range(n_slices):  # Slices
        if slc_i == 0:
            seq.add_block(make_label(type="SET", label="SLC", value=0))
        else:
            seq.add_block(make_label(type="INC", label="SLC", value=1))
        # Apply RF offsets
        freq_offset = gz90.amplitude * z[slc_i]
        rf90.freq_offset = freq_offset

        freq_offset = gz180amp * z[slc_i]
        rf180.freq_offset = freq_offset
        
        rf90.phase_offset = rf90.phase_offset - 2*pi*rf90.freq_offset*calc_rf_center(rf90)[0] # align the phase for off-center slices

        rf180.phase_offset = rf180.phase_offset - 2*pi*rf180.freq_offset*calc_rf_center(rf180)[0] #dito

        for ky_i in range(Ny):  # Phase encodes

            # RF Chopping
            # rf90.phase_offset = (ky_i*pi) % (2*pi)

            # adc.phase_offset = rf90.phase_offset
            

            # if ky_i == 0:
            #     seq.add_block(make_label(type="SET", label="LIN", value=0))
            # else:
            #     seq.add_block(make_label(type="INC", label="LIN", value=1))

            seq.add_block(rf90, gz90)

            gy_pre = make_trapezoid(channel=dir_pe, system=system, 
                                    area=phase_areas[-ky_i -1], duration=pe_duration)
            gy_post = make_trapezoid(channel=dir_pe, system=system, 
                                    area=-phase_areas[-ky_i -1], duration=pe_duration)

            seq.add_block(gx_pre, gz_reph)

            for eco_i in range(Neco):
                if eco_i == 0:
                    seq.add_block(make_label(type="SET", label="SET", value=0))
                elif eco_i > 0:
                    seq.add_block(make_label(type="INC", label="SET", value=1))
                
                # Alternate between bridges
                # gss_bridge_ = gss_bridges[eco_i%2]
                gss_bridge_ = gss_bridges[0]

                seq.add_block(make_label(type="SET", label="LIN", value=ky_i))

                # if (eco_i%2) == 0:
                #     rf180.freq_offset = freq_offset
                    # seq.add_block(make_label(type="SET", label="LIN", value=ky_i))


                # elif (eco_i%2) == 1:
                #     rf180.freq_offset = -freq_offset
                    # seq.add_block(make_label(type="SET", label="LIN", value=(Ny-1-ky_i)))

                seq.add_block(pre180delay[eco_i])
                seq.add_block(rf180, gss_bridge_)
                seq.add_block(post180delay[eco_i])

                seq.add_block(gy_pre)
                seq.add_block(gx, adc)
                seq.add_block(gy_post)

                # if (eco_i < Neco-1):
                #     seq.add_block(gy_shift)
            

            seq.add_block(gx_post, gz_spoil)
            seq.add_block(delay_TR)


# ## **PLOTTING TIMNG DIAGRAM**

if show_diag:
    # seq.plot(time_range=(0, TR*20), time_disp='ms', grad_disp='mT/m', plot_now=False)
    from utils import seqplot
    seqplot.plot(seq, time_range=(0, TR*20), time_disp='ms', grad_disp='mT/m', plot_now=True)

    # mplcursors.cursor()
    # (k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc) = seq.calculate_kspace()
    # plt.show()

## Generate Sequence Report

if detailed_rep:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

# ## **GENERATING `.SEQ` FILE**
# Uncomment the code in the cell below to generate a `.seq` file and download locally.

if write_seq:
    seq.set_definition(key="FOV", value=fov)
    seq.set_definition(key="Name", value="mese2d")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=flip90)
    seq.set_definition(key="ReconMatrixSize", value=[Nx, Ny, 1])
    seq.set_definition(key="SliceOrientation", value=slice_orientation)
    
    seq_path = os.path.join(seq_folder, f'{seq_filename}.seq')
    seq.write(seq_path)  # Save to disk

    from utils.seqinstall import seqinstall
    seqinstall(seq_path)


