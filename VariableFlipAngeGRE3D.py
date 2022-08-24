 
import numpy as np
import pypulseq as pp
import os
import mplcursors
import matplotlib.pyplot as plt
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.make_label import make_label
from utils.grad_timing import rnd2GRT
import json
from utils import seqplot

# =============
# USER OPTIONS
# =============
plot = True 
write_seq = True
detailed_rep = False

# param_filename = "t1map_MnCl2_vfagre"
param_filename = "t1map_NiCl2_vfagre"

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
# alpha = np.logspace(np.log2(1), np.log2(30),num=7,base=2)  # Flip angle
# alpha = [60, 120]  # Flip angle

slice_thickness = params['slice_thickness']  # Slice thickness
fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
TE  = params['TE'][0]  # Echo time
TR  = params['TR']     # Repetition time

slice_orientation = params['slice_orientation'] # "TRA", "COR", "SAG" 
z = params['slice_pos'][0]

ro_duration = params['readout_duration']  # ADC duration

seq_filename = params['file_name']
seq_folder   = params['output_folder']

rf_spoiling_inc = 117  # RF spoiling increment


# Grad axes
if slice_orientation == "TRA":
    dir_ro, dir_pe, dir_ss = ("x", "y", "z")
elif slice_orientation == "COR":
    dir_ro, dir_pe, dir_ss = ("x", "z", "y")
elif slice_orientation == "SAG":
    dir_ro, dir_pe, dir_ss = ("z", "y", "x")

# Set system limits
system = pp.Opts(
    max_grad=15,
    grad_unit="mT/m",
    max_slew=40,
    slew_unit="T/m/s",
    rf_ringdown_time=10e-6,
    rf_dead_time=100e-6,
    # rf_raster_time=100e-9, # [s] (100 ns)
    adc_dead_time=10e-6,
)

# ======
# CREATE EVENTS
# ======
# Create alpha-degree slice selection pulse and gradient
tbwp = 4 # Time-BW product of sinc
Trf = 2e-3 # [s] RF duration
BWrf = tbwp/Trf # RF BW
gz_area = BWrf/fov[2] * Trf


# Calculate slice selection and rephasing
gz = make_trapezoid(
    channel=dir_ss, system=system, flat_time=Trf, flat_area=gz_area
)
# gz_reph = make_trapezoid(
#     channel="z",
#     system=system,
#     area=-gz.area/2
# )

# Calculate and store rf pulses
rf = []
for fa_i in range(len(alpha)):
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
    rf.append(rf_)

# Define other gradients and ADC events
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
    areaZ = np.arange(-np.ceil((Nz-1)/2),np.floor((Nz)/2))/fov[2] - gz.area/2 # Combine kz encoding and rephasing

gz_enc_largest = pp.make_trapezoid(channel=dir_ss, area=np.max(np.abs(areaZ)), system=system)
Tz_enc = pp.calc_duration(gz_enc_largest)


# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel=dir_ro, area=2 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel=dir_ss, area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = rnd2GRT(
            TE
            - calc_duration(gz)/2
            - calc_duration(gx_pre, gz_enc_largest)
            - calc_duration(gx)/2
)
delay_TR = rnd2GRT(
            TR
            - TE
            - calc_duration(gz)/2
            - calc_duration(gx)/2
)

assert np.all(delay_TE >= 0), "Required TE can not be achieved."
assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil)), "Required TR can not be achieved."

rf_phase = 0
rf_inc = 0

seq = pp.Sequence()  # Create a new sequence object

seq.add_block(pp.make_label(label="REV", type="SET", value=1))

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over slices
for fa_i in range(len(alpha)):
    rf[fa_i].freq_offset = gz.amplitude * z
    seq.add_block(pp.make_label(type="SET", label="SET", value=fa_i))

    for kz_i in range(Nz):
        #rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        # Loop over phase encodes and define sequence blocks
        seq.add_block(make_label(type="SET", label="PAR", value=kz_i))

        gz_enc = pp.make_trapezoid(channel=dir_ss,area=areaZ[kz_i],duration=Tz_enc, system=system)
        gz_enc.id = seq.register_grad_event(gz_enc)
        
        for ky_i in range(Ny):
            seq.add_block(make_label(type="SET", label="LIN", value=ky_i))

            rf[fa_i].phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf[fa_i], gz)
            
            gy_pre = pp.make_trapezoid(
                channel=dir_pe,
                area=phase_areas[ky_i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            # seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(gx_pre, gy_pre, gz_enc)

            seq.add_block(pp.make_delay(delay_TE))

            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            spoil_block_contents = [pp.make_delay(delay_TR), gx_spoil, gy_pre, gz_spoil]
            seq.add_block(*spoil_block_contents)

ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# ======
# VISUALIZATION
# ======
if plot:
    seqplot.plot(seq, time_range=(0, 20*TR), time_disp="ms", grad_disp="mT/m", plot_now=True)

    # seq.plot(time_range=(0, 256*TR), time_disp="ms", grad_disp="mT/m", plot_now=False)
    # seq.plot_logical(label="lin", time_range=(0, 4*TR), time_disp="ms", grad_disp="mT/m", plot_now=False)

    mplcursors.cursor()
    plt.show()

if detailed_rep:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)


# =========
# WRITE .SEQ
# =========
if write_seq:
    # Prepare the sequence output for the scanner
    seq.set_definition(key="FOV", value=fov)
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

