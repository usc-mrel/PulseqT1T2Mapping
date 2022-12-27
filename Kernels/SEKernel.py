from pypulseq.Sequence.sequence import Sequence
from math import pi
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_label import make_label
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.calc_duration import calc_duration
from scipy.io import savemat
import numpy as np
from utils.grad_timing import rnd2GRT


class SEKernel:
    def __init__(self, seq: Sequence, params: dict, save_rf_info: bool=True) -> None:
        
        system = seq.system
        Nx, Ny, Nz = params['matrix_size']

        slice_thickness = params['slice_thickness']  # Slice thickness
        fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
        TE  = params['TE']     # Echo time [s]
        ETL = params['ETL']    # Echo Train Length
        TE = TE*np.arange(1, ETL+1)#[11e-3, 22e-3, 33e-3]#  # [s]
        Neco = len(TE) # Echo Train Length (ETL)

        slice_pos = params['slice_pos']
        Nslc = len(slice_pos)


        readout_time = params['readout_duration']  # ADC duration

        # ## **RF**

        rf_flip = 90  # degrees
        rf_offset = 0

        flip90 = round(rf_flip * pi / 180, 3)
        flip180 = 180 * pi / 180
        rf90, gz90, gz_reph = make_sinc_pulse(flip_angle=flip90, system=system, duration=2.5e-3, 
                                        slice_thickness=slice_thickness, apodization=0.5, 
                                        phase_offset=pi/2,
                                        time_bw_product=4, return_gz=True, use="excitation")

                                        
        rf180, gz180, _ = make_sinc_pulse(flip_angle=flip180, system=system, 
                                        duration=2.5e-3, 
                                        slice_thickness=1.25*slice_thickness, 
                                        apodization=0.5, 
                                        time_bw_product=4, phase_offset=0, 
                                        return_gz=True, use="refocusing")


        if save_rf_info:
            savemat('t2_mese_rfinfo.mat', 
                {'rfe': rf90.signal, 't_rfe': rf90.t, 
                'rfr': rf180.signal, 't_rfr': rf180.t, 
                'Ge': (100*gz90.amplitude)/(system.gamma), 
                'Gr': (100*gz180.amplitude)/(system.gamma)}
            )
        
        # ## **READOUT**
        # Readout gradients and related events
        delta_k = 1 / fov[0]
        k_width = (Nx) * delta_k
        gx = make_trapezoid(channel='x', system=system, flat_area=k_width, 
                            flat_time=readout_time)
        adc = make_adc(num_samples=Nx, duration=readout_time, delay=gx.rise_time)

        # ## **PREPHASE AND REPHASE**

        # phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
        phase_areas = np.linspace(-0.5*delta_k*Ny, 0.5*delta_k*Ny, Ny) # Stagger ky axis by delta_ky/2 to avoid ky shift at odd numbered echoes.

        gy_pe_max = make_trapezoid(channel='y', system=system, area=np.max(np.abs(phase_areas)))
        pe_duration = calc_duration(gy_pe_max)

        gx_pre = make_trapezoid(channel='x', system=system, area=(gx.area)/2)
        gx_post = make_trapezoid(channel='x', system=system, area=3*gx.area/2)

        # Register Gx grads
        gx.id = seq.register_grad_event(gx)
        gx_pre.id = seq.register_grad_event(gx_pre)
        gx_post.id = seq.register_grad_event(gx_post)

        # ## **SPOILER**
        gss_bridges = []

        gss_spoil_6piarea = 4/slice_thickness #-(gz180.area-gz180.flat_area)/2
        gss_spoil_4piarea = 2/slice_thickness
        gss_spoil_6pi = make_trapezoid(channel='z', system=system, area=gss_spoil_6piarea)
        gss_spoil_4pi = make_trapezoid(channel='z', system=system, area=gss_spoil_4piarea, rise_time=gss_spoil_6pi.rise_time, flat_time=gss_spoil_6pi.flat_time)

        # gss_spoil_neg = make_trapezoid(channel='z', system=system, area=-gss_spoil_area)

        # Bridge the spoilers
        # We will have 4 spoilers with alternating amplitude to reduce the number of Stimulated Echo (STE) paths, as we are likely to have large ETL.
        gss_times = np.cumsum([0, gss_spoil_6pi.rise_time, gss_spoil_6pi.flat_time, 
                            gss_spoil_6pi.fall_time-gz180.rise_time, gz180.flat_time, gss_spoil_6pi.rise_time-gz180.fall_time, 
                            gss_spoil_6pi.flat_time, gss_spoil_6pi.fall_time])

        gss_amps_6pi = np.array([0, gss_spoil_6pi.amplitude, gss_spoil_6pi.amplitude, gz180.amplitude, gz180.amplitude, gss_spoil_6pi.amplitude, gss_spoil_6pi.amplitude, 0])
        gss_amps_4pi = np.array([0, gss_spoil_4pi.amplitude, gss_spoil_4pi.amplitude, gz180.amplitude, gz180.amplitude, gss_spoil_4pi.amplitude, gss_spoil_4pi.amplitude, 0])

        gss_bridges.append(make_extended_trapezoid(channel='z', amplitudes=gss_amps_6pi, times=gss_times, system=system))
        gss_bridges.append(make_extended_trapezoid(channel='z', amplitudes=-gss_amps_6pi, times=gss_times, system=system))
        gss_bridges.append(make_extended_trapezoid(channel='z', amplitudes=gss_amps_4pi, times=gss_times, system=system))
        gss_bridges.append(make_extended_trapezoid(channel='z', amplitudes=-gss_amps_4pi, times=gss_times, system=system))

        rf180.delay = gss_spoil_6pi.rise_time + gss_spoil_6pi.flat_time + gss_spoil_6pi.fall_time - gz180.rise_time


        gss_bridge_duration = calc_duration(gss_bridges[0])

        # End of TR spoiler
        gz_spoil = make_trapezoid(channel='z', system=system, area=2/slice_thickness)
        gz_spoil.id = seq.register_grad_event(gz_spoil)

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
                        - pe_duration
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


            # Prepare phase encode and rewinders
            gy_pre = []
            gy_post = []

            for ky_i in range(Ny):
                gy_pre.append(make_trapezoid(channel='y', system=system, 
                            area=phase_areas[-ky_i -1], duration=pe_duration))
                gy_post.append(make_trapezoid(channel='y', system=system, 
                            area=-phase_areas[-ky_i -1], duration=pe_duration))

                gy_pre[ky_i].id = seq.register_grad_event(gy_pre[ky_i])
                gy_post[ky_i].id = seq.register_grad_event(gy_post[ky_i])

            # Store necessary variables in class
            self.seq = seq
            self.TE = TE
            self.rf90 = rf90
            self.gz90 = gz90
            self.gz_reph = gz_reph
            self.rf180 = rf180
            self.gss_bridges = gss_bridges
            self.gx_pre = gx_pre
            self.gx_post = gx_post
            self.gx = gx
            self.adc = adc
            self.gy_pre = gy_pre
            self.gy_post = gy_post
            self.gz_spoil = gz_spoil
            self.Neco = Neco
            self.pre180delay = pre180delay
            self.post180delay = post180delay
            self.gz180amp = gz180.amplitude
            self.slice_pos = slice_pos


    def duration(self):
        tot_dur = (
            calc_duration(self.gz90)/2 + self.TE[-1]
            + calc_duration(self.gx)/2
            + calc_duration(self.gy_post[-1])
            + calc_duration(self.gx_post, self.gz_spoil)
        )

        return tot_dur

    def add_kernel(self, ky_i: int, slc_i: int, is_acq: bool = True):
        seq = self.seq

        # Apply RF offsets
        freq_offset = self.gz90.amplitude * self.slice_pos[slc_i]
        self.rf90.freq_offset = freq_offset

        freq_offset = self.gz180amp * self.slice_pos[slc_i]
        self.rf180.freq_offset = freq_offset
        
        # self.rf90.phase_offset = self.rf90.phase_offset - 2*pi*self.rf90.freq_offset*calc_rf_center(self.rf90)[0] # align the phase for off-center slices

        # self.rf180.phase_offset = self.rf180.phase_offset - 2*pi*self.rf180.freq_offset*calc_rf_center(self.rf180)[0] #dito

        # RF chopping
        self.rf90.phase_offset = (self.rf90.phase_offset + pi) % (2*pi)
        self.adc.phase_offset = self.rf90.phase_offset

        seq.add_block(self.rf90, self.gz90)

        seq.add_block(self.gx_pre, self.gz_reph)

        for eco_i in range(self.Neco):
            
            # Alternate between bridges
            # gss_bridge_ = gss_bridges[eco_i%2]
            gss_bridge_ = self.gss_bridges[0]

            seq.add_block(self.pre180delay[eco_i])
            seq.add_block(self.rf180, gss_bridge_)
            seq.add_block(self.post180delay[eco_i])

            seq.add_block(self.gy_pre[ky_i])
            
            if is_acq:
                if self.Neco > 1: # Don't set SET label if there is only one echo. It could be a single echo multi contrast seq, we don't wanna overwrite.
                    seq.add_block(make_label(type="SET", label="SET", value=eco_i))

                seq.add_block(self.gx, self.adc)
            else:
                seq.add_block(self.gx)

            seq.add_block(self.gy_post[ky_i])
        

        seq.add_block(self.gx_post, self.gz_spoil)
