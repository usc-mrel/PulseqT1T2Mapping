from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.calc_duration import calc_duration
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.make_adc import make_adc
from pypulseq.Sequence.sequence import Sequence
from math import ceil, pi, sqrt
from sigpy.mri.rf import slr
import numpy as np
from utils.grad_timing import rnd2GRT

class FISPKernel:
    def __init__(self, seq: Sequence, params: dict, 
                 spoiler_area: float, spoiler_axes: str = 'x'
                 ) -> None:

        # Read params
        Nx, Ny, Nz = params['matrix_size']
        Nkz = Nz + params['kz_os_steps']
        alpha = params['flip_angle']
        slice_thickness = params['slice_thickness']  # Slice thickness
        fov = [*params['fov_inplane'], Nz*slice_thickness]  # Define FOV and resolution
        TE  = params['TE'][0]  # Echo time
        ro_duration = params['readout_duration']  # ADC duration

        system = seq.system
        # Create alpha-degree slice selection pulse and gradient
        tbwp = 8 # Time-BW product of sinc
        Trf = 2e-3 # [s] RF duration
        dt = system.rf_raster_time

        exc_pulse_type = params['exc_pulse_type'] # 'slr', 'sinc'

        # Design RF
        if exc_pulse_type == 'slr':
            raster_ratio = int(system.grad_raster_time/system.rf_raster_time)
            n = ceil((Trf/dt)/(4*raster_ratio))*4*raster_ratio
            Trf = n*dt
            bw = tbwp/Trf
            signal = slr.dzrf(n=n, tb=tbwp, ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False)

            rf, gz = make_arbitrary_rf(
                signal=signal, slice_thickness=fov[2],
                bandwidth=bw, flip_angle=alpha * pi / 180, 
                system=system, return_gz=True, use="excitation"
                )

            gzr = make_trapezoid(channel='z', area=-gz.area/2, system=system)

        elif exc_pulse_type == 'sinc':
            # Calculate and store rf pulses
            rf, gz, gzr = make_sinc_pulse(
                flip_angle=alpha * pi / 180,
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

        # -----------------------------------------
        # Define other gradients and ADC events
        # -----------------------------------------

        delta_k = 1 / fov[0]
        gx = make_trapezoid(
            channel='x', flat_area=Nx * delta_k, flat_time=ro_duration, system=system
        )

        # TODO: Oversampling.
        adc = make_adc(
            num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        gx_pre = make_trapezoid(
            channel='x', area=-gx.area / 2, system=system
        )

        # ky Phase Encode
        phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

        gy_pre = []
        gy_rew = []

        for ky_i in range(Ny):
            gy_pre.append(make_trapezoid(
                channel='y',
                area=phase_areas[ky_i],
                duration=calc_duration(gx_pre),
                system=system,
            ))

            gy_rew.append(make_trapezoid(
                channel='y',
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

        gz_enc_largest = make_trapezoid(channel='z', area=np.max(np.abs(areaZ)), system=system)
        Tz_enc = calc_duration(gz_enc_largest)

        # Gradient spoiling
        spoiler_area_peraxis = spoiler_area/sqrt(len(spoiler_axes))
        spoiler_grads = []

        for ax in spoiler_axes:
            spoiler_grads.append(make_trapezoid(channel=ax, area=spoiler_area_peraxis, system=system))
        
        # gx_spoil = make_trapezoid(channel='x', area=spoiler_area_peraxis, system=system)

        TEd = rnd2GRT(
            TE
            - calc_duration(gz)/2
            - calc_duration(gx_pre, gz_enc_largest)
            - calc_duration(gx)/2
        )

        assert np.all(TEd >= 0), f"Required TE can not be achieved. Min. TE is {TE-TEd} s."

        delay_TE = make_delay(TEd)

        # Register fixed grad events for speedup
        gx_pre.id = seq.register_grad_event(gx_pre)
        gx.id = seq.register_grad_event(gx)

        # Partition encoding preparation
        gz_enc = []
        gz_rew = []
        for kz_i in range(Nkz):
            gz_enc.append(
                make_trapezoid(channel='z',area=areaZ[kz_i], 
                duration=Tz_enc, system=system)
            )
            gz_rew.append(
                make_trapezoid(channel='z',area=-(areaZ[kz_i] + gz.area/2), 
                duration=Tz_enc, system=system)
            )

            gz_enc[kz_i].id = seq.register_grad_event(gz_enc[kz_i])
            gz_rew[kz_i].id = seq.register_grad_event(gz_rew[kz_i])

        # Store necessary variables in class
        self.seq = seq
        self.TE = TE
        self.rf = rf
        self.flip_angle = alpha
        self.gz = gz
        self.adc = adc
        self.gx = gx
        self.delay_TE = delay_TE
        self.gx_pre = gx_pre
        self.gy_pre = gy_pre
        self.gy_rew = gy_rew
        self.gz_enc = gz_enc
        self.gz_rew = gz_rew
        # self.gx_spoil = gx_spoil
        self.gspoilers = spoiler_grads


    def update_flip_angle(self, new_FA):
        old_fa = self.flip_angle
        self.rf.signal = new_FA*(self.rf.signal/old_fa)
        self.flip_angle = new_FA

    def duration(self):
        tot_dur = (
            self.TE
            + calc_duration(self.gz)/2
            + calc_duration(self.gx)/2
            # + calc_duration(self.gx_spoil, self.gz_rew[0], self.gy_rew[0])
            + calc_duration(self.gz_rew[0], self.gy_rew[0])
            + calc_duration(*(self.gspoilers))
        )
        return tot_dur

    def add_kernel(self, ky_i: int, kz_i: int, rf_spoil_upd: float, is_acq: bool = True, play_rf: bool = True):
        seq = self.seq

        # RF/ADC Phase update
        self.rf.phase_offset = rf_spoil_upd
        self.adc.phase_offset = rf_spoil_upd

        if play_rf:
            seq.add_block(self.rf, self.gz)
        else:
            seq.add_block(self.gz)

        seq.add_block(self.delay_TE)

        seq.add_block(self.gx_pre, self.gy_pre[ky_i], self.gz_enc[kz_i])

        if is_acq:
            seq.add_block(self.gx, self.adc)
        else:
            seq.add_block(self.gx)

        # seq.add_block(self.gx_spoil, self.gy_rew[ky_i], self.gz_rew[kz_i])
        seq.add_block(self.gy_rew[ky_i], self.gz_rew[kz_i])
        seq.add_block(*(self.gspoilers))

