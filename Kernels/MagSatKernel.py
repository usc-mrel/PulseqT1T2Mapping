# -------------------------------------------------------------------
# Title: Magnetization saturation (reset) kernel
# Author: Bilal Tasdelen <tasdelen at usc dot edu>
# -------------------------------------------------------------------

# TODO: Test the kernel

from pypulseq.calc_duration import calc_duration
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.Sequence.sequence import Sequence
from math import pi

class MagSatKernel:
    def __init__(self, seq: Sequence, reset_type: bool, crasher_area: float) -> None:
        system = seq.system

        if reset_type == 'bir4':
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


            t_rf = calc_duration(rf_res)

        elif reset_type == 'composite':
            from pypulseq.make_block_pulse import make_block_pulse
            # Create an RF event
            rf45x = make_block_pulse(duration=0.3e-3, flip_angle=pi/4, phase_offset=0, system=system, use="saturation")
            rf90y = make_block_pulse(duration=0.6e-3, flip_angle=pi/2, phase_offset=pi/2, system=system, use="saturation")
            rf90x = make_block_pulse(duration=0.6e-3, flip_angle=pi/2, phase_offset=0, system=system, use="saturation")
            rf45y = make_block_pulse(duration=0.3e-3, flip_angle=pi/4, phase_offset=-pi/2, system=system, use="saturation")
            rf_ringdown_delay = make_delay(system.rf_ringdown_time + system.rf_dead_time)

            t_rf = (
                  calc_duration(rf45x) + calc_duration(rf_ringdown_delay)
                + calc_duration(rf90y) + calc_duration(rf_ringdown_delay)
                + calc_duration(rf90x) + calc_duration(rf_ringdown_delay)
                + calc_duration(rf45y)
            )
            self.rfs = (rf45x, rf90y, rf90x, rf45y)
            self.rf_delay = rf_ringdown_delay
        else:
            print("Unsupported reset type..")
            return


        gx_res = make_trapezoid(channel='x', area=crasher_area, system=system)
        gy_res = make_trapezoid(channel='y', area=crasher_area, system=system)
        gz_res = make_trapezoid(channel='z', area=crasher_area, system=system)
        
        gx_res.id = seq.register_grad_event(gx_res)
        gy_res.id = seq.register_grad_event(gy_res)
        gz_res.id = seq.register_grad_event(gz_res)
        t_reset = t_rf + calc_duration(gx_res, gy_res, gz_res)

        self.t_reset = t_reset
        self.reset_type = reset_type

        self.gx = gx_res
        self.gy = gy_res
        self.gz = gz_res

    def duration(self) -> float:
        return self.t_reset

    def add_kernel(self, phase_offset: float = 0):
        if self.reset_type == 'bir4':
            self.rfs[0].phase_offset = phase_offset
            self.seq.add_block(self.gx, self.gy, self.gz)
        elif self.reset_type == 'composite':
            # Set additional phases, align them 90 diff with excitation
            self.rfs[0].phase_offset = phase_offset
            self.rfs[1].phase_offset = pi/2 + phase_offset
            self.rfs[2].phase_offset = phase_offset
            self.rfs[3].phase_offset = -pi/2 + phase_offset

            self.seq.add_block(self.rfs[0])
            self.seq.add_block(self.rf_delay)
            self.seq.add_block(self.rfs[1])
            self.seq.add_block(self.rf_delay)
            self.seq.add_block(self.rfs[2])
            self.seq.add_block(self.rf_delay)
            self.seq.add_block(self.rfs[3])
            self.seq.add_block(self.gx, self.gy, self.gz)
