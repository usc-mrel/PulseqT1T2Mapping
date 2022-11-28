from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.calc_duration import calc_duration
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_gradients import add_gradients
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse

from utils.grad_timing import rnd2GRT
from math import pi

class IRKernel:
    def __init__(self, seq: Sequence, slice_thickness: float, TI: float) -> None:
        # rf180, gz180, _ = make_sinc_pulse(flip_angle=pi, system=seq.system, 
        #                           duration=2.5e-3, 
        #                           slice_thickness=slice_thickness, 
        #                           apodization=0.5, 
        #                         time_bw_product=4, phase_offset=0, 
        #                         return_gz=True, use="inversion")

        rf180, gz180, _ = make_adiabatic_pulse(
            pulse_type='hypsec', 
            duration=8e-3, beta=800, 
            slice_thickness=slice_thickness, mu=4.9,
            return_gz=True, system=seq.system, use="inversion"
            )
        
        gz_spoil1 = make_trapezoid(channel='z', system=seq.system, area=4/slice_thickness, delay=gz180.rise_time+gz180.flat_time+gz180.delay)

        gz1 = add_gradients([gz180, gz_spoil1])

        gz_spoil = make_trapezoid(channel='z', system=seq.system, area=-4/slice_thickness)

        gx_spoil = make_trapezoid(channel='x', system=seq.system, area=-4/slice_thickness)
        gy_spoil = make_trapezoid(channel='y', system=seq.system, area=-4/slice_thickness)


        TId = (TI 
            - (calc_duration(gz1) - gz180.rise_time)
            - calc_duration(gx_spoil, gy_spoil, gz_spoil)
            )
        delay_TI = make_delay(rnd2GRT(TId))

        # Register grad events
        # gz1.id = seq.register_grad_event(gz1)  # TODO: Arbitrary gradients have an issue with register, have a look.
        gx_spoil.id = seq.register_grad_event(gx_spoil)
        gy_spoil.id = seq.register_grad_event(gy_spoil)
        gz_spoil.id = seq.register_grad_event(gz_spoil)
        
        self.t_kernel = rnd2GRT(TI + gz180.rise_time)

        self.rf = rf180
        self.gz = gz1
        self.gzs = gz_spoil
        self.gys = gy_spoil
        self.gxs = gx_spoil

        self.delay = delay_TI
        self.seq = seq
        self.TI = TI
    
    def duration(self) -> float:
        return self.t_kernel

    def add_kernel(self):
        self.seq.add_block(self.rf, self.gz)
        # self.seq.add_block(self.gzs)
        self.seq.add_block(self.delay)
        self.seq.add_block(self.gxs, self.gys, self.gzs)
