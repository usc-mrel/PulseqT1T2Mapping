from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.calc_duration import calc_duration
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.Sequence.sequence import Sequence

from utils.grad_timing import rnd2GRT
from math import pi

class IRKernel:
    def __init__(self, seq: Sequence, slice_thickness: float, TI: float) -> None:
        rf180, gz180, _ = make_sinc_pulse(flip_angle=pi, system=seq.system, 
                                  duration=2.5e-3, 
                                  slice_thickness=2*slice_thickness, 
                                  apodization=0.5, 
                                time_bw_product=4, phase_offset=pi/2, 
                                return_gz=True, use="inversion")
        
        gz_spoil = make_trapezoid(channel='z', system=seq.system, area=4/slice_thickness)


        TId = (TI 
            - calc_duration(rf180) - gz180.fall_time 
            - calc_duration(gz_spoil)
            )
        delay_TI = make_delay(rnd2GRT(TId))

        self.rf = rf180
        self.gz = gz180
        self.gss = gz_spoil
        self.delay = delay_TI
        self.seq = seq
        self.TI = TI
    
    def duration(self) -> float:
        return rnd2GRT(self.TI + self.gz.rise_time)

    def add_kernel(self):
        self.seq.add_block(self.rf, self.gz)
        self.seq.add_block(self.gss)
        self.seq.add_block(self.delay)