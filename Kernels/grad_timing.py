from math import ceil

def rnd2GRT(t:float, GRT=1e-5):
    return GRT*round(t/GRT)

def ceil2GRT(t:float, GRT=1e-5):
    return GRT*ceil(t/GRT)