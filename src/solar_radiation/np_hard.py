# it's like numpy with some functions switched out to be casadi-friendly

from numpy import *

def rad2deg(x):
    return x / pi * 180.0

def radians(x):
    return x / 180 * pi

def degrees(x):
    return rad2deg(x)