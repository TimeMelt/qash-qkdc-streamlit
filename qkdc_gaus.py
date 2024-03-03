import strawberryfields as sf
from strawberryfields import ops
import numpy as np

def setThermalState(input, wires):
    for c,q in enumerate(input):
        n = np.tan(q) * np.cos(q)
        ops.Thermal(n) | wires[c]

def rotationStep(input, wires):
    for c,q in enumerate(input):
        phi = q * np.tan(q)
        ops.Rgate(phi) | wires[c]

def displaceStep(input, wires):
    for c,q in enumerate(input):
        r = q
        phi = np.tan(r) * np.cos(r)
        ops.Dgate(r,phi) | wires[c]

def thermalLoss(input, wires):
    for c,q in enumerate(input):
        T = 0.25
        nbar = np.tan(q)
        ops.ThermalLossChannel(T, nbar) | wires[c]

def beamSplitter(input, wires):
    for c,q in enumerate(input):
        theta = q * (input.size * np.tan(q))
        phi = q * (input.size * np.tan(q))
        if c == input.size or c == input.size-1:
            pass
        else:
            ops.BSgate(theta,phi) | [wires[c], wires[c+1]]