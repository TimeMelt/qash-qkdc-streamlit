import pennylane as qml
import numpy as np

# prepare coherent state
def prepareCohState(input, pepper):
    if pepper.size == 0:
        pepper = [0.0] * input.size
    elif pepper.size < input.size:
        for l in range(input.size - pepper.size):
            pepper = np.append(pepper, 0.0)
    for q in range(input.size):
        a = pepper[q]
        phi = np.tan(a)
        qml.CoherentState(a, phi, q)

# induce thermal state
def thermalState(input):
    for c,q in enumerate(input):
        nbar = q * np.tan(q)
        qml.ThermalState(nbar, wires=c)

# displacement step
def displaceStep(input):
    for q in range(input.size):
        if q == 0:
            dx = input.size
        else:
            dx = input.size * (q+1)
        qml.Displacement(input.size, dx, wires=q)

# cubic phase rotations
def cubicPhase(input):
    for c,q in enumerate(input):
        gamma = q * np.tan(q)
        qml.CubicPhase(gamma, wires=c)

# cross kerr interactions
def crossKerr(input):
    for c,q in enumerate(input):
        theta = q * (input.size * np.tan(q))
        if c == 0:
            pass
        else:
            qml.CrossKerr(theta, wires=[0, c])

# photonic rotations
def photonRotate(input):
    for c,q in enumerate(input):
        phi = q * np.tan(q)
        qml.Rotation(phi, wires=c)

# beamsplitter interactions
def beamSplit(input):
    for c,q in enumerate(input):
        phi = q * (input.size * np.tan(q))
        theta = q * (input.size * np.tan(q))
        if c == 0:
            pass
        else:
            qml.Beamsplitter(theta, phi, wires=[0, c])