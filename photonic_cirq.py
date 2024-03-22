import pennylane as qml
import qkdc_photon as photon 

def qxBerryCirq(input, num_wires, pepper, shots):
    berry_device = qml.device('strawberryfields.fock', wires=num_wires, cutoff_dim=2, shots=shots)

    @qml.qnode(berry_device, interface="jax")
    def cirq(input, pepper):
        photon.prepareCohState(input, pepper)
        photon.thermalState(input)
        photon.photonRotate(input)
        photon.displaceStep(input)
        photon.beamSplit(input)
        photon.crossKerr(input)
        photon.cubicPhase(input)
        return [qml.expval(qml.NumberOperator(wires=i)) for i in range(num_wires)]

    return cirq(input, pepper)