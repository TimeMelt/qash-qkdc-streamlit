import pennylane as qml
from jax import random
import qkdc_electron as electron

def qxHashCirq(input, num_wires, seed, pepper):
    key = random.PRNGKey(seed)
    qdev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(qdev, interface="jax")
    def cirq(input, pepper, key):
        if pepper.size == 0:
            electron.superPos(input)
        else:
            electron.angleEmbed(input,pepper)
        electron.rotLoop(input)
        electron.singleX(input)
        electron.qutritLoop(input)
        electron.strongTangle(input, key)
        electron.rotLoop(input)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_wires)]
    
    return cirq(input, pepper, key)