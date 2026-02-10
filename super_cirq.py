import pennylane as qml
from jax import random, jit
import qkdc_electron as electron
from functools import partial
import qkdc_helper as helper

@partial(jit, static_argnames=['num_wires','device','shots'])
def qxHashCirq(input, num_wires, seed, pepper, device, shots):
    key = random.PRNGKey(seed)
    if device == 'default':
        qdev = qml.device('default.qubit', wires=num_wires, shots=shots, prng_key=key)
    else:
        qdev = qml.device('cirq.simulator', wires=num_wires, shots=shots)

    @qml.qnode(qdev, interface="jax")
    def cirq(input, pepper, key):
        if pepper is None:
            electron.superPos(input)
        else:
            electron.angleEmbed(input,pepper)
        electron.rotLoop(input)
        electron.singleX(input)
        electron.qutritLoop(input)
        electron.strongTangle(input, key)
        electron.rotLoop(input)
        return [qml.var(qml.PauliZ(wires=i)) for i in range(num_wires)]
    
    return cirq(input, pepper, key)
