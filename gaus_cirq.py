import strawberryfields as sf
from strawberryfields import ops
import qkdc_gaus

def qxGausCirq(input, num_wires):
    program = sf.Program(num_wires)
    engine = sf.Engine("gaussian", backend_options={})

    with program.context as wires:
        qkdc_gaus.setThermalState(input, wires)
        qkdc_gaus.rotationStep(input, wires)
        qkdc_gaus.displaceStep(input, wires)
        qkdc_gaus.thermalLoss(input, wires)
        qkdc_gaus.beamSplitter(input, wires)
        ops.MeasureThreshold() | wires
    
    return engine.run(program, shots=1)