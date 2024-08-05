import streamlit as st
from jax import numpy as jnp
#import photonic_cirq as p_cirq
import super_cirq as s_cirq
import qkdc_helper as helper
#import gaus_cirq as gaus
import numpy as np

logo = "./img/ui-streamlit-red.png"
gaus_logo = "./img/GausQash.png"
qash_logo = "./img/qash-red.png"

st.set_page_config(
    page_title="Qash-QKDC",
    page_icon=logo,
)

if 'output_hash' not in st.session_state:
    st.session_state.output_hash = ''

if 'grad_hash' not in st.session_state:
    st.session_state.grad_hash = ''

x64_jax = helper.x64Switch(True)

def clearOutput():
    st.session_state.output_hash = ''
    st.session_state.grad_hash = ''

def convertPepperToArr(pepp):
    pepper = pepp.split(",")
    pepper_arr = jnp.array([])
    for ch in pepper:
        if ch == "":
            pass
        else:
            pepper_arr = jnp.append(pepper_arr, float(ch))
    return pepper_arr

top_panel = st.container()
main_panel = st.container()
side_panel = st.sidebar

side_panel.title("Options")
backend_details = side_panel.toggle("Show Runtime Details")
seed_option = side_panel.number_input("Seed", value=0, format="%d")
pad_length_option = side_panel.number_input("Length of Input Padding", value=0, format="%d")
hash_precision = side_panel.selectbox("Hash Precision",('double','single'))
device_option = side_panel.selectbox("Device",('default','cirq'),on_change=clearOutput)
    if device_option != 'default':
        shots = side_panel.number_input("Shots", value=0, format="%d")
    else: 
        shots = None
"""
simulator_option = side_panel.selectbox("Simulator",('superconductor','fock', 'gaussian'),on_change=clearOutput) # 
if simulator_option == "superconductor":
    device_option = side_panel.selectbox("Device",('default','cirq'),on_change=clearOutput)
    if device_option != 'default':
        shots = side_panel.number_input("Shots", value=0, format="%d")
    else: 
        shots = None
elif simulator_option == "fock":
    shots = side_panel.number_input("Shots", value=1, min_value=1, format="%d")
output_mode = side_panel.selectbox("Output Mode",('hex','base64'))
if simulator_option != 'gaussian':
    pepper = side_panel.text_input("Pepper (comma-separated floats)", value="")
else:
    pepper = None
    shots = None
"""

def runHash(pepp, shots):
    with st.spinner("Loading Hash, Please Wait..."):
        if shots == 0:
            shots = None
        if pepp:
            pepper_arr = convertPepperToArr(pepp).astype("float64")
        else:
            pepper_arr = None
        num_wires = len(input_string)
        input = helper.createAndPad(input_string, pad_length_option, simulator_option)
        if simulator_option == 'superconductor':
            input = input.astype('float64')
            pepper = pepper_arr
            output = s_cirq.qxHashCirq(input, num_wires, seed_option, pepper, device_option, shots)
            st.session_state.output_hash = helper.processOutput(output, output_mode, hash_precision)
            st.session_state.grad_hash = helper.calcGradHash(output, output_mode, hash_precision)
        elif simulator_option == 'fock':
            input = jnp.array(input).astype('float64')
            pepper = pepper_arr
            output = p_cirq.qxBerryCirq(input, num_wires, pepper, shots)
            st.session_state.output_hash = helper.processOutput(output, output_mode, hash_precision)
            st.session_state.grad_hash = helper.calcGradHash(output, output_mode, hash_precision)
        elif simulator_option == 'gaussian':
            input = np.array(input).astype('float64')
            output = gaus.qxGausCirq(input, num_wires)
            out_list = np.array([])
            for i in range(num_wires):
                out_list = np.append(out_list, output.state.number_expectation(modes=[i]))
            st.session_state.output_hash = helper.processOutput(out_list, output_mode, hash_precision)
            st.session_state.grad_hash = helper.calcGradHash(out_list, output_mode, hash_precision)
        else:
            st.session_state.output_hash = "DEVICE ERROR"

if simulator_option == 'superconductor':
    main_panel.image(qash_logo, width=75)
    main_panel.header("Qash-QKDC (SuperConductor)", divider='rainbow')
elif simulator_option == 'fock':
    main_panel.image(qash_logo, width=75)
    main_panel.header("Qash-QKDC (Fock)", divider='rainbow')
else:
    main_panel.image(gaus_logo, width=75)
    main_panel.header("GausQash (Gaussian)", divider='rainbow')
input_string = main_panel.text_area("Enter String to Hash", value="", on_change=clearOutput)
output_string = main_panel.text_area("Output Hash Value", value=st.session_state.output_hash)
grad_string = main_panel.text_area("Gradient Hash Value", value=st.session_state.grad_hash)
run_button = main_panel.button("Run Hash Simulator", on_click=runHash, args=[pepper,shots])
if backend_details:
    main_panel.divider()
    if simulator_option == 'gaussian':
        main_panel.caption(f"compute mode: numpy")
        main_panel.caption(f"double precision mode: enabled")
        main_panel.caption(f"platform: cpu")
    else:
        main_panel.caption(f"compute mode: jax")
        main_panel.caption(f"{x64_jax}")
        main_panel.caption(f"platform: {helper.getBackend()}")





