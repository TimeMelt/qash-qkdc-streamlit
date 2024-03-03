import streamlit as st
from jax import numpy as jnp
import photonic_cirq as p_cirq
import super_cirq as s_cirq
import qkdc_helper as helper
import gaus_cirq as gaus
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

def clearOutput():
    st.session_state.output_hash = ''

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
simulator_option = side_panel.selectbox("Simulator",('superconductor','fock', 'gaussian'),on_change=clearOutput)
if simulator_option == "superconductor":
    device_option = side_panel.selectbox("Device",('default','cirq'),on_change=clearOutput)
output_mode = side_panel.selectbox("Output Mode",('hex','base64'))
if simulator_option != 'gaussian':
    pepper = side_panel.text_input("Pepper (comma-separated floats)", value="")
else:
    pepper = None

def runHash(pepp):
    with st.spinner("Loading Hash, Please Wait..."):
        if pepp:
            pepper_arr = convertPepperToArr(pepp)
        else:
            pepper_arr = None
        num_wires = len(input_string)
        input = helper.createAndPad(input_string, pad_length_option, simulator_option)
        if simulator_option == 'superconductor':
            input = input.astype('float64')
            if pepper_arr:
                pepper = pepper_arr.astype("float64")
            else:
                pepper = None
            output = s_cirq.qxHashCirq(input, num_wires, seed_option, pepper, device_option)
            st.session_state.output_hash = helper.processOutput(output, output_mode)
        elif simulator_option == 'fock':
            input = jnp.array(input).astype('float64')
            if pepper_arr:
                pepper = pepper_arr.astype("float64")
            else:
                pepper = None
            output = p_cirq.qxBerryCirq(input, num_wires, pepper)
            st.session_state.output_hash = helper.processOutput(output, output_mode)
        elif simulator_option == 'gaussian':
            input = np.array(input).astype('float64')
            output = gaus.qxGausCirq(input, num_wires)
            out_list = np.array([])
            for i in range(num_wires):
                out_list = np.append(out_list, output.state.number_expectation(modes=[i]))
            st.session_state.output_hash = helper.processOutput(out_list, output_mode)
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
run_button = main_panel.button("Run Hash Simulator", on_click=runHash, args=[pepper])
if backend_details:
    main_panel.divider()
    if simulator_option == 'gaussian':
        main_panel.caption(f"compute mode: numpy")
        main_panel.caption(f"double precision mode: enabled")
        main_panel.caption(f"platform: cpu")
    else:
        main_panel.caption(f"compute mode: jax")
        main_panel.caption(f"{helper.x64Switch(True)}")
        main_panel.caption(f"platform: {helper.getBackend()}")





