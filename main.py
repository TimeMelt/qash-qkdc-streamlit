import streamlit as st
from jax import numpy as jnp
import photonic_cirq as p_cirq
import super_cirq as s_cirq
import qkdc_helper as helper
import numpy as np

logo = "./img/ui-streamlit-red.png"

st.set_page_config(
    page_title="Qash-QKDC",
    page_icon=logo
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

st.image(logo, width=75)
main_panel = st.container()
side_panel = st.sidebar

side_panel.title("Options")
backend_details = side_panel.toggle("Show Runtime Details")
seed_option = side_panel.number_input("Seed", value=0, format="%d")
pad_length_option = side_panel.number_input("Length of Input Padding", value=0, format="%d")
simulator_option = side_panel.selectbox("Simulator",('superconductor','photonic'))
output_mode = side_panel.selectbox("Output Mode",('hex','base64'))
pepper = side_panel.text_input("Pepper (comma-separated floats)", value="")

def runHash(pepp):
    with st.spinner("Loading Hash, Please Wait..."):
        pepper_arr = convertPepperToArr(pepp)
        num_wires = len(input_string)
        input = helper.createAndPad(input_string, pad_length_option)
        if simulator_option == 'superconductor':
            input = input.astype('float64')
            pepper = pepper_arr.astype("float64")
            output = s_cirq.qxHashCirq(input, num_wires, seed_option, pepper)
        elif simulator_option == 'photonic':
            input = np.array(input).astype('float64')
            pepper = np.array(pepper_arr).astype('float64')
            output = p_cirq.qxBerryCirq(input, num_wires, pepper)
        st.session_state.output_hash = helper.processOutput(output, output_mode)

main_panel.header("Qash-QKDC (Simulator)", divider='rainbow')
input_string = main_panel.text_area("Enter String to Hash", value="", on_change=clearOutput)
output_string = main_panel.text_area("Output Hash Value", value=st.session_state.output_hash)
run_button = main_panel.button("Run Hash Simulator", on_click=runHash, args=[pepper])
if backend_details:
    main_panel.divider()
    main_panel.caption(helper.x64Switch(True))
    main_panel.caption(f"platform: {helper.getBackend()}")





