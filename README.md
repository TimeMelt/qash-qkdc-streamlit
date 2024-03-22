<p align='center'><img src="img/ui-streamlit-red.png" width="250"></p>

# qash-qkdc-streamlit
streamlit ui for qash-qkdc (quantum key derivation circuits)

### Web UI (streamlit.app): 
[![qash-qkdc-ui](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qkdc-ui.streamlit.app/)

### Updates:
- support for gradient calculation
- single and double precision modes 
- google cirq simulator now supported
    - options -> device -> cirq
- gaussian hash circuit mode now available
    - gaussian implementation using strawberryfields sdk
    - options -> simulator -> gaussian
- partial JIT implemented for superconductor circuit
    - allows for faster performance
  
### Local Deployment Instructions:
- activate python environment
- execute commands below...
    
      git clone https://github.com/TimeMelt/qash-qkdc-streamlit.git
      cd qash-qkdc-streamlit
      pip install -r requirements.txt
      streamlit run main.py

### Jupyter Notebook:
- this streamlit app is based on the [qash-qkdc](https://github.com/TimeMelt/qash-qkdc) jupyter notebook
- gaussian mode based on [GausQash](https://github.com/TimeMelt/GausQash) jupyter notebook
 
#### Credits:
- ui libraries provided by [Streamlit](https://github.com/streamlit/streamlit)
- quantum libraries provided by [PennyLane](https://github.com/PennyLaneAI/pennylane): 
    - [PennyLane research paper](https://arxiv.org/abs/1811.04968): 

        > Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

- acceleration using [JAX](https://github.com/google/jax) library: 
    > jax2018github,
    > author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
    > title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
    > url = {http://github.com/google/jax},
    > version = {0.3.13},
    > year = {2018},

- GausQash quantum libraries provided by [StrawberryFields](https://github.com/XanaduAI/strawberryfields):
    > Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and
    > Christian Weedbrook. "Strawberry Fields: A Software Platform for Photonic Quantum Computing",
    > [Quantum, 3, 129](https://quantum-journal.org/papers/q-2019-03-11-129/) (2019).

    > Thomas R. Bromley, Juan Miguel Arrazola, Soran Jahangiri, Josh Izaac, Nicolás Quesada,
    > Alain Delgado Gran, Maria Schuld, Jeremy Swinarton, Zeid Zabaneh, and Nathan Killoran.
    > "Applications of Near-Term Photonic Quantum Computers: Software and Algorithms",
    > [Quantum Sci. Technol. 5 034010](https://iopscience.iop.org/article/10.1088/2058-9565/ab8504/meta) (2020).
