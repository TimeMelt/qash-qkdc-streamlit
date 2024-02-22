<p align='center'><img src="img/ui-streamlit-red.png" width="250"></p>

# qash-qkdc-streamlit
streamlit ui for qash-qkdc (quantum key derivation circuits)

### Web UI (streamlit.app): 
[![qash-qkdc-ui](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qkdc-ui.streamlit.app/)
  
### Local Deployment Instructions:
- clone this repo
- navigate to cloned repository
- install packages from requirements.txt and run command below (inside python env)
  - streamlit run main.py

### Jupyter Notebook:
- this streamlit app is based on the jupyter notebook provided [here](https://github.com/TimeMelt/qash-qkdc)
 
#### Credits:
- ui libraries provided by [Streamlit](https://github.com/streamlit/streamlit)
- quantum libraries provided by [PennyLane](https://github.com/PennyLaneAI/pennylane): 
    - [PennyLane research paper](https://arxiv.org/abs/1811.04968): 
        
            Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968
- acceleration using [JAX]("https://github.com/google/jax") library: 
    - JAX citation:

            @software{
                jax2018github,
                author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
                title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
                url = {http://github.com/google/jax},
                version = {0.3.13},
                year = {2018},
            }
