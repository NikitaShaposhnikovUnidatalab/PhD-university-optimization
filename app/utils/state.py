import streamlit as st

QS_INPUT = {"AR": 6.5, "ER": 10.6, "FSR": 54.3, "CPF": 1.3, "IFR": 1.7, "ISR": 20.1, "IRN": 11.4, "EO": 4.0, "SUS": 1.6}
QS_WEIGHTS = {"AR": 0.30, "ER": 0.15, "FSR": 0.10, "CPF": 0.20, "IFR": 0.05, "ISR": 0.05, "IRN": 0.05, "EO": 0.05, "SUS": 0.05}
QS_MAX = {"AR": 15, "ER": 20, "FSR": 70, "CPF": 3, "IFR": 12, "ISR": 30, "IRN": 30, "EO": 15, "SUS": 10}
QS_DELTA = {"AR": 1.0, "ER": 1.0, "FSR": 1.0, "CPF": 0.3, "IFR": 2.0, "ISR": 1.0, "IRN": 5.0, "EO": 2.0, "SUS": 1.0}
QS_COST = {"AR": 50, "ER": 45, "FSR": 20, "CPF": 15, "IFR": 30, "ISR": 50, "IRN": 10, "EO": 10, "SUS": 5}
MAX_RU = 100 

def init_state_obj(name: str, data: dict):
    if name not in st.session_state:
        st.session_state[name] = {}
    for k, v in data.items():
        if k not in st.session_state[name]:
            st.session_state[name][k] = v

def init_state_value(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value
