import streamlit as st
import pandas as pd

arquivo = st.file_uploader(
    "📂 Envie o arquivo Excel",
    type=["xlsx"]
)

if arquivo is None:
    st.warning("Envie o arquivo para iniciar a análise")
    st.stop()

df = pd.read_excel(arquivo)
