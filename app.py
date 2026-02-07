import streamlit as st
import pandas as pd

arquivo = st.file_uploader(
    "📂 Envie o arquivo do dia",
    type=["xlsx"]
)

if arquivo is None:
    st.warning("Arquivo não enviado para iniciar a análise")
    st.stop()

df = pd.read_excel(arquivo)
