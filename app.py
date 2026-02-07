# =========================================
# STREAMLIT — POISSON SKYNET (HÍBRIDO + DASHBOARD)
# =========================================

import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="⚽🏆 Poisson Skynet 🏆⚽", layout="wide")
st.title("⚽🏆 Poisson Skynet 🏆⚽")

# =========================================
# HÍBRIDO — ARQUIVO PADRÃO + UPLOAD OPCIONAL
# =========================================
ARQUIVO_PADRAO = "data/POISSON_DUAS_MATRIZES.xlsx"

with st.sidebar:
    st.header("📂 Dados")
    arquivo_upload = st.file_uploader("Enviar outro Excel (opcional)", type=["xlsx"])

# =========================================
# LEITURA DO ARQUIVO (BYTES)
# =========================================
if arquivo_upload:
    file_bytes = arquivo_upload.getvalue()
    st.success("📤 Arquivo enviado pelo usuário")
elif os.path.exists(ARQUIVO_PADRAO):
    with open(ARQUIVO_PADRAO, "rb") as f:
        file_bytes = f.read()
    st.info("📊 Arquivo padrão do dia")
else:
    st.error("❌ Nenhum arquivo disponível")
    st.stop()

# =========================================
# CACHE — LEITURA
# =========================================
@st.cache_data
def carregar_dados_bytes(file_bytes):
    xls = pd.ExcelFile(file_bytes)
    df_mgf = pd.read_excel(xls, "Poisson_Media_Gols")
    df_exg = pd.read_excel(xls, "Poisson_Ataque_Defesa")
    return df_mgf, df_exg

df_mgf, df_exg = carregar_dados_bytes(file_bytes)

for df in (df_mgf, df_exg):
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]

# =========================================
# CACHE — RADAR (SOMENTE EV CASA)
# =========================================
@st.cache_data
def montar_radar(df_exg):
    df = df_exg.copy()
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]

    if {"Odds_Casa", "Odd_Justa_Home"}.issubset(df.columns):
        df["EV_CASA"] = (df["Odds_Casa"] / df["Odd_Justa_Home"]) - 1
    else:
        df["EV_CASA"] = np.nan

    return df

df_radar = montar_radar(df_exg)

# =========================================
# FUNÇÕES AUX
# =========================================
def calc_ev(odd_real, odd_justa):
    try:
        return (odd_real / odd_justa) - 1
    except:
        return None

def calcular_matriz_poisson(lh, la, max_gols=4):
    matriz = np.zeros((max_gols + 1, max_gols + 1))
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            matriz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    return matriz * 100

def exibir_matriz(matriz, home, away, titulo):
    df = pd.DataFrame(matriz)
    fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=120)
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", cbar=False, ax=ax)
    ax.set_xlabel(away)
    ax.set_ylabel(home)
    st.pyplot(fig)
    plt.close(fig)

def top_placares(matriz, n=6):
    df = pd.DataFrame(matriz)
    m = df.reset_index().melt(id_vars="index")
    m.columns = ["Gols_Home", "Gols_Away", "Probabilidade%"]
    return m.sort_values("Probabilidade%", ascending=False).head(n)

def sinal(ev):
    if ev > 0.05:
        return "🟢 Value"
    elif ev > 0:
        return "🟡 Leve"
    else:
        return "🔴 Negativo"

# =========================================
# ABAS
# =========================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📈⚽ Visão Geral",
    "📊🎯 Resumo",
    "📁🏆 Dados",
    "🔢⚽ Poisson — MGF",
    "⚔️⚽ Poisson — ATK x DEF"
])

# =========================================
# ABA 0 — VISÃO GERAL
# =========================================
with tab0:
    st.subheader("📈 Radar de Oportunidades — EV Casa")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jogos", len(df_radar))
    c2.metric("EV +", (df_radar["EV_CASA"] > 0).sum())
    c3.metric("Melhor EV", f"{df_radar['EV_CASA'].max()*100:.2f}%")

    min_ev = st.slider("EV mínimo (%)", -20.0, 50.0, 0.0) / 100
    top_n = st.selectbox("Ranking", [5, 10, 15, 20])

    df_f = df_radar[df_radar["EV_CASA"] >= min_ev].copy()
    df_f["SINAL"] = df_f["EV_CASA"].apply(sinal)
    df_f["EV_%"] = df_f["EV_CASA"].map(lambda x: f"{x*100:.2f}%")

    ranking = df_f.sort_values("EV_CASA", ascending=False).head(top_n)[
        ["JOGO", "Odds_Casa", "Odd_Justa_Home", "EV_%", "SINAL"]
    ]

    st.dataframe(ranking, use_container_width=True)

    jogo_detalhe = st.selectbox("🎯 Abrir jogo no detalhe", ranking["JOGO"].unique())
    if st.button("🎯 Abrir jogo"):
        st.session_state["jogo_selecionado"] = jogo_detalhe
        st.experimental_rerun()

    st.markdown("### 🔥 Heatmap — EV Casa")
    heat = df_f.set_index("JOGO")[["EV_CASA"]]

    fig, ax = plt.subplots(figsize=(6, max(4, len(heat) * 0.25)))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# =========================================
# ABA 1 — RESUMO JOGO A JOGO
# =========================================
with tab1:
    jogos = df_mgf["JOGO"].tolist()
    jogo = st.selectbox(
        "⚽ Escolha o jogo",
        jogos,
        index=jogos.index(st.session_state.get("jogo_selecionado", jogos[0]))
    )

    linha_mgf = df_mgf[df_mgf["JOGO"] == jogo].iloc[0]
    linha_exg = df_exg[df_exg["JOGO"] == jogo].iloc[0]

    st.subheader(jogo)

    ev = calc_ev(linha_exg["Odds_Casa"], linha_exg["Odd_Justa_Home"])
    st.metric("Odds Casa", linha_exg["Odds_Casa"])
    st.metric("Odd Justa Casa", linha_exg["Odd_Justa_Home"])
    st.metric("EV Casa", f"{ev*100:.2f}%" if ev is not None else "—")

# =========================================
# ABA 2 — DADOS
# =========================================
with tab2:
    st.dataframe(df_exg, use_container_width=True)

# =========================================
# ABA 3 — POISSON MGF
# =========================================
with tab3:
    matriz = calcular_matriz_poisson(
        linha_mgf["ExG_Home_MGF"],
        linha_mgf["ExG_Away_MGF"]
    )
    exibir_matriz(matriz, linha_mgf["Home_Team"], linha_mgf["Visitor_Team"], "Poisson — MGF")
    st.dataframe(top_placares(matriz), use_container_width=True)

# =========================================
# ABA 4 — POISSON ATK x DEF
# =========================================
with tab4:
    matriz = calcular_matriz_poisson(
        linha_exg["ExG_Home_ATKxDEF"],
        linha_exg["ExG_Away_ATKxDEF"]
    )
    exibir_matriz(matriz, linha_exg["Home_Team"], linha_exg["Visitor_Team"], "Poisson — ATK x DEF")
    st.dataframe(top_placares(matriz), use_container_width=True)
