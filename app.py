# =========================================
# STREAMLIT — POISSON SKYNET (HÍBRIDO)
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
st.set_page_config(page_title="Poisson Skynet ⚽", layout="wide")
st.title("⚽ Poisson Skynet — Jogo a Jogo")

# =========================================
# HÍBRIDO — ARQUIVO PADRÃO + UPLOAD OPCIONAL
# =========================================
ARQUIVO_PADRAO = "data/POISSON_DUAS_MATRIZES.xlsx"

with st.sidebar:
    st.header("📂 Dados")
    arquivo_upload = st.file_uploader(
        "Enviar outro Excel (opcional)",
        type=["xlsx"]
    )

if arquivo_upload:
    xls = pd.ExcelFile(arquivo_upload)
    st.success("📤 Usando arquivo enviado pelo usuário")
elif os.path.exists(ARQUIVO_PADRAO):
    xls = pd.ExcelFile(ARQUIVO_PADRAO)
    st.info("📊 Usando arquivo padrão do repositório")
else:
    st.error("❌ Nenhum arquivo disponível (nem upload nem padrão)")
    st.stop()

# =========================================
# LEITURA DAS ABAS
# =========================================
df_mgf = pd.read_excel(xls, "Poisson_Media_Gols")
df_exg = pd.read_excel(xls, "Poisson_Ataque_Defesa")

for df in (df_mgf, df_exg):
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]

# =========================================
# SELEÇÃO DE JOGO
# =========================================
jogo = st.selectbox("⚽ Escolha o jogo", df_mgf["JOGO"].unique())

linha_mgf = df_mgf[df_mgf["JOGO"] == jogo].iloc[0]
linha_exg = df_exg[df_exg["JOGO"] == jogo].iloc[0]

# =========================================
# FUNÇÕES AUX
# =========================================
def get_val(linha, col, fmt=None, default="—"):
    if col in linha.index and pd.notna(linha[col]):
        try:
            return fmt.format(linha[col]) if fmt else linha[col]
        except:
            return default
    return default

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
    df = pd.DataFrame(
        matriz,
        index=[str(i) for i in range(matriz.shape[0])],
        columns=[str(i) for i in range(matriz.shape[1])]
    )

    st.subheader(titulo)

    fig = plt.figure(figsize=(3.2, 2.8), dpi=120)
    ax = fig.add_axes([0.12, 0.18, 0.78, 0.72])

    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        square=True,
        cbar=False,
        linewidths=0.3,
        annot_kws={"size": 7},
        ax=ax
    )

    ax.set_xlabel(away, fontsize=8)
    ax.set_ylabel(home, fontsize=8)
    ax.tick_params(labelsize=7)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

def top_placares(matriz, n=6):
    df = pd.DataFrame(matriz)
    m = df.reset_index().melt(id_vars="index")
    m.columns = ["Gols_Home", "Gols_Away", "Probabilidade%"]
    m = m.sort_values("Probabilidade%", ascending=False).head(n).reset_index(drop=True)
    m["Probabilidade%"] = m["Probabilidade%"].map(lambda x: f"{x:.2f}%")
    return m

# =========================================
# ABAS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumo",
    "📁 Dados Completos",
    "🔢 Poisson — Média de Gols",
    "⚔️ Poisson — Ataque x Defesa"
])

# =========================================
# ABA 1 — RESUMO
# =========================================
with tab1:
    st.subheader(jogo)

    st.markdown("### 🎯 Odds")
    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_exg["Odds_Casa"], linha_exg["Odd_Justa_Home"])
        st.metric("Odds Casa", linha_exg["Odds_Casa"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%" if ev is not None else "—")

    with o2:
        ev = calc_ev(linha_exg["Odds_Empate"], linha_exg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_exg["Odds_Empate"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%" if ev is not None else "—")

    with o3:
        ev = calc_ev(linha_exg["Odds_Visitante"], linha_exg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%" if ev is not None else "—")

# =========================================
# ABA 2 — DADOS COMPLETOS
# =========================================
with tab2:
    for aba in xls.sheet_names:
        with st.expander(aba):
            st.dataframe(pd.read_excel(xls, aba), use_container_width=True)

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
