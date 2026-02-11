# =========================================
# STREAMLIT — POISSON SKYNET (HÍBRIDO)
# =========================================
import re
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_autorefresh import st_autorefresh
import glob
from PIL import Image

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="⚽🏆Poisson Skynet🏆⚽",
    layout="wide"
)

st.title("⚽🏆 Poisson Skynet 🏆⚽")

st.markdown("""
<style>

/* SELECTBOX */
.stSelectbox label {
    font-size: 22px !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    font-size: 20px !important;
}

/* MÉTRICAS */
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 900 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 🎬 BANNER CARROSSEL — DEFINITIVO (FUNCIONA MESMO)
# =========================================
from streamlit_autorefresh import st_autorefresh
import glob

BANNERS = sorted(glob.glob("assets/banner*.png"))

if not BANNERS:
    st.error("Nenhuma imagem encontrada em assets/banner*.png")
else:

    # 🔥 força rerun a cada 10s
    count = st_autorefresh(interval=10000, key="banner_refresh")

    # índice automático
    banner_idx = count % len(BANNERS)

    c1, c2, c3 = st.columns([1, 8, 1])

    # setas funcionam
    if "manual_idx" not in st.session_state:
        st.session_state.manual_idx = banner_idx

    with c1:
        if st.button("◀", use_container_width=True):
            st.session_state.manual_idx = (st.session_state.manual_idx - 1) % len(BANNERS)

    with c3:
        if st.button("▶", use_container_width=True):
            st.session_state.manual_idx = (st.session_state.manual_idx + 1) % len(BANNERS)

    # usa manual OU auto
    final_idx = st.session_state.manual_idx if st.session_state.manual_idx != banner_idx else banner_idx

    with c2:
        st.image(BANNERS[final_idx], use_container_width=True)

st.title("⚽🏆Poisson Skynet🏆⚽")

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
    st.success("📤 Utilizando arquivo enviado pelo usuário")

elif os.path.exists(ARQUIVO_PADRAO):
    xls = pd.ExcelFile(ARQUIVO_PADRAO)
    st.info("📊 Utilizando arquivo padrão")

else:
    st.error("❌ Nenhum arquivo disponível (nem upload nem padrão)")
    st.stop()

# =========================================
# LEITURA DAS ABAS
# =========================================
df_mgf = pd.read_excel(xls, "Poisson_Media_Gols")
df_exg = pd.read_excel(xls, "Poisson_Ataque_Defesa")
df_vg  = pd.read_excel(xls, "Poisson_VG")  # <<< FALTAVA ISSO

for df in (df_mgf, df_exg, df_vg):
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]
    
# 🔥 DEFINA AQUI (ANTES DAS TABS)
jogos_lista = df_mgf["JOGO"].tolist()

if "jogo" not in st.session_state or st.session_state["jogo"] not in jogos_lista:
    st.session_state["jogo"] = jogos_lista[0]

jogo = st.selectbox("⚽ Escolha o jogo", jogos_lista)
linha_mgf = df_mgf[df_mgf["JOGO"] == jogo].iloc[0]
linha_exg = df_exg[df_exg["JOGO"] == jogo].iloc[0]
linha_vg  = df_vg[df_vg["JOGO"] == jogo].iloc[0]  # <<< FALTAVA ISSO

st.session_state["jogo"] = jogo
# =========================================
# FUNÇÕES AUX
# =========================================
def get_val(linha, col, fmt=None, default="—"):
    if col in linha.index and pd.notna(linha[col]):
        try:
            return fmt.format(linha[col]) if fmt else linha[col]
        except Exception:
            return default
    return default


def calc_ev(odd_real, odd_justa):
    try:
        return (odd_real / odd_justa) - 1
    except Exception:
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

    m = (
        m.sort_values("Probabilidade%", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    m["Probabilidade%"] = m["Probabilidade%"].map(lambda x: f"{x:.2f}%")
    return m

# =========================================
# 🎨 ESTILO CARDS (NOVO)
# =========================================
def cor_card(txt):
    if not isinstance(txt, str):
        return "#2b2b2b"

    txt = txt.lower()

    if "domínio" in txt:
        return "#123524"   # verde
    if "favorito" in txt:
        return "#3a3a1a"   # amarelo
    if "btts" in txt or "aberto" in txt or "caos" in txt:
        return "#3a1414"   # vermelho

    return "#2b2b2b"       # neutro


def calcular_score(row):
    try:
        ph = 1 / row["Odd_Justa_Home"]
        pa = 1 / row["Odd_Justa_Away"]

        edge = abs(ph - pa)

        score = edge * 20   # normaliza 0–10
        return round(min(score, 10), 2)

    except:
        return 0
      
# =========================================
# 🎯 CARD REUTILIZÁVEL (BANNER INTERPRETAÇÃO)
# =========================================
def mostrar_card(df_base, jogo):

    if "Interpretacao" not in df_base.columns:
        return

    linha = df_base[df_base["JOGO"] == jogo]
    if linha.empty:
        return

    row = linha.iloc[0]

    score = calcular_score(row)
    estrelas = "⭐" * round(score / 2) + "☆" * (5 - round(score / 2))
    cor = cor_card(row["Interpretacao"])

    card = f"""
    <div style="
        background:{cor};
        padding:18px;
        border-radius:14px;
        box-shadow:0 0 10px rgba(0,0,0,0.45);
        color:white;
        font-size:18px;
        font-weight:600;
        margin-bottom:18px;
    ">
    🧠 {row['Interpretacao']}
    <br>
    <span style="font-size:26px;">{estrelas}</span>
    </div>
    """

    st.markdown(card, unsafe_allow_html=True)

# =========================================
# ABAS
# =========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"📊 Resumo",
"📁 Dados",
"🔢 MGF",
"⚔️ ATK x DEF",
"💰 VG"
])

# =========================================
# ABA 1 — RESUMO
# =========================================
with tab1:
    st.subheader(jogo)

    # -------- LINHA 1 — ODDS
    st.markdown("### 🎯 Odds")
    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_exg["Odds_Casa"], linha_exg["Odd_Justa_Home"])
        st.metric("Odds Casa", linha_exg["Odds_Casa"])

        st.metric("Odd_Over_1,5FT", linha_exg["Odd_Over_1,5FT"])
        st.metric("VR01", get_val(linha_exg, "VR01", "{:.2f}"))

    with o2:
        ev = calc_ev(linha_exg["Odds_Empate"], linha_exg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_exg["Odds_Empate"])
        st.metric("Odds_Over_2,5FT", linha_exg["Odds_Over_2,5FT"])
        st.metric("COEF_OVER1FT", get_val(linha_exg, "COEF_OVER1FT", "{:.2f}"))

    with o3:
        ev = calc_ev(linha_exg["Odds_Visitante"], linha_exg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odds_Under_2,5FT", linha_exg["Odds_Under_2,5FT"])
        st.metric("Odd_BTTS_YES", linha_exg["Odd_BTTS_YES"])

    st.markdown("---")

    # -------- LINHA 2 — Métricas
    st.markdown("### 📊Métricas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Placar Provável", get_val(linha_mgf, "Placar_Mais_Provavel"))
        st.metric("Posse Home (%)", get_val(linha_exg, "Posse_Bola_Home", "{:.2f}"))
        st.metric("PPJH", get_val(linha_exg, "PPJH", "{:.2f}"))
        st.metric("Media_CG_H_01", get_val(linha_mgf, "Media_CG_H_01", "{:.2f}"))
        st.metric("CV_CG_H_01", get_val(linha_mgf, "CV_CG_H_01", "{:.2f}"))

    with c2:
        st.metric("Posse Away (%)", get_val(linha_exg, "Posse_Bola_Away", "{:.2f}"))
        st.metric("PPJA", get_val(linha_exg, "PPJA", "{:.2f}"))
        st.metric("Media_CG_A_01", get_val(linha_mgf, "Media_CG_A_01", "{:.2f}"))
        st.metric("CV_CG_A_01", get_val(linha_mgf, "CV_CG_A_01", "{:.2f}"))
        st.metric("ExG_Home_MGF", get_val(linha_mgf, "ExG_Home_MGF", "{:.2f}"))

    with c3:
        st.metric("Força Ataque Home (%)", get_val(linha_exg, "FAH", "{:.2f}"))
        st.metric("Precisão Chutes H (%)", get_val(linha_exg, "Precisao_CG_H", "{:.2f}"))
        st.metric("Chutes H (Marcar)", get_val(linha_mgf, "CHM", "{:.2f}"))
        st.metric("MGF_H", get_val(linha_mgf, "MGF_H", "{:.2f}"))
        st.metric("CV_GF_H", get_val(linha_mgf, "CV_GF_H", "{:.2f}"))

    with c4:
        st.metric("Força Ataque Away (%)", get_val(linha_exg, "FAA", "{:.2f}"))
        st.metric("Precisão Chutes A (%)", get_val(linha_exg, "Precisao_CG_A", "{:.2f}"))
        st.metric("Chutes A (Marcar)", get_val(linha_mgf, "CAM", "{:.2f}"))
        st.metric("MGF_A", get_val(linha_mgf, "MGF_A", "{:.2f}"))
        st.metric("CV_GF_A", get_val(linha_mgf, "CV_GF_A", "{:.2f}"))

    with c5:
        st.metric("Força Defesa Home (%)", get_val(linha_exg, "FDH", "{:.2f}"))
        st.metric("Clean Games Home (%)", get_val(linha_exg, "Clean_Games_H"))
        st.metric("Chutes H (Sofrer)", get_val(linha_mgf, "CHS", "{:.2f}"))
        st.metric("MGC_H", get_val(linha_mgf, "MGC_H", "{:.2f}"))
        st.metric("CV_GC_H", get_val(linha_mgf, "CV_GC_H", "{:.2f}"))

    with c6:
        st.metric("Força Defesa Away (%)", get_val(linha_exg, "FDA", "{:.2f}"))
        st.metric("Clean Games Away (%)", get_val(linha_exg, "Clean_Games_A"))
        st.metric("Chutes A (Sofrer)", get_val(linha_mgf, "CAS", "{:.2f}"))
        st.metric("MGC_A", get_val(linha_mgf, "MGC_A", "{:.2f}"))
        st.metric("CV_GC_A", get_val(linha_mgf, "CV_GC_A", "{:.2f}"))


    # -------- LINHA 3 — MGF
    st.markdown("### 📊 MGF")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Placar Provável", get_val(linha_mgf, "Placar_Mais_Provavel"))

    with a2:
        st.metric("Clean Sheet Home (%)", get_val(linha_mgf, "Clean_Sheet_Home_%", "{:.2f}"))

    with a3:
        st.metric("Clean Sheet Away (%)", get_val(linha_mgf, "Clean_Sheet_Away_%", "{:.2f}"))

    # -------- LINHA 4 — ATK x DEF
    st.markdown("### ⚔️ Ataque x Defesa")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric("Placar Provável", get_val(linha_exg, "Placar_Mais_Provavel"))

    with e2:
        st.metric("ExG_Home_ATKxDEF", get_val(linha_exg, "ExG_Home_ATKxDEF", "{:.2f}"))

    with e3:
        st.metric("ExG_Away_ATKxDEF", get_val(linha_exg, "ExG_Away_ATKxDEF", "{:.2f}"))

    # -------- LINHA 5 — VG
    st.markdown("### 💰 Gols Value")
    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric("Placar Provável", get_val(linha_vg, "Placar_Mais_Provavel"))

    with b2:
        st.metric("ExG_Home_VG", get_val(linha_vg, "ExG_Home_VG", "{:.2f}"))

    with b3:
        st.metric("ExG_Away_VG", get_val(linha_vg, "ExG_Away_VG", "{:.2f}"))

# =========================================
# ABA 2 — DADOS COMPLETOS
# =========================================
with tab2:
    for aba in xls.sheet_names:
        with st.expander(aba):
            st.dataframe(
                pd.read_excel(xls, aba),
                use_container_width=True
            )


# =========================================
# ABA 3 — POISSON MGF
# =========================================
with tab3:

    mostrar_card(df_mgf, jogo)

    st.subheader(jogo)

    st.markdown("### 🎯 Odds Justas MGF")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_mgf["Odds_Casa"], linha_mgf["Odd_Justa_Home"])
        st.metric("Odds Casa", linha_mgf["Odds_Casa"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o2:
        ev = calc_ev(linha_mgf["Odds_Empate"], linha_mgf["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_mgf["Odds_Empate"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o3:
        ev = calc_ev(linha_mgf["Odds_Visitante"], linha_mgf["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_mgf["Odds_Visitante"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%")

    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_mgf["ExG_Home_MGF"],
        linha_mgf["ExG_Away_MGF"]
    )

    exibir_matriz(matriz,
                  linha_mgf["Home_Team"],
                  linha_mgf["Visitor_Team"],
                  "Poisson — MGF")

    st.dataframe(top_placares(matriz), use_container_width=True)
    

# =========================================
# ABA 4 — POISSON ATK x DEF
# =========================================
with tab4:

    mostrar_card(df_exg, jogo)

    st.subheader(jogo)

    st.markdown("### ⚔️ Odds & Modelo ATK x DEF")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_exg["Odds_Casa"], linha_exg["Odd_Justa_Home"])
        st.metric("Odds Casa", linha_exg["Odds_Casa"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o2:
        ev = calc_ev(linha_exg["Odds_Empate"], linha_exg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_exg["Odds_Empate"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o3:
        ev = calc_ev(linha_exg["Odds_Visitante"], linha_exg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%")

    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_exg["ExG_Home_ATKxDEF"],
        linha_exg["ExG_Away_ATKxDEF"]
    )

    exibir_matriz(matriz,
                  linha_exg["Home_Team"],
                  linha_exg["Visitor_Team"],
                  "Poisson — ATK x DEF")

    st.dataframe(top_placares(matriz), use_container_width=True)


# =========================================
# ABA 5 — VG
# =========================================
with tab5:

    mostrar_card(df_vg, jogo)

    st.subheader("💰 Valor do Gol (VG)")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_vg["Odds_Casa"], linha_vg["Odd_Justa_Home"])
        st.metric("Odds Casa", linha_vg["Odds_Casa"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o2:
        ev = calc_ev(linha_vg["Odds_Empate"], linha_vg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_vg["Odds_Empate"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")

    with o3:
        ev = calc_ev(linha_vg["Odds_Visitante"], linha_vg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_vg["Odds_Visitante"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%")

    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_vg["ExG_Home_VG"],
        linha_vg["ExG_Away_VG"]
    )

    exibir_matriz(matriz,
                  linha_vg["Home_Team"],
                  linha_vg["Visitor_Team"],
                  "Poisson — Valor do Gol (VG)")

    st.dataframe(top_placares(matriz), use_container_width=True)
