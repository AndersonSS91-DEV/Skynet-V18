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

/* ===== Fonte Global ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* ===== Título Principal ===== */
h1 {
    font-size: 46px !important;
    font-weight: 900 !important;
}

/* ===== Subheader ===== */
h3 {
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* ===== MÉTRICAS (SELETOR CERTO) ===== */

/* Label */
div[data-testid="metric-container"] label {
    font-size: 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    opacity: 0.6 !important;
    font-weight: 700 !important;
}

/* Valor */
div[data-testid="metric-container"] > div {
    font-size: 48px !important;
    font-weight: 900 !important;
}

/* ===== TABS ===== */
button[data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] {
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* ===== DATAFRAME ===== */
div[data-testid="stDataFrame"] * {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 🎬 BANNER CARROSSEL — FIX DEFINITIVO REAL
# =========================================
import streamlit as st
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

ASSETS = Path("assets")

BANNERS = sorted(str(p) for p in ASSETS.glob("banner*.*"))

if not BANNERS:
    st.warning("⚠️ Coloque imagens em /assets/banner1.png, banner2.png ...")

else:
    total = len(BANNERS)

    # autoplay (a cada 2 min)
    refresh_count = st_autorefresh(interval=120000, key="banner_refresh")

    # inicia estado
    if "banner_idx" not in st.session_state:
        st.session_state.banner_idx = 0

    # autoplay só incrementa (não sobrescreve)
    if refresh_count:
        st.session_state.banner_idx = (st.session_state.banner_idx + 1) % total

    c1, c2, c3 = st.columns([1, 8, 1])

    # ◀
    with c1:
        if st.button("◀", use_container_width=True):
            st.session_state.banner_idx = (st.session_state.banner_idx - 1) % total

    # ▶
    with c3:
        if st.button("▶", use_container_width=True):
            st.session_state.banner_idx = (st.session_state.banner_idx + 1) % total

    with c2:
        st.image(BANNERS[st.session_state.banner_idx], use_container_width=True)


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

    
def calcular_over_under(matriz, max_gols=4):
    linhas = [0.5, 1.5, 2.5, 3.5, 4.5]
    resultados = {}

    # converter para probabilidade
    matriz_prob = matriz / 100

    for linha in linhas:
        over = sum(
            matriz_prob[i][j]
            for i in range(max_gols+1)
            for j in range(max_gols+1)
            if i + j > linha
        )
        resultados[f'Over {linha}'] = over * 100
        resultados[f'Under {linha}'] = (1 - over) * 100

    return resultados

def mostrar_over_under(matriz, titulo):
    ou = calcular_over_under(matriz)

    st.markdown(f"### ⚽ {titulo}")

    df_ou = pd.DataFrame({
        "Linha": ["0.5","1.5","2.5","3.5","4.5"],
        "Over %": [ou['Over 0.5'], ou['Over 1.5'], ou['Over 2.5'], ou['Over 3.5'], ou['Over 4.5']],
        "Under %": [ou['Under 0.5'], ou['Under 1.5'], ou['Under 2.5'], ou['Under 3.5'], ou['Under 4.5']]
    }).round(2)

    st.dataframe(df_ou, use_container_width=True)
    
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
# ⚽ MÉTRICAS OFENSIVAS SKYNET
# =========================================

def eficiencia_finalizacao(chutes_por_gol):
    if chutes_por_gol == 0:
        return 0
    return (1 / chutes_por_gol) * 100


def ajustar_exg_por_eficiencia(exg, ief, media_liga=30):
    fator = ief / media_liga
    return exg * fator


def time_letal(ief, exg):
    return ief > 45 and exg > 1.2


def over_valor_oculto(ief_home, ief_away, exg_total):
    return (ief_home + ief_away) > 80 and exg_total > 2.2


def anti_xg(gols, exg):
    if exg == 0:
        return 0
    return gols / exg


def score_ofensivo(ief, exg, finalizacoes):
    score = (ief*0.4) + (exg*20*0.4) + (finalizacoes*0.2)
    return min(round(score,1), 99)


def rank_time(score):
    if score > 85: return "S"
    if score > 75: return "A"
    if score > 65: return "B"
    if score > 55: return "C"
    return "D"

# =========================================
# RADAR PROFISSIONAL
# =========================================
import numpy as np
import matplotlib.pyplot as plt

def radar_profissional(valores, titulo="Radar Ofensivo", cor="#00E5FF"):
    labels = [
    "Eficiência",
    "ExG",
    "Finalizações",
    "Precisão",
    "Posse",
    "Ataque",
    "Defesa"
]


    valores = np.array(valores)
    angulos = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    valores = np.concatenate((valores, [valores[0]]))
    angulos = np.concatenate((angulos, [angulos[0]]))

    fig = plt.figure(figsize=(4,4), facecolor="#0e1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0e1117")

    ax.set_ylim(0, 100)
    ax.plot(angulos, valores, linewidth=2, color=cor)

    ax.fill(angulos, valores, alpha=0.25, color=cor)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels, fontsize=8, color="white")

    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444")

    ax.set_title(titulo, fontsize=11, color="white")

    return fig
# =========================================
# DOMÍNIO OFENSIVO
# =========================================
def dominio_ofensivo(home_vals, away_vals):
    score_home = sum(home_vals)
    score_away = sum(away_vals)

    if score_home > score_away * 1.15:
        return "HOME"
    elif score_away > score_home * 1.15:
        return "AWAY"
    else:
        return "EQUILIBRADO"


# =========================================
# RADAR ESTILO FIFA
# =========================================
def radar_fifa(valores, titulo="Atributos Ofensivos"):
    return radar_profissional(valores, titulo, "#FFD166")


# =========================================
# SCORE GERAL DO JOGO
# =========================================
def score_jogo(home_vals, away_vals):
    s_home = sum(home_vals)/len(home_vals)
    s_away = sum(away_vals)/len(away_vals)
    total = (s_home + s_away) / 2
    return round(total,1)


# =========================================
# TENDÊNCIA DE GOLS
# =========================================
def tendencia_gols(ief_home, ief_away, exg_total):
    if exg_total > 2.6 and (ief_home + ief_away) > 70:
        return "ALTÍSSIMA"
    elif exg_total > 2.2:
        return "ALTA"
    elif exg_total > 1.8:
        return "MODERADA"
    else:
        return "BAIXA"


# 🎨 BTTS (NOVO)
def calcular_btts_e_odd(matriz):
    # matriz deve estar em PROBABILIDADE (0–1), NÃO %
    btts_prob = sum(
        matriz[i][j]
        for i in range(1, matriz.shape[0])
        for j in range(1, matriz.shape[1])
    )

    btts_pct = btts_prob * 100
    odd_justa = round(1 / btts_prob, 2) if btts_prob > 0 else np.nan

    return btts_pct, odd_justa
    
def radar_comparativo(home_vals, away_vals, home, away):
    labels = [
        "Eficiência","ExG","Finalizações",
        "Precisão","Posse","Ataque","Defesa"
    ]

    angulos = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angulos = np.concatenate((angulos, [angulos[0]]))

    home_vals = np.concatenate((home_vals, [home_vals[0]]))
    away_vals = np.concatenate((away_vals, [away_vals[0]]))

    fig = plt.figure(figsize=(5,5), facecolor="#0e1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0e1117")

    # HOME azul
    ax.plot(angulos, home_vals, linewidth=2, color="#00BFFF")
    ax.fill(angulos, home_vals, alpha=0.25, color="#00BFFF")

    # AWAY laranja
    ax.plot(angulos, away_vals, linewidth=2, color="#FF7A00")
    ax.fill(angulos, away_vals, alpha=0.18, color="#FF7A00")

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="white")

    ax.set_ylim(0, 100)

    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444")

    return fig

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
"📊🧠 Resumo",
"📁🧠 Dados",
"📊⚽ MGF",
"⚔️⚽ ATK x DEF",
"💎⚽ VG"
])


# =========================================
# ABA 1 — RESUMO
# =========================================
with tab1:

    st.markdown("### 🏁 Resultado")

    gh = linha_exg.get("Result Home")
    ga = linha_exg.get("Result Visitor")
    gh_ht = linha_exg.get("Result_Home_HT")
    ga_ht = linha_exg.get("Result_Visitor_HT")

    if pd.notna(gh) and pd.notna(ga):

        gh = int(gh)
        ga = int(ga)
        gh_ht = int(gh_ht) if pd.notna(gh_ht) else 0
        ga_ht = int(ga_ht) if pd.notna(ga_ht) else 0

        home = linha_exg["Home_Team"]
        away = linha_exg["Visitor_Team"]

        if gh > ga:
            home_display = f"🔵 {home}"
            away_display = away
        elif ga > gh:
            home_display = home
            away_display = f"🔵 {away}"
        else:
            home_display = home
            away_display = away

        st.markdown(
            f"""
            <div style="font-size:26px; font-weight:700; margin-bottom:16px;">
                {home_display} {gh} x {ga} {away_display}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="font-size:14px; margin-bottom:6px; opacity:0.6;">
                Resultado HT
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="font-size:24px; font-weight:700;">
                {home} {gh_ht} x {ga_ht} {away}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("⏳ Jogo ainda não finalizado")

    st.markdown("---")


    # 👇 AQUI CONTINUA SUA ABA NORMAL
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
    st.markdown("### 📊📈Métricas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Posse Home (%)", get_val(linha_exg, "Posse_Bola_Home", "{:.2f}"))
        st.metric("PPJH", get_val(linha_exg, "PPJH", "{:.2f}"))
        st.metric("Media_CG_H_01", get_val(linha_mgf, "Media_CG_H_01", "{:.2f}"))
        st.metric("CV_CG_H_01", get_val(linha_mgf, "CV_CG_H_01", "{:.2f}"))

    with c2:
        st.metric("Posse Away (%)", get_val(linha_exg, "Posse_Bola_Away", "{:.2f}"))
        st.metric("PPJA", get_val(linha_exg, "PPJA", "{:.2f}"))
        st.metric("Media_CG_A_01", get_val(linha_mgf, "Media_CG_A_01", "{:.2f}"))
        st.metric("CV_CG_A_01", get_val(linha_mgf, "CV_CG_A_01", "{:.2f}"))

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

    st.markdown("---")

    # -------- LINHA 3 — MGF
    st.markdown("### ⚽🥅 MGF")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Placar Provável", get_val(linha_mgf, "Placar_Mais_Provavel"))
        st.metric("BTTS_YES_VG (%)", linha_mgf["BTTS_%"])
        
    with a2:
        st.metric("ExG_Home_MGF", get_val(linha_mgf, "ExG_Home_MGF", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_mgf, "Clean_Sheet_Home_%", "{:.2f}"))

    with a3:
        st.metric("ExG_Away_MGF", get_val(linha_mgf, "ExG_Away_MGF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_mgf, "Clean_Sheet_Away_%", "{:.2f}"))

    st.markdown("---")
    
    # -------- LINHA 4 — ATK x DEF
    st.markdown("### ⚽⚔️ Ataque x Defesa")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric("Placar Provável", get_val(linha_exg, "Placar_Mais_Provavel"))
        st.metric("BTTS_YES_VG (%)", linha_exg["BTTS_%"])
        
    with e2:
        st.metric("ExG_Home_ATKxDEF", get_val(linha_exg, "ExG_Home_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_exg, "Clean_Sheet_Home_%", "{:.2f}"))
        
    with e3:
        st.metric("ExG_Away_ATKxDEF", get_val(linha_exg, "ExG_Away_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_exg, "Clean_Sheet_Away_%", "{:.2f}"))

    st.markdown("---")
    
    # -------- LINHA 5 — VG
    st.markdown("### ⚽💎 Gols Value")
    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric("Placar Provável", get_val(linha_vg, "Placar_Mais_Provavel"))
        st.metric("BTTS_YES_VG (%)", linha_vg["BTTS_%"])
        
    with b2:
        st.metric("ExG_Home_VG", get_val(linha_vg, "ExG_Home_VG", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_vg, "Clean_Sheet_Home_%", "{:.2f}"))

    with b3:
        st.metric("ExG_Away_VG", get_val(linha_vg, "ExG_Away_VG", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_vg, "Clean_Sheet_Away_%", "{:.2f}"))

    st.markdown("---")
    # =========================================
    # 🎯 RADAR + INTELIGÊNCIA OFENSIVA
    # =========================================

    # ===== MÉTRICAS HOME =====
    ief_home = eficiencia_finalizacao(linha_mgf["CHM"])
    exg_home = linha_mgf["ExG_Home_MGF"]
    shots_home = linha_mgf["CHM"]
    precision_home = linha_exg["Precisao_CG_H"]
    btts_home = linha_mgf["BTTS_%"]
    posse_home = linha_exg["Posse_Bola_Home"]
    atk_home = linha_exg["FAH"]
    def_home = linha_exg["FDH"]

    # ===== MÉTRICAS AWAY =====
    ief_away = eficiencia_finalizacao(linha_mgf["CAM"])
    exg_away = linha_mgf["ExG_Away_MGF"]
    shots_away = linha_mgf["CAM"]
    precision_away = linha_exg["Precisao_CG_A"]
    btts_away = linha_mgf["BTTS_%"]
    posse_away = linha_exg["Posse_Bola_Away"]
    atk_away = linha_exg["FAA"]
    def_away = linha_exg["FDA"]

    def norm_exg(x): return min(x * 40, 100)
    def norm_shots(x): return min((x / 15) * 100, 100)

    radar_home = [
    ief_home,
    norm_exg(exg_home),
    norm_shots(shots_home),
    precision_home,
    posse_home,
    atk_home,
    def_home
]


    radar_away = [
    ief_away,
    norm_exg(exg_away),
    norm_shots(shots_away),
    precision_away,
    posse_away,
    atk_away,
    def_away
]
    home_team = linha_exg["Home_Team"]
    away_team = linha_exg["Visitor_Team"]
    
    st.markdown(
    f"### 🎯 Radar Ofensivo 🔵 <span style='color:#00BFFF'>{home_team}</span> x "
    f"<span style='color:#FF7A00'>{away_team}</span>",
    unsafe_allow_html=True
    )

    st.pyplot(
        radar_comparativo(
            radar_home,
            radar_away,
            home_team,
            away_team
        )
    )

    # ===== ALERTAS =====

    if time_letal(ief_home, exg_home):
        st.success("🔥 Home LETAL hoje")

    if time_letal(ief_away, exg_away):
        st.success("🔥 Away LETAL hoje")

    if over_valor_oculto(ief_home, ief_away, exg_home+exg_away):
        st.warning("💰 Over com valor oculto detectado")

    # ===== DOMÍNIO OFENSIVO =====

    dominio = dominio_ofensivo(radar_home, radar_away)

    if dominio == "HOME":
        st.success("⚔️ Domínio Ofensivo: HOME")
    elif dominio == "AWAY":
        st.success("⚔️ Domínio Ofensivo: AWAY")
    else:
        st.info("⚖️ Ataques equilibrados")

    # ===== SCORE DO JOGO =====

    score = score_jogo(radar_home, radar_away)
    st.metric("🔥 Score Ofensivo do Jogo", score)

    # ===== TENDÊNCIA DE GOLS =====

    tendencia = tendencia_gols(
        ief_home,
        ief_away,
        exg_home + exg_away
    )

    if tendencia == "ALTÍSSIMA":
        st.error("🚨 Tendência ALTÍSSIMA de gols")
    elif tendencia == "ALTA":
        st.warning("🔥 Tendência ALTA de gols")
    else:
        st.info(f"Tendência de gols: {tendencia}")


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

    st.markdown("### 🎯⚽🥅 Odds Justas MGF")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_mgf["Odds_Casa"], linha_mgf["Odd_Justa_Home"])
        ev_btts = calc_ev(linha_mgf["Odd_BTTS_YES"], linha_mgf["Odd_Justa_BTTS"])
        
        st.metric("Odds Casa", linha_mgf["Odds_Casa"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("Placar Provável", get_val(linha_mgf, "Placar_Mais_Provavel"))
        
        st.metric("Odd BTTS Yes", linha_mgf["Odd_BTTS_YES"])
        st.metric("Odd Justa BTTS", linha_mgf["Odd_Justa_BTTS"])
        st.metric("EV BTTS", f"{ev_btts*100:.2f}%")
        
    with o2:
        ev = calc_ev(linha_mgf["Odds_Empate"], linha_mgf["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_mgf["Odds_Empate"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("ExG_Home_MGF", get_val(linha_mgf, "ExG_Home_MGF", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_mgf, "Clean_Sheet_Home_%", "{:.2f}"))
        st.metric("BTTS_YES_VG (%)", linha_mgf["BTTS_%"])
        
    with o3:
        ev = calc_ev(linha_mgf["Odds_Visitante"], linha_mgf["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_mgf["Odds_Visitante"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%") 
        st.metric("ExG_Away_MGF", get_val(linha_mgf, "ExG_Away_MGF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_mgf, "Clean_Sheet_Away_%", "{:.2f}"))

    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_mgf["ExG_Home_MGF"],
        linha_mgf["ExG_Away_MGF"]
    )

    exibir_matriz(
        matriz,
        linha_mgf["Home_Team"],
        linha_mgf["Visitor_Team"],
        "🔢⚽🥅 Poisson — MGF"
    )

    mostrar_over_under(
        matriz,
        "Over/Under — Média de Gols (MGF)"
    )

    st.dataframe(top_placares(matriz), use_container_width=True)

# =========================================
# ABA 4 — POISSON ATK x DEF
# =========================================
with tab4:

    mostrar_card(df_exg, jogo)

    st.markdown("### 🎯⚔️ Odds Justas ATK x DEF")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev = calc_ev(linha_exg["Odds_Casa"], linha_exg["Odd_Justa_Home"])
        ev_btts = calc_ev(linha_exg["Odd_BTTS_YES"], linha_exg["Odd_Justa_BTTS"])
        
        st.metric("Odds Casa", linha_exg["Odds_Casa"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Home"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("Placar Provável", get_val(linha_exg, "Placar_Mais_Provavel"))

        st.metric("Odd BTTS Yes", linha_exg["Odd_BTTS_YES"])
        st.metric("Odd Justa BTTS", linha_exg["Odd_Justa_BTTS"])
        st.metric("EV BTTS", f"{ev_btts*100:.2f}%")
        
    with o2:
        ev = calc_ev(linha_exg["Odds_Empate"], linha_exg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_exg["Odds_Empate"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("ExG_Home_ATKxDEF", get_val(linha_exg, "ExG_Home_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_exg, "Clean_Sheet_Home_%", "{:.2f}"))
        st.metric("BTTS_YES_VG (%)", linha_exg["BTTS_%"])
        
    with o3:
        ev = calc_ev(linha_exg["Odds_Visitante"], linha_exg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%") 
        st.metric("ExG_Away_ATKxDEF", get_val(linha_exg, "ExG_Away_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_exg, "Clean_Sheet_Away_%", "{:.2f}"))
        
    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_exg["ExG_Home_ATKxDEF"],
        linha_exg["ExG_Away_ATKxDEF"]
    )

    exibir_matriz(
        matriz,
        linha_exg["Home_Team"],
        linha_exg["Visitor_Team"],
        "🔢⚔️ Poisson — ATK x DEF"
    )

    mostrar_over_under(
        matriz,
        "Over/Under — Ataque x Defesa"
    )

    st.dataframe(top_placares(matriz), use_container_width=True)

       
# =========================================
# ABA 5 — VG
# =========================================
with tab5:

    mostrar_card(df_vg, jogo)

    st.subheader("🎯💎⚽ Odds Justas VG")

    o1, o2, o3 = st.columns(3)

    with o1:
        ev_home = calc_ev(linha_vg["Odds_Casa"], linha_vg["Odd_Justa_Home"])
        ev_btts = calc_ev(linha_vg["Odd_BTTS_YES"], linha_vg["Odd_Justa_BTTS"])

        st.metric("Odds Casa", linha_vg["Odds_Casa"])
        st.metric("Odd Justa Casa", linha_vg["Odd_Justa_Home"])
        st.metric("EV Casa", f"{ev_home*100:.2f}%")

        st.metric("Placar Provável", get_val(linha_vg, "Placar_Mais_Provavel"))

        st.metric("Odd BTTS Yes", linha_vg["Odd_BTTS_YES"])
        st.metric("Odd Justa BTTS", linha_vg["Odd_Justa_BTTS"])
        st.metric("EV BTTS", f"{ev_btts*100:.2f}%")

    with o2:
        ev = calc_ev(linha_vg["Odds_Empate"], linha_vg["Odd_Justa_Draw"])
        st.metric("Odds Empate", linha_vg["Odds_Empate"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Draw"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("ExG_Home_VG", get_val(linha_vg, "ExG_Home_VG", "{:.2f}"))
        st.metric("Clean Sheet Home (%)", get_val(linha_vg, "Clean_Sheet_Home_%", "{:.2f}"))
        st.metric("BTTS_YES_VG (%)", linha_vg["BTTS_%"])

    with o3:
        ev = calc_ev(linha_vg["Odds_Visitante"], linha_vg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_vg["Odds_Visitante"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("ExG_Away_VG", get_val(linha_vg, "ExG_Away_VG", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_vg, "Clean_Sheet_Away_%", "{:.2f}"))

    st.markdown("---")

    matriz = calcular_matriz_poisson(
        linha_vg["ExG_Home_VG"],
        linha_vg["ExG_Away_VG"]
    )

    exibir_matriz(
        matriz,
        linha_vg["Home_Team"],
        linha_vg["Visitor_Team"],
        "🔢💰⚽Poisson — Valor do Gol (VG)"
    )

    mostrar_over_under(
        matriz,
        "Over/Under — Valor do Gol (VG)"
    )

    st.dataframe(top_placares(matriz), use_container_width=True)
