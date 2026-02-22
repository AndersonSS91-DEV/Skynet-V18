# =========================================
# ⚽🏆 POISSON SKYNET 🏆⚽
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="⚽🏆Poisson Skynet🏆⚽", layout="wide")
st.title("⚽🏆 Poisson Skynet 🏆⚽")

# =========================================
# 🎬 BANNER
# =========================================
ASSETS = Path("assets")
banners = sorted(str(p) for p in ASSETS.glob("banner*.*"))

if banners:
    refresh = st_autorefresh(interval=120000, key="banner")
    if "banner_idx" not in st.session_state:
        st.session_state.banner_idx = 0
    if refresh:
        st.session_state.banner_idx = (st.session_state.banner_idx + 1) % len(banners)

    c1,c2,c3 = st.columns([1,8,1])
    with c1:
        if st.button("◀"):
            st.session_state.banner_idx -= 1
    with c3:
        if st.button("▶"):
            st.session_state.banner_idx += 1
    with c2:
        st.image(banners[st.session_state.banner_idx % len(banners)], use_container_width=True)

# =========================================
# 📂 CARREGAR DADOS
# =========================================
ARQUIVO_PADRAO = "data/POISSON_DUAS_MATRIZES.xlsx"

with st.sidebar:
    arquivo = st.file_uploader("Enviar Excel", type=["xlsx"])

if arquivo:
    xls = pd.ExcelFile(arquivo)
elif os.path.exists(ARQUIVO_PADRAO):
    xls = pd.ExcelFile(ARQUIVO_PADRAO)
else:
    st.error("Arquivo não encontrado")
    st.stop()

# =========================================
# 📊 LEITURA
# =========================================
df_mgf = pd.read_excel(xls,"Poisson_Media_Gols")
df_exg = pd.read_excel(xls,"Poisson_Ataque_Defesa")
df_vg  = pd.read_excel(xls,"Poisson_VG")
df_ht  = pd.read_excel(xls,"Poisson_HT")

for df in (df_mgf,df_exg,df_vg,df_ht):
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]

# =========================================
# 🔧 NORMALIZAÇÕES
# =========================================
def norm_exg(x): return min(x*40,100)
def norm_shots(x): return min((x/15)*100,100)

# =========================================
# 🔥 SCORE OFENSIVO CONSENSO
# =========================================
scores=[]
radar_map={}

for _,row in df_mgf.iterrows():

    exg_row=df_exg[(df_exg.Home_Team==row.Home_Team)&(df_exg.Visitor_Team==row.Visitor_Team)]
    vg_row=df_vg[(df_vg.Home_Team==row.Home_Team)&(df_vg.Visitor_Team==row.Visitor_Team)]

    if exg_row.empty or vg_row.empty:
        scores.append(np.nan)
        continue

    exg_row=exg_row.iloc[0]
    vg_row=vg_row.iloc[0]

    ief_home=(1/row.CHM)*100 if row.CHM>0 else 0
    ief_away=(1/row.CAM)*100 if row.CAM>0 else 0

    radar_home=np.mean([
        [ief_home,norm_exg(row.ExG_Home_MGF),norm_shots(row.CHM),exg_row.Precisao_CG_H,row["BTTS_%"]],
        [exg_row.FAH,norm_exg(exg_row.ExG_Home_ATKxDEF),norm_shots(row.CHM),exg_row.Precisao_CG_H,exg_row["BTTS_%"]],
        [exg_row.FAH,norm_exg(vg_row.ExG_Home_VG),norm_shots(row.CHM),exg_row.Precisao_CG_H,vg_row["BTTS_%"]]
    ],axis=0)

    radar_away=np.mean([
        [ief_away,norm_exg(row.ExG_Away_MGF),norm_shots(row.CAM),exg_row.Precisao_CG_A,row["BTTS_%"]],
        [exg_row.FAA,norm_exg(exg_row.ExG_Away_ATKxDEF),norm_shots(row.CAM),exg_row.Precisao_CG_A,exg_row["BTTS_%"]],
        [exg_row.FAA,norm_exg(vg_row.ExG_Away_VG),norm_shots(row.CAM),exg_row.Precisao_CG_A,vg_row["BTTS_%"]]
    ],axis=0)

    radar_map[row["JOGO"]] = (radar_home, radar_away, ief_home, ief_away)
    scores.append(((sum(radar_home)/5 + sum(radar_away)/5)/2))

df_mgf["Score_Ofensivo"]=scores

# =========================================
# 🎯 ESCOLHER JOGO
# =========================================
jogo = st.selectbox("⚽ Escolha o jogo", df_mgf["JOGO"])

linha_mgf=df_mgf[df_mgf.JOGO==jogo].iloc[0]
linha_exg=df_exg[df_exg.JOGO==jogo].iloc[0]
linha_vg=df_vg[df_vg.JOGO==jogo].iloc[0]

radar_home, radar_away, ief_home, ief_away = radar_map[jogo]

# =========================================
# 📡 RADAR COMPARATIVO
# =========================================
def radar_plot(home,away):
    labels=["Eficiência","ExG","Finalizações","Precisão","BTTS"]
    ang=np.linspace(0,2*np.pi,len(labels),endpoint=False)
    ang=np.concatenate((ang,[ang[0]]))
    home=np.concatenate((home,[home[0]]))
    away=np.concatenate((away,[away[0]]))

    fig=plt.figure(figsize=(4,4))
    ax=fig.add_subplot(111,polar=True)
    ax.plot(ang,home,linewidth=2)
    ax.fill(ang,home,alpha=0.25)
    ax.plot(ang,away,linewidth=2)
    ax.fill(ang,away,alpha=0.15)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0,100)
    return fig

st.pyplot(radar_plot(radar_home,radar_away))

# =========================================
# ⚔️ DOMÍNIO
# =========================================
def dominio(home,away):
    if sum(home)>sum(away)*1.15: return "HOME ⚔️"
    if sum(away)>sum(home)*1.15: return "AWAY ⚔️"
    return "EQUILIBRADO ⚖️"

st.subheader(dominio(radar_home,radar_away))

# =========================================
# 🔥 TENDÊNCIA GOLS
# =========================================
exg_total = linha_exg.ExG_Home_ATKxDEF + linha_exg.ExG_Away_ATKxDEF

def tendencia(exg_total,ief_h,ief_a):
    if exg_total>2.6 and ief_h+ief_a>70: return "🚨🔥 ALTÍSSIMA"
    if exg_total>2.2: return "🔥 ALTA"
    if exg_total>1.8: return "⚡ MODERADA"
    return "❄️ BAIXA"

st.info(tendencia(exg_total,ief_home,ief_away))

# =========================================
# 💀 TIME LETAL
# =========================================
if ief_home>45 and exg_total/2>1.2:
    st.success("💀🔥⚽ HOME LETAL")
if ief_away>45 and exg_total/2>1.2:
    st.success("💀🔥⚽ AWAY LETAL")

# =========================================
# 🎯 MATRIZ POISSON + BTTS
# =========================================
def matriz_poisson(lh,la):
    m=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            m[i,j]=poisson.pmf(i,lh)*poisson.pmf(j,la)
    return m

matriz = matriz_poisson(linha_exg.ExG_Home_ATKxDEF, linha_exg.ExG_Away_ATKxDEF)

btts = sum(matriz[i][j] for i in range(1,5) for j in range(1,5))*100
st.metric("BTTS %", f"{btts:.1f}%")

# =========================================
# 🛡️ SCORE DEFENSIVO
# =========================================
def score_def(fd,cs,chs,mgc):
    if pd.isna(fd): fd=50
    if pd.isna(cs): cs=30
    if pd.isna(chs) or chs==0: chs=10
    if pd.isna(mgc) or mgc==0: mgc=1
    res=min((chs/15)*100,100)
    conc=max(0,100-(mgc*40))
    return round(fd*0.35+cs*0.25+res*0.2+conc*0.2,1)

st.metric("Score Defesa Home", score_def(linha_exg.FDH, linha_exg.CS_H, linha_mgf.CHM, linha_exg.MGC_H))
st.metric("Score Defesa Away", score_def(linha_exg.FDA, linha_exg.CS_A, linha_mgf.CAM, linha_exg.MGC_A))

# =========================================
# 💣 INTENSIDADE DO JOGO
# =========================================
score_jogo = df_mgf[df_mgf.JOGO==jogo]["Score_Ofensivo"].values[0]

def intensidade(s):
    if s<35: return "❄️🧊 FRIO"
    if s<60: return "⚡ EQUILIBRADO"
    if s<80: return "🔥 PRESSÃO"
    if s<85: return "💣 QUENTE"
    return "💀 PIROTÉCNICO"

st.header(intensidade(score_jogo))

# =========================================
# FUNÇÕES AUX
# =========================================
def get_val(linha, col, fmt=None, default="—"):
    """
    Retorna valor seguro da coluna.
    Evita crash se coluna não existir ou for NaN.
    """
    if col in linha.index and pd.notna(linha[col]):
        try:
            return fmt.format(linha[col]) if fmt else linha[col]
        except:
            return default
    return default


def calc_ev(odd_real, odd_justa):
    """
    Calcula Valor Esperado (EV)
    > 0  = valor positivo
    < 0  = valor negativo
    """
    try:
        return (odd_real / odd_justa) - 1
    except:
        return None
        
# =========================================
# ESTATÍSTICAS DO SCORE (CONSENSO)
# =========================================
media_score = df_mgf["Score_Ofensivo"].mean()
desvio_score = df_mgf["Score_Ofensivo"].std()

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

    st.markdown("### 🎯 Odds")

    o1, o2, o3 = st.columns(3)

with o1:
    st.metric("Odds Casa", linha_exg["Odds_Casa"])
    st.metric("Odd Over 1.5", linha_exg["Odd_Over_1,5FT"])
    st.metric("VR01", get_val(linha_exg, "VR01", "{:.2f}"))

with o2:
    st.metric("Odds Empate", linha_exg["Odds_Empate"])
    st.metric("Odd Over 2.5", linha_exg["Odds_Over_2,5FT"])
    st.metric("COEF_OVER1FT", get_val(linha_exg, "COEF_OVER1FT", "{:.2f}"))

with o3:
    st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
    st.metric("Odd Under 2.5", linha_exg["Odds_Under_2,5FT"])
    st.metric("BTTS Yes", linha_exg["Odd_BTTS_YES"])

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
        st.metric("MGF_HT_Home", get_val(linha_ht, "MGF_HT_Home", "{:.2f}"))
        st.metric("CV_MGF_HT_Home", get_val(linha_ht, "CV_MGF_HT_Home", "{:.2f}"))

    with c4:
        st.metric("Força Ataque Away (%)", get_val(linha_exg, "FAA", "{:.2f}"))
        st.metric("Precisão Chutes A (%)", get_val(linha_exg, "Precisao_CG_A", "{:.2f}"))
        st.metric("Chutes A (Marcar)", get_val(linha_mgf, "CAM", "{:.2f}"))
        st.metric("MGF_A", get_val(linha_mgf, "MGF_A", "{:.2f}"))
        st.metric("CV_GF_A", get_val(linha_mgf, "CV_GF_A", "{:.2f}"))
        st.metric("MGF_HT_Away", get_val(linha_ht, "MGF_HT_Away", "{:.2f}"))
        st.metric("CV_MGF_HT_Away", get_val(linha_ht, "CV_MGF_HT_Away", "{:.2f}"))
                  
    with c5:
        st.metric("Força Defesa Home (%)", get_val(linha_exg, "FDH", "{:.2f}"))
        st.metric("Clean Games Home (%)", get_val(linha_exg, "Clean_Games_H"))
        st.metric("Chutes H (Sofrer)", get_val(linha_mgf, "CHS", "{:.2f}"))
        st.metric("MGC_H", get_val(linha_mgf, "MGC_H", "{:.2f}"))
        st.metric("CV_GC_H", get_val(linha_mgf, "CV_GC_H", "{:.2f}"))
        st.metric("MGC_HT_Home", get_val(linha_ht, "MGC_HT_Home", "{:.2f}"))
        st.metric("CV_MGC_HT_Home", get_val(linha_ht, "CV_MGC_HT_Home", "{:.2f}"))
        
    with c6:
        st.metric("Força Defesa Away (%)", get_val(linha_exg, "FDA", "{:.2f}"))
        st.metric("Clean Games Away (%)", get_val(linha_exg, "Clean_Games_A"))
        st.metric("Chutes A (Sofrer)", get_val(linha_mgf, "CAS", "{:.2f}"))
        st.metric("MGC_A", get_val(linha_mgf, "MGC_A", "{:.2f}"))
        st.metric("CV_GC_A", get_val(linha_mgf, "CV_GC_A", "{:.2f}"))
        st.metric("MGC_HT_Away", get_val(linha_ht, "MGC_HT_Away", "{:.2f}"))
        st.metric("CV_MGC_HT_Away", get_val(linha_ht, "CV_MGC_HT_Away", "{:.2f}"))
        
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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_mgf, "Home_Abrir_Placar"))
        
    with a3:
        st.metric("ExG_Away_MGF", get_val(linha_mgf, "ExG_Away_MGF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_mgf, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_mgf, "Away_Abrir_Placar"))
        
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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_exg, "Home_Abrir_Placar"))
        
    with e3:
        st.metric("ExG_Away_ATKxDEF", get_val(linha_exg, "ExG_Away_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_exg, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_exg, "Away_Abrir_Placar"))

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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_vg, "Home_Abrir_Placar"))
        
    with b3:
        st.metric("ExG_Away_VG", get_val(linha_vg, "ExG_Away_VG", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_vg, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_vg, "Away_Abrir_Placar"))
                  
    st.markdown("---")

    # =========================================
    # 🔢 POISSON CONSENSO
    # =========================================

    lambda_home = np.mean([
        linha_mgf["ExG_Home_MGF"],
        linha_exg["ExG_Home_ATKxDEF"],
        linha_vg["ExG_Home_VG"]
    ])

    lambda_away = np.mean([
        linha_mgf["ExG_Away_MGF"],
        linha_exg["ExG_Away_ATKxDEF"],
        linha_vg["ExG_Away_VG"]
    ])

    st.markdown("### 🔢⚽ Poisson Consenso")

    matriz_consenso = calcular_matriz_poisson(lambda_home, lambda_away)

    exibir_matriz(
        matriz_consenso,
        linha_exg["Home_Team"],
        linha_exg["Visitor_Team"],
        "Probabilidades de Placar (Consenso)"
    )

    mostrar_over_under(
        matriz_consenso,
        "Over/Under — Consenso"
    )

    # =========================================
    # 🎯 RADAR CONSENSO
    # =========================================

    radar_home_mgf = [
        eficiencia_finalizacao(linha_mgf["CHM"]),
        norm_exg(linha_mgf["ExG_Home_MGF"]),
        norm_shots(linha_mgf["CHM"]),
        linha_exg["Precisao_CG_H"],
        linha_mgf["BTTS_%"]
    ]

    radar_away_mgf = [
        eficiencia_finalizacao(linha_mgf["CAM"]),
        norm_exg(linha_mgf["ExG_Away_MGF"]),
        norm_shots(linha_mgf["CAM"]),
        linha_exg["Precisao_CG_A"],
        linha_mgf["BTTS_%"]
    ]

    radar_home_exg = [
        linha_exg["FAH"],
        norm_exg(linha_exg["ExG_Home_ATKxDEF"]),
        norm_shots(linha_mgf["CHM"]),
        linha_exg["Precisao_CG_H"],
        linha_exg["BTTS_%"]
    ]

    radar_away_exg = [
        linha_exg["FAA"],
        norm_exg(linha_exg["ExG_Away_ATKxDEF"]),
        norm_shots(linha_mgf["CAM"]),
        linha_exg["Precisao_CG_A"],
        linha_exg["BTTS_%"]
    ]

    radar_home_vg = [
        linha_exg["FAH"],
        norm_exg(linha_vg["ExG_Home_VG"]),
        norm_shots(linha_mgf["CHM"]),
        linha_exg["Precisao_CG_H"],
        linha_vg["BTTS_%"]
    ]

    radar_away_vg = [
        linha_exg["FAA"],
        norm_exg(linha_vg["ExG_Away_VG"]),
        norm_shots(linha_mgf["CAM"]),
        linha_exg["Precisao_CG_A"],
        linha_vg["BTTS_%"]
    ]

    radar_home_consenso = np.mean(
        [radar_home_mgf, radar_home_exg, radar_home_vg], axis=0
    )

    radar_away_consenso = np.mean(
        [radar_away_mgf, radar_away_exg, radar_away_vg], axis=0
    )

    st.markdown("### 🎯 Radar Ofensivo Consenso")

    st.markdown(
        f"### <span style='color:#00BFFF'>{linha_exg['Home_Team']}</span> x "
        f"<span style='color:#FF7A00'>{linha_exg['Visitor_Team']}</span>",
        unsafe_allow_html=True
    )

    fig = radar_comparativo(
        radar_home_consenso,
        radar_away_consenso,
        linha_exg["Home_Team"],
        linha_exg["Visitor_Team"]
    )

    st.pyplot(fig, use_container_width=False)


    cards_ofensivos(
        radar_home_consenso,
        radar_away_consenso,
        radar_home_consenso[0],   # eficiência home
        radar_away_consenso[0],   # eficiência away
        lambda_home + lambda_away
    )

# =============================
# ⚡ CARD HT
# =============================
jogo_ht = df_ht[df_ht["JOGO"] == jogo]

if not jogo_ht.empty:

    ht = jogo_ht.iloc[0]

    selo_ht = ht.get("Selo_HT", "")
    if pd.isna(selo_ht):
        selo_ht = ""

    st.info(
        f"""⚡ Probabilidade de Gol no 1º Tempo

🔥 Gol HT: {ht['Prob_Gol_HT']}%   |   ❄️ 0x0 HT: {ht['Prob_0x0_HT']}%

🏠 Home marca HT: {ht['Gol_HT_Home_%']}%   |   ✈️ Away marca HT: {ht['Gol_HT_Away_%']}%

{selo_ht}
"""
    )
    
    # =========================================
    # 🔥 SCORE OFENSIVO NORMALIZADO (0–100 REAL)
    # =========================================
    score_bruto = ((sum(radar_home_consenso)/5 + sum(radar_away_consenso)/5)/2)
    z = (score_bruto - media_score) / desvio_score
    score_ofensivo = 50 + (z * 18)
    score_ofensivo = max(min(score_ofensivo, 100), 0)

    st.metric("🔥 Score Ofensivo", f"{score_ofensivo:.0f}")
    st.info(classificar_intensidade(score_ofensivo))
    
    # =========================================
    # 🛡️🏔️🧤🥅 DEFESA CONSENSO
    # =========================================
    st.markdown("### 🛡️🏔️🧤🥅 Defesa Consenso")

    base_home = score_defensivo(
        linha_exg["FDH"],
        linha_exg["Clean_Games_H"],
        linha_mgf["CHS"],
        linha_mgf["MGC_H"]
    )

    base_away = score_defensivo(
        linha_exg["FDA"],
        linha_exg["Clean_Games_A"],
        linha_mgf["CAS"],
        linha_mgf["MGC_A"]
    )

    estrutura_home = max(min((linha_exg["FDH"] - linha_exg["FAA"] + 100)/2,100),0)
    estrutura_away = max(min((linha_exg["FDA"] - linha_exg["FAH"] + 100)/2,100),0)

    clean_home = linha_mgf["Clean_Sheet_Home_%"]
    clean_away = linha_mgf["Clean_Sheet_Away_%"]

    def_home = round(base_home*0.5 + estrutura_home*0.3 + clean_home*0.2,1)
    def_away = round(base_away*0.5 + estrutura_away*0.3 + clean_away*0.2,1)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(linha_exg["Home_Team"], def_home)
        st.info(classificar_defesa(def_home))

    with c2:
        st.metric(linha_exg["Visitor_Team"], def_away)
        st.info(classificar_defesa(def_away))

    
    # =========================================
    # 🧠 LEITURA CONSENSO
    # =========================================

    st.markdown("### 🧠 Leitura Ofensiva Consenso")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            leitura_consenso(
                linha_exg["Home_Team"],
                radar_home_consenso
            )
        )

    with col2:
        st.info(
            leitura_consenso(
                linha_exg["Visitor_Team"],
                radar_away_consenso
            )
        )




# =========================================
# ABA 2 — DADOS COMPLETOS
# =========================================
with tab2:
    for aba in xls.sheet_names:
        with st.expander(aba):
            df_temp = pd.read_excel(xls, aba)
            st.dataframe(df_temp, use_container_width=True)


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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_mgf, "Home_Abrir_Placar"))
        st.metric("BTTS_YES_MGF (%)", linha_mgf["BTTS_%"])
        
    with o3:
        ev = calc_ev(linha_mgf["Odds_Visitante"], linha_mgf["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_mgf["Odds_Visitante"])
        st.metric("Odd Justa", linha_mgf["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%") 
        st.metric("ExG_Away_MGF", get_val(linha_mgf, "ExG_Away_MGF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_mgf, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_mgf, "Away_Abrir_Placar"))

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

       # ===== RADAR MGF =====
    ief_home = eficiencia_finalizacao(linha_mgf["CHM"])
    ief_away = eficiencia_finalizacao(linha_mgf["CAM"])

    radar_home_mgf = [
        ief_home,
        min(linha_mgf["ExG_Home_MGF"] * 40, 100),
        min((linha_mgf["CHM"]/15)*100, 100),
        linha_exg["Precisao_CG_H"],
        linha_mgf["BTTS_%"]
    ]

    radar_away_mgf = [
        ief_away,
        min(linha_mgf["ExG_Away_MGF"] * 40, 100),
        min((linha_mgf["CAM"]/15)*100, 100),
        linha_exg["Precisao_CG_A"],
        linha_mgf["BTTS_%"]
    ]

    st.markdown("### 🎯 Radar Ofensivo — MGF")

    fig = radar_comparativo(
        radar_home_mgf,
        radar_away_mgf,
        linha_mgf["Home_Team"],
        linha_mgf["Visitor_Team"]
    )

    st.pyplot(fig, use_container_width=False)

    cards_ofensivos(
    radar_home_mgf,
    radar_away_mgf,
    ief_home,
    ief_away,
    linha_mgf["ExG_Home_MGF"] + linha_mgf["ExG_Away_MGF"]
)

    st.markdown("### 🧱 Defesa — Histórico (MGF)")

    def_home = score_defensivo(
        linha_exg["FDH"],
        linha_mgf["Clean_Sheet_Home_%"],
        linha_mgf["CHS"],
        linha_mgf["MGC_H"]
    )

    def_away = score_defensivo(
        linha_exg["FDA"],
        linha_mgf["Clean_Sheet_Away_%"],
        linha_mgf["CAS"],
        linha_mgf["MGC_A"]
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(linha_mgf["Home_Team"], def_home)
        st.info(classificar_defesa(def_home))

    with c2:
        st.metric(linha_mgf["Visitor_Team"], def_away)
        st.info(classificar_defesa(def_away))

    st.markdown("### 🧠 Leitura Ofensiva (Histórico)")

    col1, col2 = st.columns(2)

    with col1:
        st.info(leitura_ofensiva(
            linha_mgf["Home_Team"],
            *radar_home_mgf
        ))

    with col2:
        st.info(leitura_ofensiva(
            linha_mgf["Visitor_Team"],
            *radar_away_mgf
        ))

# =========================================
    # ===== RADAR ATK x DEF =====
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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_exg, "Home_Abrir_Placar"))
        st.metric("BTTS_YES_ATKxDEF (%)", linha_exg["BTTS_%"])
        
    with o3:
        ev = calc_ev(linha_exg["Odds_Visitante"], linha_exg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odd Justa", linha_exg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%") 
        st.metric("ExG_Away_ATKxDEF", get_val(linha_exg, "ExG_Away_ATKxDEF", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_exg, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_exg, "Away_Abrir_Placar"))
        
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

       # ===== RADAR ATK x DEF =====
    radar_home_exg = [
        linha_exg["FAH"],
        min(linha_exg["ExG_Home_ATKxDEF"] * 40, 100),
        min((linha_mgf["CHM"]/15)*100, 100),
        linha_exg["Precisao_CG_H"],
        linha_exg["BTTS_%"]
    ]

    radar_away_exg = [
        linha_exg["FAA"],
        min(linha_exg["ExG_Away_ATKxDEF"] * 40, 100),
        min((linha_mgf["CAM"]/15)*100, 100),
        linha_exg["Precisao_CG_A"],
        linha_exg["BTTS_%"]
    ]

    st.markdown("### ⚔️ Radar Tático")

    fig = radar_comparativo(
        radar_home_exg,
        radar_away_exg,
        linha_exg["Home_Team"],
        linha_exg["Visitor_Team"]
    )

    st.pyplot(fig, use_container_width=False)

    cards_ofensivos(
        radar_home_exg,
        radar_away_exg,
        radar_home_exg[0],
        radar_away_exg[0],
        linha_exg["ExG_Home_ATKxDEF"] + linha_exg["ExG_Away_ATKxDEF"]
    )
    
    st.markdown("### 🧱 Defesa Estrutural")

    estrutura_home = max(min((linha_exg["FDH"] - linha_exg["FAA"] + 100)/2,100),0)
    estrutura_away = max(min((linha_exg["FDA"] - linha_exg["FAH"] + 100)/2,100),0)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(linha_exg["Home_Team"], round(estrutura_home,1))
        st.info(classificar_defesa(estrutura_home))

    with c2:
        st.metric(linha_exg["Visitor_Team"], round(estrutura_away,1))
        st.info(classificar_defesa(estrutura_away))


    st.markdown("### 🧠 Leitura Tática")

    col1, col2 = st.columns(2)

    with col1:
        st.info(leitura_ofensiva(
            linha_exg["Home_Team"],
            *radar_home_exg
        ))

    with col2:
        st.info(leitura_ofensiva(
            linha_exg["Visitor_Team"],
            *radar_away_exg
        ))
       
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
        st.metric("Home Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_vg, "Home_Abrir_Placar"))
        st.metric("BTTS_YES_VG (%)", linha_vg["BTTS_%"])

    with o3:
        ev = calc_ev(linha_vg["Odds_Visitante"], linha_vg["Odd_Justa_Away"])
        st.metric("Odds Visitante", linha_vg["Odds_Visitante"])
        st.metric("Odd Justa", linha_vg["Odd_Justa_Away"])
        st.metric("EV", f"{ev*100:.2f}%")
        st.metric("ExG_Away_VG", get_val(linha_vg, "ExG_Away_VG", "{:.2f}"))
        st.metric("Clean Sheet Away (%)", get_val(linha_vg, "Clean_Sheet_Away_%", "{:.2f}"))
        st.metric("Away Marcar 1º Gol1️⃣⚽ (%)", get_val(linha_vg, "Away_Abrir_Placar"))

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

    # ===== RADAR VG =====
    radar_home_vg = [
        linha_exg["FAH"],
        min(linha_vg["ExG_Home_VG"] * 40, 100),
        min((linha_mgf["CHM"]/15)*100, 100),
        linha_exg["Precisao_CG_H"],
        linha_vg["BTTS_%"]
    ]

    radar_away_vg = [
        linha_exg["FAA"],
        min(linha_vg["ExG_Away_VG"] * 40, 100),
        min((linha_mgf["CAM"]/15)*100, 100),
        linha_exg["Precisao_CG_A"],
        linha_vg["BTTS_%"]
    ]

    st.markdown("### 💎 Radar Ofensivo — Valor")

    fig = radar_comparativo(
        radar_home_vg,
        radar_away_vg,
        linha_vg["Home_Team"],
        linha_vg["Visitor_Team"]
    )

    st.pyplot(fig, use_container_width=False)

    cards_ofensivos(
        radar_home_vg,
        radar_away_vg,
        radar_home_vg[0],
        radar_away_vg[0],
        linha_vg["ExG_Home_VG"] + linha_vg["ExG_Away_VG"]
    )       
    st.markdown("---")
