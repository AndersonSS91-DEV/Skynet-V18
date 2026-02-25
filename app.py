# =========================================
# STREAMLIT — POISSON SKYNET (HÍBRIDO)
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import glob
from pathlib import Path
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import random
from streamlit_autorefresh import st_autorefresh
from PIL import Image
import base64
from pathlib import Path
import unicodedata
# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="⚽🏆Poisson Skynet🏆⚽",
    layout="wide"
)
st.markdown(
    """
    <h1 style="text-align:center;">⚽🏆 Poisson Skynet 🏆⚽</h1>
    <hr style="width:560px; margin:auto; border:4px solid #FFD700;">
    """,
    unsafe_allow_html=True
)
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

/* ===== MÉTRICAS ===== */
div[data-testid="metric-container"] label {
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    opacity: 0.6 !important;
    font-weight: 700 !important;
}

div[data-testid="metric-container"] > div {
    font-size: 48px !important;
    font-weight: 900 !important;
}

/* ===== ALERT CARDS ===== */
div[data-testid="stAlert"] {
    font-size: 20px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* mantém cores originais do Streamlit */
div[data-testid="stAlert"] p {
    font-size: 20px !important;
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
div[data-testid="stDataFrame"] table {
    font-size: 17px !important;
}

div[data-testid="stDataFrame"] th {
    font-size: 17px !important;
    font-weight: 700 !important;
}

div[data-testid="stDataFrame"] td {
    font-size: 17px !important;
}

/* texto dos cards st.info */
div[data-testid="stAlert"] {
    color: white !important;
}

div[data-testid="stAlert"] p {
    color: white !important;
}


/* ===== TABELAS DATAFRAME ===== */

div[data-testid="stDataFrame"] table {
    font-size: 22px !important;
}
div[data-testid="stDataFrame"] th {
    font-size: 20px !important;
}

/* ===== TABELAS DATAFRAME ===== */

div[data-testid="stDataFrame"] table {
    font-size: 22px !important;
}
div[data-testid="stDataFrame"] th {
    font-size: 20px !important;
}

/* texto do valor selecionado */
div[data-baseweb="select"] div {
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* opções dentro da lista */
ul[role="listbox"] li {
    font-size: 16px !important;
}

button[aria-selected="true"] {
    font-size: 22px !important;
    color: #00E5FF !important;
}

button[data-baseweb="tab"][aria-selected="true"] * {
    font-size: 26px !important;
    color: #00E5FF !important;
}
/* ===== SELECTBOX TAMANHO ===== */

/* caixa externa */
div[data-baseweb="select"] {
    min-height: 55px !important;
}

/* área clicável */
div[data-baseweb="select"] > div {
    min-height: 55px !important;
    display: flex;
    align-items: center;
}

/* texto selecionado */
div[data-baseweb="select"] span {
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* itens do dropdown */
ul[role="listbox"] li {
    font-size: 18px !important;
    padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# 🎬 BANNER CARROSSEL (OFICIAL SKYNET)
# =========================================
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

ASSETS = Path("assets")
BANNERS = sorted(ASSETS.glob("banner*"))

if BANNERS:

    refresh = st_autorefresh(interval=120000, key="banner")

    if "banner_idx" not in st.session_state:
        st.session_state.banner_idx = 0

    if refresh:
        st.session_state.banner_idx = (st.session_state.banner_idx + 1) % len(BANNERS)

    colL, colC, colR = st.columns([1,6,1])

    with colL:
        if st.button("◀", use_container_width=True):
            st.session_state.banner_idx -= 1

    with colR:
        if st.button("▶", use_container_width=True):
            st.session_state.banner_idx += 1

    st.session_state.banner_idx %= len(BANNERS)

    with colC:
        st.image(str(BANNERS[st.session_state.banner_idx]), use_container_width=True)
        
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
df_ht = pd.read_excel(xls,  "Poisson_HT")

for df in (df_mgf, df_exg, df_vg, df_ht):
    df["JOGO"] = df["Home_Team"] + " x " + df["Visitor_Team"]
    df_ht["JOGO"] = df_ht["Home_Team"] + " x " + df_ht["Visitor_Team"]

# =========================================
# 🔥 SCORE OFENSIVO CONSENSO 0–100
# =========================================

score_raw = []

for _, row in df_mgf.iterrows():

    jogo = row["Home_Team"] + " x " + row["Visitor_Team"]

    exg_row = df_exg[df_exg["Home_Team"].eq(row["Home_Team"]) &
                     df_exg["Visitor_Team"].eq(row["Visitor_Team"])]

    vg_row  = df_vg[df_vg["Home_Team"].eq(row["Home_Team"]) &
                    df_vg["Visitor_Team"].eq(row["Visitor_Team"])]

    if exg_row.empty or vg_row.empty:
        score_raw.append(np.nan)
        continue

    exg_row = exg_row.iloc[0]
    vg_row  = vg_row.iloc[0]

    # eficiência
    ief_home = (1 / row["CHM"]) * 100 if row["CHM"] > 0 else 0
    ief_away = (1 / row["CAM"]) * 100 if row["CAM"] > 0 else 0

    # normalizações radar
    def norm_exg(x): return min(x * 40, 100)
    def norm_shots(x): return min((x / 15) * 100, 100)

    radar_home = np.mean([
        [ief_home, norm_exg(row["ExG_Home_MGF"]), norm_shots(row["CHM"]), exg_row["Precisao_CG_H"], row["BTTS_%"]],
        [exg_row["FAH"], norm_exg(exg_row["ExG_Home_ATKxDEF"]), norm_shots(row["CHM"]), exg_row["Precisao_CG_H"], exg_row["BTTS_%"]],
        [exg_row["FAH"], norm_exg(vg_row["ExG_Home_VG"]), norm_shots(row["CHM"]), exg_row["Precisao_CG_H"], vg_row["BTTS_%"]]
    ], axis=0)

    radar_away = np.mean([
        [ief_away, norm_exg(row["ExG_Away_MGF"]), norm_shots(row["CAM"]), exg_row["Precisao_CG_A"], row["BTTS_%"]],
        [exg_row["FAA"], norm_exg(exg_row["ExG_Away_ATKxDEF"]), norm_shots(row["CAM"]), exg_row["Precisao_CG_A"], exg_row["BTTS_%"]],
        [exg_row["FAA"], norm_exg(vg_row["ExG_Away_VG"]), norm_shots(row["CAM"]), exg_row["Precisao_CG_A"], vg_row["BTTS_%"]]
    ], axis=0)

    score = ((sum(radar_home)/5 + sum(radar_away)/5) / 2)
    score_raw.append(score)

df_mgf["Score_Ofensivo"] = score_raw

# 🔥 recalibra para 0–100
# df_mgf["Score_Ofensivo_100"] = recalibrar_0_100(df_mgf["Score_Ofensivo"])

def recalibrar_0_100(serie):
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return serie * 0

    return ((serie - minimo) / (maximo - minimo)) * 100

# =========================================    
# 🔥 DEFINA AQUI (ANTES DAS TABS)
jogos_lista = df_mgf["JOGO"].tolist()

if "jogo" not in st.session_state or st.session_state["jogo"] not in jogos_lista:
    st.session_state["jogo"] = jogos_lista[0]

st.markdown("### ⚽ Escolha o jogo")
jogo = st.selectbox(label="",options=jogos_lista)
linha_mgf = df_mgf[df_mgf["JOGO"] == jogo].iloc[0]
linha_exg = df_exg[df_exg["JOGO"] == jogo].iloc[0]
linha_vg  = df_vg[df_vg["JOGO"] == jogo].iloc[0]
linha_ht  = df_ht[df_ht["JOGO"] == jogo].iloc[0]  # ✅ ADICIONE ESTA

st.session_state["jogo"] = jogo
# =========================================
# FUNÇÕES AUX
# =========================================
# =========================================
# 🔰 MATCH PROFISSIONAL DE ESCUDOS (FIX FINAL)
# =========================================

def escudo_path(nome_time):
    import os, re, unicodedata

    pasta = "escudos"
    placeholder = os.path.join(pasta, "time_vazio.png")

    if not os.path.exists(pasta):
        return placeholder

    if not nome_time:
        return placeholder

    # 🔥 apelidos manuais
    APELIDOS = {
    "inter milan": "inter",
    "inter": "inter",
        
    # BODO GLIMT → arquivo bodo.png
    "bodø glimt": "bodo",
    "bodo glimt": "bodo",
    "bodo / glimt": "bodo",
    "fk bodo glimt": "bodo",
        
    "olympiacos": "olympiakos",
    "olympiacos f.c.": "olympiakos",
    "estrela": "estrela amadora",
}
    def limpar(txt):
        txt = str(txt).lower().strip()

        # remove acentos
        txt = unicodedata.normalize('NFKD', txt)\
              .encode('ASCII','ignore').decode('ASCII')

        # transforma separadores em espaço
        txt = re.sub(r'[\/\-_]+', ' ', txt)

        # remove termos inúteis
        txt = re.sub(r'\b(fc|f\.c\.|club|sc)\b', '', txt)

        # remove categorias base
        txt = re.sub(r'\b(u17|u19|u20|u21|u23)\b', '', txt)

        # remove espaços duplicados
        txt = re.sub(r'\s+', ' ', txt).strip()

        return txt

    alvo = limpar(nome_time)

    if alvo in APELIDOS:
        alvo = APELIDOS[alvo]

    arquivos = [a for a in os.listdir(pasta) if a.endswith(".png")]

    for arq in arquivos:
        nome_arq = limpar(arq.replace(".png",""))

        if nome_arq == alvo:
            return os.path.join(pasta, arq)

        if nome_arq.replace(" ","") == alvo.replace(" ",""):
            return os.path.join(pasta, arq)

    # 1️⃣ match exato
    for arq in arquivos:
        if limpar(arq.replace(".png","")) == alvo:
            return os.path.join(pasta, arq)

    # 2️⃣ match compacto
    alvo2 = alvo.replace(" ", "")
    for arq in arquivos:
        if limpar(arq.replace(".png","")).replace(" ","") == alvo2:
            return os.path.join(pasta, arq)

    # 3️⃣ tokens
    alvo_tokens = set(alvo.split())

    for arq in arquivos:
        nome_arq = limpar(arq.replace(".png",""))
        if alvo_tokens.issubset(set(nome_arq.split())):
            return os.path.join(pasta, arq)

    return placeholder
    
    # (FIM DO BLOCO)
        
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

    fig = plt.figure(figsize=(2.4, 2.0), dpi=120)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')

    tabela = ax.table(
        cellText=df_ou.values,
        colLabels=df_ou.columns,
        cellLoc='center',
        loc='center'
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(7)      # 👈 tamanho igual ao Poisson
    tabela.scale(1.1, 1.2)      # 👈 altura e largura das células

    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight='bold')

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    
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
    "BTTS"
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

# =========================================
# LEITURA OFENSIVA
# =========================================
def leitura_ofensiva(nome, eficiencia, exg, finalizacoes, precisao, btts):
    
    texto = f"{nome}\n\n"

    if eficiencia > 50:
        texto += "✔ Eficiência alta\n"
    elif eficiencia > 35:
        texto += "✔ Eficiência média\n"
    else:
        texto += "✔ Eficiência baixa\n"

    if exg > 70:
        texto += "✔ ExG muito alto\n"
    elif exg > 45:
        texto += "✔ ExG moderado\n"
    else:
        texto += "✔ ExG baixo\n"

    if finalizacoes < 30:
        texto += "✔ Poucas finalizações\n"
    elif finalizacoes > 70:
        texto += "✔ Muitas finalizações\n"
    else:
        texto += "✔ Volume equilibrado\n"

    if precisao > 55:
        texto += "✔ Alta precisão\n"
    else:
        texto += "✔ Precisão média\n"

    if btts < 45:
        texto += "✔ BTTS baixo\n"
    else:
        texto += "✔ BTTS moderado/alto\n"

    texto += "\n🧠 leitura:\n"

    if eficiencia > 50 and exg > 60:
        texto += "👉 cria chances de alta qualidade\n"
        texto += "🎯🔫 precisa de poucas oportunidades\n"
        texto += "🎯🔥 perfil letal\n"
    elif finalizacoes > 70 and eficiencia < 40:
        texto += "👉 volume alto, qualidade baixa\n"
        texto += "👉 chuta muito e marca pouco\n"
    else:
        texto += "👉 perfil ofensivo equilibrado\n"

    return texto

# =========================================
# 🔥 RECALIBRA SCORE OFENSIVO (0–100)
# =========================================
def recalibrar_0_100(serie):
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return serie * 0

    return ((serie - minimo) / (maximo - minimo)) * 100


# =========================================
# 🎯 CLASSIFICA INTENSIDADE OFENSIVA
# =========================================
def classificar_intensidade(score):

    if score < 35:
        return "❄️🧊 Jogo frio"

    elif score < 60:
        return "⚡⚽ Equilibrado"

    elif score < 80:
        return "⚡💥🔥⚽ Pressão ofensiva"

    elif score < 85:
        return "🔥💣💥⚽ Jogo Quente"

    else:
        return "💀💣🔥💥⚽ Jogo Pirotécnico"

# =========================================
# LEITURA OFENSIVA
# =========================================
def leitura_consenso(nome, radar_vals):

    eficiencia, exg, finalizacoes, precisao, btts = radar_vals

    linhas = []

    # Eficiência
    if eficiencia > 50:
        linhas.append("✓ Eficiência ofensiva alta")
    elif eficiencia > 35:
        linhas.append("✓ Eficiência ofensiva média")
    else:
        linhas.append("✓ Eficiência ofensiva baixa")

    # Criação
    if exg > 70:
        linhas.append("✓ Criação de chances muito alta")
    elif exg > 45:
        linhas.append("✓ Criação ofensiva moderada")
    else:
        linhas.append("✓ Baixa criação ofensiva")

    # Volume
    if finalizacoes > 70:
        linhas.append("✓ Volume ofensivo intenso")
    elif finalizacoes < 30:
        linhas.append("✓ Poucas finalizações")
    else:
        linhas.append("✓ Volume equilibrado")

    # Precisão
    if precisao > 55:
        linhas.append("✓ Alta precisão nas finalizações")
    else:
        linhas.append("✓ Precisão mediana")

    # Perfil do jogo
    if btts > 60:
        linhas.append("✓ Jogos abertos com frequência")
    else:
        linhas.append("✓ Tendência a jogos controlados")

    # 🧠 leitura final
    if eficiencia > 50 and exg > 60:
        leitura = "👉 time cria chances claras\n👉 perfil ofensivo letal"

    elif finalizacoes > 70 and eficiencia < 40:
        leitura = "👉 volume alto com baixa qualidade"

    elif exg < 40:
        leitura = "👉 dificuldade para criar oportunidades"

    else:
        leitura = "👉 perfil ofensivo equilibrado"

    texto = "\n".join(linhas)

    return f"""
**{nome}**

{texto}

🧠 leitura:
{leitura}
"""

# =========================================
# RADAR COMPARATIVO
# =========================================
def radar_comparativo(home_vals, away_vals, home, away):

    labels = [
        "Eficiência",
        "ExG",
        "Finalizações",
        "Precisão",
        "BTTS"
    ]

    home_vals = np.array(home_vals, dtype=float)
    away_vals = np.array(away_vals, dtype=float)

    if len(home_vals) != len(labels):
        st.error(f"Radar HOME inválido: {len(home_vals)} valores")
        return

    if len(away_vals) != len(labels):
        st.error(f"Radar AWAY inválido: {len(away_vals)} valores")
        return

    angulos = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angulos = np.concatenate((angulos, [angulos[0]]))

    home_vals = np.concatenate((home_vals, [home_vals[0]]))
    away_vals = np.concatenate((away_vals, [away_vals[0]]))

    fig = plt.figure(figsize=(3.2, 2.8), dpi=120, facecolor="#0e1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0e1117")

    # HOME azul
    ax.plot(angulos, home_vals, linewidth=2, color="#00BFFF")
    ax.fill(angulos, home_vals, alpha=0.25, color="#00BFFF")

    # AWAY laranja
    ax.plot(angulos, away_vals, linewidth=2, color="#FF7A00")
    ax.fill(angulos, away_vals, alpha=0.18, color="#FF7A00")

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels, fontsize=7, color="white")

    ax.set_ylim(0, 100)

    ax.tick_params(axis='y', labelsize=6, colors="white")
    ax.spines["polar"].set_color("#444")

    return fig

def radar_consenso(radars):
    return np.mean(radars, axis=0)

def norm_exg(x): 
    return min(x * 40, 100)

def norm_shots(x): 
    return min((x / 15) * 100, 100)


#CARDS
def cards_ofensivos(radar_home, radar_away, ief_home, ief_away, exg_total):
    
    dominio = dominio_ofensivo(radar_home, radar_away)

    if dominio == "HOME":
        st.success("⚔️ Domínio Ofensivo: HOME")
    elif dominio == "AWAY":
        st.error("⚔️ Domínio Ofensivo: AWAY")
    else:
        st.warning("⚖️ Jogo equilibrado")

    if time_letal(ief_home, exg_total/2):
        st.success("💀🔥⚽ Home LETAL hoje")

    if time_letal(ief_away, exg_total/2):
        st.success("💀🔥⚽ Away LETAL hoje")

    # score = score_jogo(radar_home, radar_away)

    tendencia = tendencia_gols(ief_home, ief_away, exg_total)

    if tendencia == "ALTÍSSIMA":
        st.error("🚨🔥⚽🚨🔥⚽ Altíssima tendência de gols")
    elif tendencia == "ALTA":
        st.warning("🔥⚽🔥⚽ Tendência alta de gols")
    else:
        st.info(f"Tendência: {tendencia}")

# =========================================
# SCORE DEFENSIVO BASE
# =========================================
def score_defensivo(fd, clean_sheet, chs, mgc):

    if pd.isna(fd): fd = 50
    if pd.isna(clean_sheet): clean_sheet = 30
    if pd.isna(chs) or chs == 0: chs = 10
    if pd.isna(mgc) or mgc == 0: mgc = 1

    resistencia = min((chs / 15) * 100, 100)
    concessao = max(0, 100 - (mgc * 40))

    score = (
        fd * 0.35 +
        clean_sheet * 0.25 +
        resistencia * 0.20 +
        concessao * 0.20
    )

    return round(score,1)


def classificar_defesa(score):

    if score >= 60:
        return "⛰️🚫⚽ Defesa MUITO sólida"

    elif score >= 55:
        return "🛡️🚫⚽ Defesa confiável"

    elif score >= 45:
        return "⚠️🚫⚽ Defesa instável"

    else:
        return "🔥🔥🔥⚽⚽⚽Defesa vulnerável"

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
    <div style="background:{cor};
                padding:18px;
                border-radius:14px;
                box-shadow:0 0 10px rgba(0,0,0,0.45);
                color:white;
                font-size:18px;
                font-weight:600;
                margin-bottom:18px;">
        🧠 {row["Interpretacao"]}
        <br>
        <span style="font-size:26px;">{estrelas}</span>
    </div>
    """

    st.markdown(card, unsafe_allow_html=True)

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
    home = linha_exg["Home_Team"]
    away = linha_exg["Visitor_Team"]

    esc_home = escudo_path(home)
    esc_away = escudo_path(away)

    col1, col2, col3 = st.columns([2,1,2])

    with col1:
        st.image(esc_home, width=90)
        st.markdown(f"<center><b>{home}</b></center>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h2 style='text-align:center'>VS</h2>", unsafe_allow_html=True)

    with col3:
        st.image(esc_away, width=90)
        st.markdown(f"<center><b>{away}</b></center>", unsafe_allow_html=True)

    st.markdown("---")

    # ===== ODDS =====
    st.markdown("### 🎯 Odds")
    o1, o2, o3 = st.columns(3)

    with o1:
        st.metric("Odds Casa", linha_exg["Odds_Casa"])
        st.metric("Odd Over 1.5", linha_exg["Odd_Over_1,5FT"])
        st.metric("VR01", get_val(linha_exg, "VR01", "{:.2f}"))

    with o2:
        st.metric("Odds Empate", linha_exg["Odds_Empate"])
        st.metric("Odds Over 2.5", linha_exg["Odds_Over_2,5FT"])
        st.metric("Coef Over1FT", get_val(linha_exg, "COEF_OVER1FT", "{:.2f}"))

    with o3:
        st.metric("Odds Visitante", linha_exg["Odds_Visitante"])
        st.metric("Odds Under 2.5", linha_exg["Odds_Under_2,5FT"])
        st.metric("BTTS YES", linha_exg["Odd_BTTS_YES"])

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

        st.info(
            f"""⚡ Probabilidade de Gol no 1º Tempo

🔥 Gol HT: {ht['Prob_Gol_HT']}%   |   ❄️ 0x0 HT: {ht['Prob_0x0_HT']}%

🏠 Home marca HT: {ht['Gol_HT_Home_%']}%   |   ✈️ Away marca HT: {ht['Gol_HT_Away_%']}%

{ht['Selo_HT']}
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

    
    st.markdown("### 🧱 Defesa Probabilística")

    def_home = linha_vg["Clean_Sheet_Home_%"]
    def_away = linha_vg["Clean_Sheet_Away_%"]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(linha_vg["Home_Team"], round(def_home,1))
        st.info(classificar_defesa(def_home))

    with c2:
        st.metric(linha_vg["Visitor_Team"], round(def_away,1))
        st.info(classificar_defesa(def_away))

    st.markdown("### 🧠 Leitura de Valor Ofensivo")

    col1, col2 = st.columns(2)

    with col1:
        st.info(leitura_ofensiva(
            linha_vg["Home_Team"],
            *radar_home_vg
        ))

    with col2:
        st.info(leitura_ofensiva(
            linha_vg["Visitor_Team"],
            *radar_away_vg
        ))
