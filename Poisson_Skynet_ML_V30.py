# ============================================================
# POISSON SKYNET ML V30
# V30.2.0
#
# Engine de Machine Learning
#
# Objetivo:
# Gerar BASE_ML_V30.csv
#
# Desenvolvido a partir da Poisson Skynet V42
# ============================================================

# ============================================================
# BLOCO 01 - IMPORTAÇÕES
# ============================================================

from pathlib import Path
from datetime import datetime
import warnings
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# BLOCO 02 - CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARQ_HISTORICO = BASE_DIR / "CSV_LIMPO.csv"

ARQ_PACKBALL = BASE_DIR / "PackBall.csv"

ARQ_BASE_ML = BASE_DIR / "BASE_ML_V30.csv"

ARQ_LOG = BASE_DIR / "LOG_V30.txt"

DATA_INICIO = time.time()

MAX_GOLS = 5

SEED = 42

pd.set_option("display.max_columns", None)

pd.set_option("display.width", 250)

np.random.seed(SEED)

# ============================================================
# BLOCO 03 - LEITURA DOS DADOS
# ============================================================

print("=" * 70)
print("POISSON SKYNET ML V30")
print("=" * 70)

print("\nCarregando histórico...")

df_hist = pd.read_csv(

    ARQ_HISTORICO,

    sep=";",

    encoding="utf-8-sig",

    low_memory=False

)

print(f"Histórico carregado: {len(df_hist):,} jogos")

print("\nCarregando PackBall...")

df_pack = pd.read_csv(

    ARQ_PACKBALL,

    sep=";",

    encoding="utf-8-sig",

    low_memory=False

)

print(f"Jogos do dia: {len(df_pack):,}")

print("\nArquivos carregados com sucesso.")

# ============================================================
# BLOCO 04 - AUDITORIA DA BASE
# ============================================================

print("\n" + "=" * 70)
print("AUDITORIA DA BASE")
print("=" * 70)

# ============================================================
# REMOVE COLUNAS DUPLICADAS
# ============================================================

df_hist = df_hist.loc[:, ~df_hist.columns.duplicated()]

df_pack = df_pack.loc[:, ~df_pack.columns.duplicated()]

# ============================================================
# VERIFICAÇÕES
# ============================================================

print(f"\nHistórico")

print(f"Linhas...............: {len(df_hist):,}")

print(f"Colunas..............: {len(df_hist.columns):,}")

print(f"Nulos................: {int(df_hist.isna().sum().sum()):,}")

print(f"Duplicados...........: {int(df_hist.duplicated().sum()):,}")

print(f"Memória..............: {df_hist.memory_usage(deep=True).sum()/1024/1024:.2f} MB")

print(f"\nPackBall")

print(f"Linhas...............: {len(df_pack):,}")

print(f"Colunas..............: {len(df_pack.columns):,}")

print(f"Nulos................: {int(df_pack.isna().sum().sum()):,}")

print(f"Duplicados...........: {int(df_pack.duplicated().sum()):,}")

# ============================================================
# DATAS
# ============================================================

if "Hour" in df_hist.columns:

    df_hist["Hour"] = pd.to_datetime(

        df_hist["Hour"],

        dayfirst=True,

        errors="coerce"

    )

    print(

        f"\nDatas inválidas......: "

        f"{int(df_hist['Hour'].isna().sum()):,}"

    )

# ============================================================
# COLUNAS CONSTANTES
# ============================================================

constantes = [

    c

    for c in df_hist.columns

    if df_hist[c].nunique(dropna=False) <= 1

]

print(

    f"Colunas constantes...: "

    f"{len(constantes)}"

)

if constantes:

    print("\nPrimeiras constantes:")

    for c in constantes[:20]:

        print(f" - {c}")

# ============================================================
# RESUMO
# ============================================================

print("\nAuditoria concluída com sucesso.")

print("=" * 70)


# ============================================================
# BLOCO 05 - PADRONIZAÇÃO DA BASE
# ============================================================

print("\n" + "=" * 70)
print("PADRONIZAÇÃO DA BASE")
print("=" * 70)

# ============================================================
# REMOVE ESPAÇOS DOS NOMES DAS COLUNAS
# ============================================================

df_hist.columns = (

    df_hist.columns

    .str.strip()

)

df_pack.columns = (

    df_pack.columns

    .str.strip()

)

# ============================================================
# REMOVE ESPAÇOS DOS TEXTOS
# ============================================================

for df in [

    df_hist,

    df_pack

]:

    col_texto = df.select_dtypes(

        include="object"

    ).columns

    for col in col_texto:

        df[col] = (

            df[col]

            .astype(str)

            .str.strip()

            .replace(

                {

                    "nan": np.nan,

                    "None": np.nan,

                    "": np.nan

                }

            )

        )

# ============================================================
# CONVERTE DATAS
# ============================================================

for df in [

    df_hist,

    df_pack

]:

    if "Hour" in df.columns:

        df["Hour"] = pd.to_datetime(

            df["Hour"],

            dayfirst=True,

            errors="coerce"

        )

# ============================================================
# CONVERTE RESULTADOS
# ============================================================

COL_RESULTADOS = [

    "Result Home",

    "Result Visitor",

    "Result_Home_HT",

    "Result_Visitor_HT"

]

for df in [

    df_hist,

    df_pack

]:

    for col in COL_RESULTADOS:

        if col in df.columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

# ============================================================
# CONVERTE TODAS AS ODDS
# ============================================================

for df in [

    df_hist,

    df_pack

]:

    odds_cols = [

        c

        for c in df.columns

        if (

            "Odd" in c

            or "Odds" in c

        )

    ]

    for col in odds_cols:

        df[col] = (

            df[col]

            .astype(str)

            .str.replace(",", ".", regex=False)

        )

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )

# ============================================================
# CONVERTE DEMAIS NUMÉRICAS
# ============================================================

IGNORAR = [

    "League",

    "Country",

    "Home_Team",

    "Visitor_Team",

    "Hour"

]

for df in [

    df_hist,

    df_pack

]:

    for col in df.columns:

        if col in IGNORAR:

            continue

        if df[col].dtype == object:

            try:

                serie = (

                    df[col]

                    .astype(str)

                    .str.replace(",", ".", regex=False)

                )

                convertido = pd.to_numeric(

                    serie,

                    errors="coerce"

                )

                if convertido.notna().sum() > 0:

                    df[col] = convertido

            except:

                pass

# ============================================================
# ORDENA HISTÓRICO
# ============================================================

if "Hour" in df_hist.columns:

    df_hist = (

        df_hist

        .sort_values(

            "Hour"

        )

        .reset_index(

            drop=True

        )

    )

# ============================================================
# RESUMO
# ============================================================

print("\nBase padronizada com sucesso.")

print(f"Histórico : {len(df_hist):,} jogos")

print(f"PackBall  : {len(df_pack):,} jogos")

print("=" * 70)


# ============================================================
# BLOCO IA 01 - FEATURES OFICIAIS
# ============================================================

FEATURES_ML = [

    # ========================================================
    # MERCADO
    # ========================================================

    "Odds_Casa",
    "Odds_Empate",
    "Odds_Visitante",

    "Odd_Over_0,5FT",
    "Odd_Under_0,5FT",

    "Odd_Over_1,5FT",
    "Odd_Under_1,5FT",

    "Odds_Over_2,5FT",
    "Odds_Under_2,5FT",

    "Odd_Over_3,5FT",
    "Odd_Under_3,5FT",

    "Odd_Over_4,5FT",
    "Odd_Under_4,5FT",

    "Odd_BTTS_YES",
    "Odd_BTTS_NO",

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    "PPJH",
    "PPJA",

    "FAH",
    "FAA",

    "FDH",
    "FDA",

    "MGFH",
    "MGFA",

    "MGCH",
    "MGCA",

    "MG_Global",

    "VG_H",
    "VG_A",

    # ========================================================
    # HT
    # ========================================================

    "MGF_HT_Home",
    "MGF_HT_Away",

    "MGC_HT_Home",
    "MGC_HT_Away",

    # ========================================================
    # FORMA
    # ========================================================

    "Win4_H",
    "Win4_A",

    "Los4_H",
    "Los4_A",

    "Eficiência_H",
    "Eficiência_A",

    "Eficiência_HT_H",
    "Eficiência_HT_A",

    "Eficiência_2nd_H",
    "Eficiência_2nd_A",

    "Scored_First_H",
    "Scored_First_A",

    "Conceded_First_H",
    "Conceded_First_A",

    "FS_Win_H",
    "FS_Win_A",

    # ========================================================
    # POISSON MGF
    # ========================================================

    "ExG_Home_MGF",
    "ExG_Away_MGF",

    # ========================================================
    # POISSON ATK x DEF
    # ========================================================

    "ExG_Home_ATKxDEF",
    "ExG_Away_ATKxDEF",

    # ========================================================
    # POISSON VG
    # ========================================================

    "ExG_Home_VG",
    "ExG_Away_VG",

    # ========================================================
    # CONSENSO
    # ========================================================

    "ExG_Home_Consenso",
    "ExG_Away_Consenso",

    # ========================================================
    # POISSON
    # ========================================================

    "BTTS_%",
    "Clean_Sheet_Home_%",
    "Clean_Sheet_Away_%",

    "Odd_Justa_Home",
    "Odd_Justa_Draw",
    "Odd_Justa_Away",

    "Prob_Over_0_5",
    "Prob_Over_1_5",
    "Prob_Over_2_5",
    "Prob_Over_3_5",
    "Prob_Over_4_5",

    "Prob_Under_0_5",
    "Prob_Under_1_5",
    "Prob_Under_2_5",
    "Prob_Under_3_5",
    "Prob_Under_4_5",

    # ========================================================
    # INDICADORES
    # ========================================================

    "VR01",
    "VAR01_00",
    "VAR01_01",
    "COEF_OVER1FT",

    # ========================================================
    # CONTEXTO
    # ========================================================

    "League",
    "Country"

]

# ============================================================
# TARGETS
# ============================================================

TARGETS = [

    "LAY00",
    "LAY01",
    "LAY10",
    "LAY22",
    "LAYGH",
    "LAYGA"

]


# ============================================================
# BLOCO IA 02 - PREPARAÇÃO DA BASE ML
# ============================================================

print("\n" + "=" * 70)
print("PREPARANDO BASE MACHINE LEARNING")
print("=" * 70)

# ============================================================
# REMOVE COLUNAS DUPLICADAS
# ============================================================

df_hist = df_hist.loc[:, ~df_hist.columns.duplicated()]

# ============================================================
# FEATURES EXISTENTES
# ============================================================

FEATURES_VALIDAS = [

    col

    for col in FEATURES_ML

    if col in df_hist.columns

]

print(f"\nFeatures encontradas : {len(FEATURES_VALIDAS)}")

print(f"Features esperadas   : {len(FEATURES_ML)}")

# ============================================================
# TARGETS
# ============================================================

for col in TARGETS:

    if col not in df_hist.columns:

        df_hist[col] = np.nan

# ============================================================
# BASE ML
# ============================================================

COLUNAS_BASE = (

    FEATURES_VALIDAS

    + TARGETS

)

COLUNAS_BASE = list(

    dict.fromkeys(

        COLUNAS_BASE

    )

)

BASE_ML = (

    df_hist

    [

        COLUNAS_BASE

    ]

    .copy()

)

# ============================================================
# CONVERTE NUMÉRICAS
# ============================================================

for col in FEATURES_VALIDAS:

    if col in [

        "League",

        "Country"

    ]:

        continue

    BASE_ML[col] = pd.to_numeric(

        BASE_ML[col],

        errors="coerce"

    )

# ============================================================
# PREENCHE NaN
# ============================================================

for col in FEATURES_VALIDAS:

    if col in [

        "League",

        "Country"

    ]:

        continue

    mediana = BASE_ML[col].median()

    if pd.isna(mediana):

        mediana = 0

    BASE_ML[col] = (

        BASE_ML[col]

        .fillna(

            mediana

        )

    )

# ============================================================
# RESUMO
# ============================================================

print(f"\nJogos..............: {len(BASE_ML):,}")

print(f"Features...........: {len(FEATURES_VALIDAS)}")

print(f"Targets............: {len(TARGETS)}")

print("=" * 70)


# ============================================================
# BLOCO IA 03 - FEATURES OFICIAIS DA IA
# ============================================================

FEATURES_ML = [

    # ========================================================
    # MERCADO
    # ========================================================

    "Odds_Casa",
    "Odds_Empate",
    "Odds_Visitante",

    "Odd_Over_0,5FT",
    "Odd_Under_0,5FT",

    "Odd_Over_1,5FT",
    "Odd_Under_1,5FT",

    "Odds_Over_2,5FT",
    "Odds_Under_2,5FT",

    "Odd_Over_3,5FT",
    "Odd_Under_3,5FT",

    "Odd_Over_4,5FT",
    "Odd_Under_4,5FT",

    "Odd_BTTS_YES",
    "Odd_BTTS_NO",

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    "PPJH",
    "PPJA",

    "FAH",
    "FAA",

    "FDH",
    "FDA",

    "MGFH",
    "MGFA",

    "MGCH",
    "MGCA",

    "MG_Global",

    "VG_H",
    "VG_A",

    # ========================================================
    # FORMA
    # ========================================================

    "Win4_H",
    "Win4_A",

    "Los4_H",
    "Los4_A",

    "Eficiência_H",
    "Eficiência_A",

    "Eficiência_HT_H",
    "Eficiência_HT_A",

    "Eficiência_2nd_H",
    "Eficiência_2nd_A",

    "FS_Win_H",
    "FS_Win_A",

    "Scored_First_H",
    "Scored_First_A",

    "Conceded_First_H",
    "Conceded_First_A",

    # ========================================================
    # POISSON MGF
    # ========================================================

    "ExG_Home_MGF",
    "ExG_Away_MGF",

    # ========================================================
    # POISSON ATK x DEF
    # ========================================================

    "ExG_Home_ATKxDEF",
    "ExG_Away_ATKxDEF",

    # ========================================================
    # POISSON VG
    # ========================================================

    "ExG_Home_VG",
    "ExG_Away_VG",

    # ========================================================
    # CONSENSO
    # ========================================================

    "ExG_Home_Consenso",
    "ExG_Away_Consenso",

    # ========================================================
    # POISSON
    # ========================================================

    "BTTS_%",
    "Clean_Sheet_Home_%",
    "Clean_Sheet_Away_%",

    "Odd_Justa_Home",
    "Odd_Justa_Draw",
    "Odd_Justa_Away",

    "Prob_Over_0_5",
    "Prob_Over_1_5",
    "Prob_Over_2_5",
    "Prob_Over_3_5",
    "Prob_Over_4_5",

    "Prob_Under_0_5",
    "Prob_Under_1_5",
    "Prob_Under_2_5",
    "Prob_Under_3_5",
    "Prob_Under_4_5",

    # ========================================================
    # INDICADORES PRÓPRIOS
    # ========================================================
    "VR01",
    "COEF_OVER1,5FT",

    # ========================================================
    # CONTEXTO
    # ========================================================
    "League",
    "Country"
]
# ============================================================
# TARGETS
# ============================================================
TARGETS = [
    "LAY00",
    "LAY01",
    "LAY10",
    "LAY22",
    "LAYGH",
    "LAYGA"]

# ============================================================
# BLOCO IA 04 - INDICADORES DERIVADOS
# ============================================================

print("\nCalculando indicadores...")

# ============================================================
# VARIÁVEIS DERIVADAS
# ============================================================

BASE_ML["VAR01_00"] = (

    BASE_ML["Odds_Empate"]

    - BASE_ML["Odds_Casa"]

).round(2)

BASE_ML["VAR01_01"] = (

    BASE_ML["Odds_Empate"]

    - BASE_ML["Odds_Visitante"]

).round(2)

BASE_ML["COEF_OVER1FT"] = (

    BASE_ML["Odds_Empate"]

    / BASE_ML["Odds_Over_2,5FT"]

).round(2)

BASE_ML["VR01"] = (

    BASE_ML["Odd_BTTS_YES"]

    - BASE_ML["Odds_Over_2,5FT"]

).round(2)

# ============================================================
# VALIDAÇÃO
# ============================================================

print("Indicadores calculados:")

print(" - VAR01_00")

print(" - VAR01_01")

print(" - COEF_OVER1FT")

print(" - VR01")

# ============================================================
# BLOCO IA 04 - FUNÇÕES BASE DO POISSON
# ============================================================

from scipy.stats import poisson


def matriz_poisson(lh, la):

    return np.outer(

        [poisson.pmf(i, lh) for i in range(MAX_GOLS + 1)],

        [poisson.pmf(j, la) for j in range(MAX_GOLS + 1)]

    )


def safe_odds(prob):

    if prob <= 0:

        return np.nan

    return round(1 / prob, 2)


def placar_mais_provavel(matriz):

    if matriz.size == 0:

        return np.nan

    if np.all(np.isnan(matriz)):

        return np.nan

    i, j = np.unravel_index(

        np.argmax(matriz),

        matriz.shape

    )

    return f"{i}x{j}"


def calcular_btts_e_odd(matriz):

    prob = sum(

        matriz[i][j]

        for i in range(1, MAX_GOLS + 1)

        for j in range(1, MAX_GOLS + 1)

    )

    return (

        round(prob * 100, 2),

        safe_odds(prob)

    )


def calcular_primeiro_gol(

    exg_home,

    exg_away,

    matriz,

    CHM=None,

    CAM=None

):

    if pd.isna(exg_home):

        return np.nan, np.nan

    if pd.isna(exg_away):

        return np.nan, np.nan

    lam_total = exg_home + exg_away

    if lam_total <= 0:

        return np.nan, np.nan

    p_zero = matriz[0, 0]

    base_home = exg_home / lam_total

    base_away = exg_away / lam_total

    ajuste_home = 1

    ajuste_away = 1

    if (

        pd.notna(CHM)

        and

        pd.notna(CAM)

    ):

        total = CHM + CAM

        if total > 0:

            ajuste_home = CHM / total

            ajuste_away = CAM / total

    home = (

        0.7 * base_home

        +

        0.3 * ajuste_home

    ) * (

        1 - p_zero

    )

    away = (

        0.7 * base_away

        +

        0.3 * ajuste_away

    ) * (

        1 - p_zero

    )

    return (

        round(home * 100, 2),

        round(away * 100, 2)

    )
# ============================================================
# BLOCO IA 05 - POISSON MGF
# ============================================================

if pd.notna(MGF_H) and pd.notna(MGF_A):

    matriz = matriz_poisson(

        MGF_H,

        MGF_A

    )

    home_fp, away_fp = calcular_primeiro_gol(

        matriz,

        H,

        A,

        jogo,

        MGF_H,

        MGF_A,

        MGC_H,

        MGC_A,

        CHM,

        CAM,

        CHS,

        CAS,

        EPS

    )

    linha_mgf["Home_Abrir_Placar"] = home_fp

    linha_mgf["Away_Abrir_Placar"] = away_fp

    prob_home = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i)

    )

    prob_draw = sum(

        matriz[i][i]

        for i in range(MAX_GOLS + 1)

    )

    prob_away = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i + 1, MAX_GOLS + 1)

    )

    btts_pct, btts_odd = calcular_btts_e_odd(

        matriz

    )

    linha_mgf.update({

        "ExG_Home_MGF": round(MGF_H, 2),

        "ExG_Away_MGF": round(MGF_A, 2),

        "Placar_Mais_Provavel": placar_mais_provavel(

            matriz

        ),

        "BTTS_%": round(

            btts_pct,

            2

        ),

        "Odd_Justa_BTTS": btts_odd,

        "Odd_Justa_Home": safe_odds(

            prob_home

        ),

        "Odd_Justa_Draw": safe_odds(

            prob_draw

        ),

        "Odd_Justa_Away": safe_odds(

            prob_away

        ),

        "Clean_Sheet_Home_%": round(

            matriz[:, 0].sum() * 100,

            2

        ),

        "Clean_Sheet_Away_%": round(

            matriz[0, :].sum() * 100,

            2

        ),

        "Interpretacao": interpretar_forca_mix(

            jogo["Home_Team"],

            jogo["Visitor_Team"],

            prob_home,

            prob_away,

            jogo["Odds_Casa"],

            jogo["Odds_Visitante"],

            MGF_H,

            MGF_A,

            VR01,

            COEF_OVER1FT

        )

    })

    for col in [

        "PPJH",

        "PPJA",

        "FAH",

        "FAA",

        "FDH",

        "FDA",

        "Posse_Bola_Home",

        "Posse_Bola_Away",

        "Precisao_CG_H",

        "Precisao_CG_A",

        "Clean_Games_H",

        "Clean_Games_A"

    ]:

        linha_mgf[col] = jogo.get(col)

    for i in range(MAX_GOLS + 1):

        for j in range(MAX_GOLS + 1):

            linha_mgf[f"{i}x{j}"] = round(

                matriz[i][j] * 100,

                2

            )

else:

    linha_mgf.update({

        "ExG_Home_MGF": np.nan,

        "ExG_Away_MGF": np.nan,

        "Placar_Mais_Provavel": np.nan,

        "BTTS_%": np.nan,

        "Odd_Justa_BTTS": np.nan,

        "Odd_Justa_Home": np.nan,

        "Odd_Justa_Draw": np.nan,

        "Odd_Justa_Away": np.nan,

        "Clean_Sheet_Home_%": np.nan,

        "Clean_Sheet_Away_%": np.nan,

        "Interpretacao": np.nan

    })


# =====================================================
# POISSON — ATAQUE x DEFESA
# =====================================================

if pd.notna(MGF_H) and pd.notna(MGC_A) and pd.notna(MGF_A) and pd.notna(MGC_H):

    ExG_H_atkdef = (MGF_H + MGC_A) / 2

    ExG_A_atkdef = (MGF_A + MGC_H) / 2

else:

    ExG_H_atkdef = np.nan

    ExG_A_atkdef = np.nan


if pd.notna(ExG_H_atkdef) and pd.notna(ExG_A_atkdef):

    matriz = matriz_poisson(

        ExG_H_atkdef,

        ExG_A_atkdef

    )

    home_fp, away_fp = calcular_primeiro_gol(

        matriz,

        H,

        A,

        jogo,

        ExG_H_atkdef,

        ExG_A_atkdef,

        MGC_H,

        MGC_A,

        CHM,

        CAM,

        CHS,

        CAS,

        EPS

    )

    linha_exg["Home_Abrir_Placar"] = home_fp

    linha_exg["Away_Abrir_Placar"] = away_fp

    prob_home = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i)

    )

    prob_draw = sum(

        matriz[i][i]

        for i in range(MAX_GOLS + 1)

    )

    prob_away = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i + 1, MAX_GOLS + 1)

    )

    btts_pct, btts_odd = calcular_btts_e_odd(

        matriz

    )

    linha_exg.update({

        "ExG_Home_ATKxDEF": round(

            ExG_H_atkdef,

            2

        ),

        "ExG_Away_ATKxDEF": round(

            ExG_A_atkdef,

            2

        ),

        "Placar_Mais_Provavel": placar_mais_provavel(

            matriz

        ),

        "BTTS_%": round(

            btts_pct,

            2

        ),

        "Odd_Justa_BTTS": btts_odd,

        "Odd_Justa_Home": safe_odds(

            prob_home

        ),

        "Odd_Justa_Draw": safe_odds(

            prob_draw

        ),

        "Odd_Justa_Away": safe_odds(

            prob_away

        ),

        "Clean_Sheet_Home_%": round(

            matriz[:, 0].sum() * 100,

            2

        ),

        "Clean_Sheet_Away_%": round(

            matriz[0, :].sum() * 100,

            2

        ),

        "Interpretacao": interpretar_forca_mix(

            jogo["Home_Team"],

            jogo["Visitor_Team"],

            prob_home,

            prob_away,

            jogo["Odds_Casa"],

            jogo["Odds_Visitante"],

            ExG_H_atkdef,

            ExG_A_atkdef,

            VR01,

            COEF_OVER1FT

        )

    })

    for col in [

        "PPJH",

        "PPJA",

        "FAH",

        "FAA",

        "FDH",

        "FDA",

        "Posse_Bola_Home",

        "Posse_Bola_Away",

        "Precisao_CG_H",

        "Precisao_CG_A",

        "Clean_Games_H",

        "Clean_Games_A"

    ]:

        linha_exg[col] = jogo.get(col)

    for i in range(MAX_GOLS + 1):

        for j in range(MAX_GOLS + 1):

            linha_exg[f"{i}x{j}"] = round(

                matriz[i][j] * 100,

                2

            )

else:

    linha_exg.update({

        "ExG_Home_ATKxDEF": np.nan,

        "ExG_Away_ATKxDEF": np.nan,

        "Placar_Mais_Provavel": np.nan,

        "BTTS_%": np.nan,

        "Odd_Justa_BTTS": np.nan,

        "Odd_Justa_Home": np.nan,

        "Odd_Justa_Draw": np.nan,

        "Odd_Justa_Away": np.nan,

        "Clean_Sheet_Home_%": np.nan,

        "Clean_Sheet_Away_%": np.nan,

        "Home_Abrir_Placar": np.nan,

        "Away_Abrir_Placar": np.nan,

        "Interpretacao": "Dados insuficientes para ATKxDEF"

    })

    for i in range(MAX_GOLS + 1):

        for j in range(MAX_GOLS + 1):

            linha_exg[f"{i}x{j}"] = np.nan

saida_exg.append(

    linha_exg

)
# =====================================================
# POISSON — VG
# =====================================================

ExG_H_vg = np.nan
ExG_A_vg = np.nan

if (
    pd.notna(H.get("Media_VG_H"))
    and pd.notna(A.get("Media_VG_A"))
    and pd.notna(jogo.get("Odds_Casa"))
    and pd.notna(jogo.get("Odds_Visitante"))
):

    Odd_H_jogo = jogo.get("Odds_Casa")
    Odd_A_jogo = jogo.get("Odds_Visitante")

    ExG_H_vg = H.get("Media_VG_H") * Odd_A_jogo
    ExG_A_vg = A.get("Media_VG_A") * Odd_H_jogo

if pd.notna(ExG_H_vg) and pd.notna(ExG_A_vg):

    matriz = matriz_poisson(

        ExG_H_vg,

        ExG_A_vg

    )

    home_fp, away_fp = calcular_primeiro_gol(

        matriz,

        H,

        A,

        jogo,

        ExG_H_vg,

        ExG_A_vg,

        MGC_H,

        MGC_A,

        CHM,

        CAM,

        CHS,

        CAS,

        EPS

    )

    linha_vg["Home_Abrir_Placar"] = home_fp
    linha_vg["Away_Abrir_Placar"] = away_fp

    prob_home = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i)

    )

    prob_draw = sum(

        matriz[i][i]

        for i in range(MAX_GOLS + 1)

    )

    prob_away = sum(

        matriz[i][j]

        for i in range(MAX_GOLS + 1)

        for j in range(i + 1, MAX_GOLS + 1)

    )

    btts_pct, btts_odd = calcular_btts_e_odd(

        matriz

    )

    linha_vg.update({

        "ExG_Home_VG": round(

            ExG_H_vg,

            2

        ),

        "ExG_Away_VG": round(

            ExG_A_vg,

            2

        ),

        "Placar_Mais_Provavel": placar_mais_provavel(

            matriz

        ),

        "BTTS_%": round(

            btts_pct,

            2

        ),

        "Odd_Justa_BTTS": btts_odd,

        "Odd_Justa_Home": safe_odds(

            prob_home

        ),

        "Odd_Justa_Draw": safe_odds(

            prob_draw

        ),

        "Odd_Justa_Away": safe_odds(

            prob_away

        ),

        "Clean_Sheet_Home_%": round(

            matriz[:, 0].sum() * 100,

            2

        ),

        "Clean_Sheet_Away_%": round(

            matriz[0, :].sum() * 100,

            2

        ),

        "Interpretacao": interpretar_forca_mix(

            jogo.get("Home_Team"),

            jogo.get("Visitor_Team"),

            prob_home,

            prob_away,

            Odd_H_jogo,

            Odd_A_jogo,

            ExG_H_vg,

            ExG_A_vg,

            VR01,

            COEF_OVER1FT

        )

    })

    for col in [

        "PPJH",
        "PPJA",
        "FAH",
        "FAA",
        "FDH",
        "FDA",
        "Posse_Bola_Home",
        "Posse_Bola_Away",
        "Precisao_CG_H",
        "Precisao_CG_A",
        "Clean_Games_H",
        "Clean_Games_A"

    ]:

        linha_vg[col] = jogo.get(col)

    for i in range(MAX_GOLS + 1):

        for j in range(MAX_GOLS + 1):

            linha_vg[f"{i}x{j}"] = round(

                matriz[i][j] * 100,

                2

            )

else:

    linha_vg.update({

        "ExG_Home_VG": np.nan,
        "ExG_Away_VG": np.nan,
        "Placar_Mais_Provavel": np.nan,
        "BTTS_%": np.nan,
        "Odd_Justa_BTTS": np.nan,
        "Odd_Justa_Home": np.nan,
        "Odd_Justa_Draw": np.nan,
        "Odd_Justa_Away": np.nan,
        "Clean_Sheet_Home_%": np.nan,
        "Clean_Sheet_Away_%": np.nan,
        "Home_Abrir_Placar": np.nan,
        "Away_Abrir_Placar": np.nan,
        "Interpretacao": "Dados insuficientes para VG"

    })

    for i in range(MAX_GOLS + 1):

        for j in range(MAX_GOLS + 1):

            linha_vg[f"{i}x{j}"] = np.nan

saida_vg.append(

    linha_vg

)

# =====================================================
# CONSENSO
# =====================================================

linha_consenso = base.copy()

linha_consenso.update({

    'ExG_Home_Consenso': round(lambda_home, 2) if pd.notna(lambda_home) else np.nan,
    'ExG_Away_Consenso': round(lambda_away, 2) if pd.notna(lambda_away) else np.nan,
    'ExG_Total': exg_total,
    'Dominio_Ofensivo': dominio,
    'Time_Letal': time_letal,
    'Tendencia_Gols': tendencia,
    'Defesa_Home': defesa_home,
    'Defesa_Away': defesa_away,

    # =====================================================
    # CONSENSO TEMPORAL
    # =====================================================

    "GF_0-15_Home": H.get("GF_0-15_Home"),
    "GF_0-15_Away": A.get("GF_0-15_Away"),

    "GF_16-30_Home": H.get("GF_16-30_Home"),
    "GF_16-30_Away": A.get("GF_16-30_Away"),

    "GF_31-45_Home": H.get("GF_31-45_Home"),
    "GF_31-45_Away": A.get("GF_31-45_Away"),

    "GF_46-60_Home": H.get("GF_46-60_Home"),
    "GF_46-60_Away": A.get("GF_46-60_Away"),

    "GF_61-75_Home": H.get("GF_61-75_Home"),
    "GF_61-75_Away": A.get("GF_61-75_Away"),

    "GF_76-90_Home": H.get("GF_76-90_Home"),
    "GF_76-90_Away": A.get("GF_76-90_Away"),

    # =====================================================
    # GC
    # =====================================================

    "GC_0-15_Home": H.get("GC_0-15_Home"),
    "GC_0-15_Away": A.get("GC_0-15_Away"),

    "GC_16-30_Home": H.get("GC_16-30_Home"),
    "GC_16-30_Away": A.get("GC_16-30_Away"),

    "GC_31-45_Home": H.get("GC_31-45_Home"),
    "GC_31-45_Away": A.get("GC_31-45_Away"),

    "GC_46-60_Home": H.get("GC_46-60_Home"),
    "GC_46-60_Away": A.get("GC_46-60_Away"),

    "GC_61-75_Home": H.get("GC_61-75_Home"),
    "GC_61-75_Away": A.get("GC_61-75_Away"),

    "GC_76-90_Home": H.get("GC_76-90_Home"),
    "GC_76-90_Away": A.get("GC_76-90_Away"),

    # =====================================================
    # EFICIÊNCIA
    # =====================================================

    "Eficiência_HT_H": H.get("Eficiência_HT_H"),
    "Eficiência_HT_A": A.get("Eficiência_HT_A"),

    "Eficiência_2nd_H": H.get("Eficiência_2nd_H"),
    "Eficiência_2nd_A": A.get("Eficiência_2nd_A"),

    "Eficiência_H": H.get("Eficiência_H"),
    "Eficiência_A": A.get("Eficiência_A"),

    # =====================================================
    # COMPORTAMENTAL
    # =====================================================

    "FS_Win_H": H.get("FS_Win_H"),
    "FS_Win_A": A.get("FS_Win_A"),

    "Changer_H": H.get("Changer_H"),
    "Changer_A": A.get("Changer_A"),

    "Clean_Games_H": H.get("Clean_Games_H"),
    "Clean_Games_A": A.get("Clean_Games_A"),

    "NS_Games_H": H.get("NS_Games_H"),
    "NS_Games_A": A.get("NS_Games_A"),

    "Win4_H": H.get("Win4_H"),
    "Win4_A": A.get("Win4_A"),

    "Los4_H": H.get("Los4_H"),
    "Los4_A": A.get("Los4_A"),

    # =====================================================
    # CONSENSOS
    # =====================================================

    "Home_Abrir_Placar_Consenso": np.nanmean([
        linha_mgf.get("Home_Abrir_Placar"),
        linha_exg.get("Home_Abrir_Placar"),
        linha_vg.get("Home_Abrir_Placar")
    ]),

    "Away_Abrir_Placar_Consenso": np.nanmean([
        linha_mgf.get("Away_Abrir_Placar"),
        linha_exg.get("Away_Abrir_Placar"),
        linha_vg.get("Away_Abrir_Placar")
    ]),

    "Clean_Sheet_Home_Consenso": np.nanmean([
        linha_mgf.get("Clean_Sheet_Home_%"),
        linha_exg.get("Clean_Sheet_Home_%"),
        linha_vg.get("Clean_Sheet_Home_%")
    ]),

    "Clean_Sheet_Away_Consenso": np.nanmean([
        linha_mgf.get("Clean_Sheet_Away_%"),
        linha_exg.get("Clean_Sheet_Away_%"),
        linha_vg.get("Clean_Sheet_Away_%")
    ]),

    # MGF

    "ExG_Home_MGF": linha_mgf.get("ExG_Home_MGF"),
    "ExG_Away_MGF": linha_mgf.get("ExG_Away_MGF"),

    # ATK x DEF

    "ExG_Home_ATKxDEF": linha_exg.get("ExG_Home_ATKxDEF"),
    "ExG_Away_ATKxDEF": linha_exg.get("ExG_Away_ATKxDEF"),

    # VG

    "ExG_Home_VG": linha_vg.get("ExG_Home_VG"),
    "ExG_Away_VG": linha_vg.get("ExG_Away_VG"),

})

for col in [

    "PPJH",
    "PPJA",
    "FAH",
    "FAA",
    "FDH",
    "FDA",
    "Posse_Bola_Home",
    "Posse_Bola_Away",
    "Precisao_CG_H",
    "Precisao_CG_A",
    "Clean_Games_H",
    "Clean_Games_A"

]:

    linha_consenso[col] = jogo.get(col)

saida_consenso.append(

    linha_consenso

)



