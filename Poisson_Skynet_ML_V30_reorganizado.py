# =========================================================
# BLOCO 1 — LEITURA E PREPARAÇÃO (VERSÃO ESTÁVEL SIMPLES)
# =========================================================

pd.set_option('display.max_columns', None)

# =========================================================
# 1️⃣ HISTÓRICO (df)
# =========================================================

df = pd.read_csv(
    "/content/drive/MyDrive/Colab Notebooks/CONVERTENDO/CSV_LIMPO.csv",
    sep=';',
    encoding='utf-8-sig',
    low_memory=False
)

df = df.replace({pd.NA: np.nan})

# 🔥 ÚNICA REGRA IMPORTANTE
df['Hour'] = pd.to_datetime(
    df['Hour'],
    dayfirst=True,
    errors='raise'
)

df = df.sort_values('Hour').reset_index(drop=True)

# 🔥🔥🔥 CORREÇÃO DEFINITIVA DE STRING → FLOAT (SÓ COLUNA K+)

for c in df.columns[10:]:

    df[c] = (
        df[c]
        .astype(str)
        .str.replace(',', '.', regex=False)
    )

    df[c] = pd.to_numeric(
        df[c],
        errors='coerce'
    )

print("✅ df (histórico):", df.shape)

# =========================================================
# 2️⃣ PACKBALL (df_teams)
# =========================================================

df_teams = pd.read_csv(
    "/content/drive/MyDrive/Colab Notebooks/CONVERTENDO/CSV_CRU/PackBall Custom EXP GOLS (01_07_2026).csv",
    sep=';',
    encoding='utf-8-sig',
    keep_default_na=True
)

df_teams = df_teams.replace({pd.NA: np.nan})

df_teams['Hour'] = pd.to_datetime(
    df_teams['Hour'],
    format='mixed',
    dayfirst=True,
    errors='raise'
)

df_teams = (
    df_teams
    .sort_values('Hour')
    .reset_index(drop=True)
)

# limpar textos

for c in [
    'Country',
    'Short',
    'League',
    'Home_Team',
    'Visitor_Team'
]:

    if c in df_teams.columns:

        df_teams[c] = (
            df_teams[c]
            .astype(str)
            .str.strip()
        )

# 🔥🔥🔥 CORREÇÃO DEFINITIVA DE STRING → FLOAT (SÓ COLUNA K+)

for c in df_teams.columns[10:]:

    df_teams[c] = (
        df_teams[c]
        .astype(str)
        .str.replace(',', '.', regex=False)
    )

    df_teams[c] = pd.to_numeric(
        df_teams[c],
        errors='coerce'
    )

print("✅ df_teams:", df_teams.shape)

# =========================================================
# 3️⃣ FILTRO DO DIA
# =========================================================

DATA_ALVO = df_teams['Hour'].dt.date.iloc[0]

df_v_teams = (
    df_teams[
        df_teams['Hour'].dt.date == DATA_ALVO
    ]
    .sort_values('Hour')
    .reset_index(drop=True)
)

df_v_teams = df_v_teams.replace({pd.NA: np.nan})

if df_v_teams.empty:

    raise RuntimeError(
        "❌ df_v_teams vazio — erro no filtro"
    )

print("✅ df_v_teams:", df_v_teams.shape)

display(df.head(10))

# ============================================================
# BLOCO 06 — FUNÇÕES GERAIS
# ============================================================

# ============================================================
# MGF REAL
# ============================================================

def calcular_mgf_real(df, team_home, team_away, game_datetime):

    df = df.sort_values("Hour")

    jogos_home = df[
        (df["Home_Team"] == team_home) &
        (df["Hour"] < game_datetime)
    ].tail(1)

    jogos_away = df[
        (df["Visitor_Team"] == team_away) &
        (df["Hour"] < game_datetime)
    ].tail(1)

    if jogos_home.empty or jogos_away.empty:
        return np.nan, np.nan, np.nan, np.nan

    h = jogos_home.iloc[0]
    a = jogos_away.iloc[0]

    return (
        h["MGF_H"],
        h["MGC_H"],
        a["MGF_A"],
        a["MGC_A"]
    )


# ============================================================
# MATRIZ POISSON
# ============================================================

def matriz_poisson(lh, la):

    return np.outer(
        [poisson.pmf(i, lh) for i in range(MAX_GOLS + 1)],
        [poisson.pmf(j, la) for j in range(MAX_GOLS + 1)]
    )


# ============================================================
# SAFE ODDS
# ============================================================

def safe_odds(prob, min_prob=0.01):

    if pd.isna(prob):
        return np.nan

    prob = max(prob, min_prob)

    return round(1 / prob, 2)


# ============================================================
# FULL TIME
# ============================================================

def calcular_probabilidades_ft(exg_home, exg_away, max_goals=8):

    if pd.isna(exg_home) or pd.isna(exg_away):
        return np.nan, np.nan, np.nan

    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0

    for gh in range(max_goals + 1):
        for ga in range(max_goals + 1):

            p = (
                poisson.pmf(gh, exg_home)
                * poisson.pmf(ga, exg_away)
            )

            if gh > ga:
                prob_home += p
            elif gh == ga:
                prob_draw += p
            else:
                prob_away += p

    return prob_home, prob_draw, prob_away


# ============================================================
# OVER / UNDER
# ============================================================

def calcular_over_under(exg_home, exg_away, limite):

    if pd.isna(exg_home) or pd.isna(exg_away):
        return np.nan, np.nan

    lam = exg_home + exg_away

    under = poisson.cdf(int(limite), lam)
    over = 1 - under

    return round(over * 100, 2), round(under * 100, 2)


# ============================================================
# BTTS
# ============================================================

def calcular_btts(exg_home, exg_away):

    if pd.isna(exg_home) or pd.isna(exg_away):
        return np.nan

    p_h0 = poisson.pmf(0, exg_home)
    p_a0 = poisson.pmf(0, exg_away)

    btts = 1 - p_h0 - p_a0 + (p_h0 * p_a0)

    return round(btts * 100, 2)


# ============================================================
# PRIMEIRO GOL
# ============================================================

def calcular_primeiro_gol(
    exg_home,
    exg_away,
    matriz,
    CHM=None,
    CAM=None
):

    if pd.isna(exg_home) or pd.isna(exg_away):
        return np.nan, np.nan

    lam_total = exg_home + exg_away

    if lam_total == 0:
        return np.nan, np.nan

    p_zero = matriz[0, 0] if matriz.size else 0

    base_home = exg_home / lam_total
    base_away = exg_away / lam_total

    ajuste_home = 1
    ajuste_away = 1

    if pd.notna(CHM) and pd.notna(CAM):

        total = CHM + CAM

        if total > 0:
            ajuste_home = CHM / total
            ajuste_away = CAM / total

    home = (
        (0.7 * base_home + 0.3 * ajuste_home)
        * (1 - p_zero)
    )

    away = (
        (0.7 * base_away + 0.3 * ajuste_away)
        * (1 - p_zero)
    )

    return round(home * 100, 2), round(away * 100, 2)


# ============================================================
# INTERPRETAÇÃO
# ============================================================

def interpretar_forca_mix(
    home,
    away,
    prob_home,
    prob_away,
    odd_home,
    odd_away,
    exg_h,
    exg_a,
    vr01,
    coef
):

    frases = []

    edge_prob = prob_home - prob_away
    edge_odds = (
        (1 / odd_home if odd_home else 0)
        - (1 / odd_away if odd_away else 0)
    )
    edge_exg = (exg_h - exg_a) / 3

    score = (
        edge_prob * 0.5
        + edge_odds * 0.3
        + edge_exg * 0.2
    )

    if score > 0.35:
        frases.append(f"Domínio do {home}")
    elif score > 0.15:
        frases.append(f"{home} favorito")
    elif score < -0.35:
        frases.append(f"Domínio do {away}")
    elif score < -0.15:
        frases.append(f"{away} favorito")
    else:
        frases.append("Jogo equilibrado")

    if pd.notna(coef):

        if coef >= 1.95:
            frases.append("Tendência Over 1,5FT")
        elif coef <= 1.70:
            frases.append("Tendência Under")

    if pd.notna(vr01):

        if vr01 >= 0.12:
            frases.append("Gols do Favorito")
        elif vr01 <= -0.10:
            frases.append("BTTS / Jogo Aberto")

    return " | ".join(frases)



# ============================================================
# BLOCO 07 — LOOP PRINCIPAL
# ============================================================

lista_resultados = []

for idx, row in df_v_teams.iterrows():

    Team_Home = row["Home_Team"]
    Team_Away = row["Visitor_Team"]
    game_datetime = row["Hour"]

    try:

        # ----------------------------------------------------
        # MGF REAL
        # ----------------------------------------------------
        MGF_H, MGC_H, MGF_A, MGC_A = calcular_mgf_real(
            df,
            Team_Home,
            Team_Away,
            game_datetime
        )

        if any(pd.isna(x) for x in [MGF_H, MGC_H, MGF_A, MGC_A]):
            continue

        Odd_H = float(row["Odds_Casa"])
        Odd_D = float(row["Odds_Empate"])
        Odd_A = float(row["Odds_Visitante"])

        resultado = {
            "Country": row.get("Country"),
            "League": row.get("League"),
            "Home_Team": Team_Home,
            "Visitor_Team": Team_Away,
            "Hour": game_datetime,
            "Odds_Casa": Odd_H,
            "Odds_Empate": Odd_D,
            "Odds_Visitante": Odd_A,
            "PPJH": row.get("PPJH"),
            "PPJA": row.get("PPJA"),
            "FAH": row.get("FAH"),
            "FAA": row.get("FAA"),
            "FDH": row.get("FDH"),
            "FDA": row.get("FDA"),
            "VR01": row.get("VR01"),
            "COEF_OVER1FT": row.get("COEF_OVER1FT"),
            "MGF_H": MGF_H,
            "MGC_H": MGC_H,
            "MGF_A": MGF_A,
            "MGC_A": MGC_A,
        }

        # =====================================================
        # BLOCO 07.2 — POISSON MGF
        # =====================================================

        ExG_Home_MGF = np.sqrt(MGF_H * MGC_A)
        ExG_Away_MGF = np.sqrt(MGF_A * MGC_H)

        ExG_Home_MGF = min(ExG_Home_MGF, 5)
        ExG_Away_MGF = min(ExG_Away_MGF, 5)

        resultado["ExG_Home_MGF"] = round(ExG_Home_MGF, 3)
        resultado["ExG_Away_MGF"] = round(ExG_Away_MGF, 3)

        matriz_mgf = matriz_poisson(
            ExG_Home_MGF,
            ExG_Away_MGF
        )

        pH, pD, pA = calcular_probabilidades_ft(
            ExG_Home_MGF,
            ExG_Away_MGF,
            MAX_GOLS
        )

        resultado["Odd_Justa_Home_MGF"] = safe_odds(pH)
        resultado["Odd_Justa_Draw_MGF"] = safe_odds(pD)
        resultado["Odd_Justa_Away_MGF"] = safe_odds(pA)

        # -----------------------------
        # Over / Under
        # -----------------------------

        for linha in [0.5, 1.5, 2.5, 3.5, 4.5]:

            over, under = calcular_over_under(
                ExG_Home_MGF,
                ExG_Away_MGF,
                linha
            )

            resultado[f"MGF_Over_{linha}"] = over
            resultado[f"MGF_Under_{linha}"] = under

        # -----------------------------
        # BTTS
        # -----------------------------

        resultado["MGF_BTTS"] = calcular_btts(
            ExG_Home_MGF,
            ExG_Away_MGF
        )

        # -----------------------------
        # Clean Sheet
        # -----------------------------

        resultado["MGF_CleanSheet_Home"] = round(
            poisson.pmf(0, ExG_Away_MGF) * 100,
            2
        )

        resultado["MGF_CleanSheet_Away"] = round(
            poisson.pmf(0, ExG_Home_MGF) * 100,
            2
        )

        # -----------------------------
        # Primeiro Gol
        # -----------------------------

        pg_home, pg_away = calcular_primeiro_gol(
            ExG_Home_MGF,
            ExG_Away_MGF,
            matriz_mgf,
            row.get("CHM"),
            row.get("CAM")
        )

        resultado["MGF_PrimeiroGol_Home"] = pg_home
        resultado["MGF_PrimeiroGol_Away"] = pg_away
        

        # =====================================================
        # BLOCO 07.3 — POISSON ATK × DEF
        # =====================================================

        ExG_Home_ATKDEF = (MGF_H + MGC_A) / 2
        ExG_Away_ATKDEF = (MGF_A + MGC_H) / 2

        ExG_Home_ATKDEF = min(ExG_Home_ATKDEF, 5)
        ExG_Away_ATKDEF = min(ExG_Away_ATKDEF, 5)

        resultado["ExG_Home_ATKDEF"] = round(ExG_Home_ATKDEF, 3)
        resultado["ExG_Away_ATKDEF"] = round(ExG_Away_ATKDEF, 3)

        matriz_atkdef = matriz_poisson(
            ExG_Home_ATKDEF,
            ExG_Away_ATKDEF
        )

        pH, pD, pA = calcular_probabilidades_ft(
            ExG_Home_ATKDEF,
            ExG_Away_ATKDEF,
            MAX_GOLS
        )

        resultado["Odd_Justa_Home_ATKDEF"] = safe_odds(pH)
        resultado["Odd_Justa_Draw_ATKDEF"] = safe_odds(pD)
        resultado["Odd_Justa_Away_ATKDEF"] = safe_odds(pA)

        # -----------------------------
        # Over / Under
        # -----------------------------

        for linha in [0.5, 1.5, 2.5, 3.5, 4.5]:

            over, under = calcular_over_under(
                ExG_Home_ATKDEF,
                ExG_Away_ATKDEF,
                linha
            )

            resultado[f"ATKDEF_Over_{linha}"] = over
            resultado[f"ATKDEF_Under_{linha}"] = under

        # -----------------------------
        # BTTS
        # -----------------------------

        resultado["ATKDEF_BTTS"] = calcular_btts(
            ExG_Home_ATKDEF,
            ExG_Away_ATKDEF
        )

        # -----------------------------
        # Clean Sheet
        # -----------------------------

        resultado["ATKDEF_CleanSheet_Home"] = round(
            poisson.pmf(0, ExG_Away_ATKDEF) * 100,
            2
        )

        resultado["ATKDEF_CleanSheet_Away"] = round(
            poisson.pmf(0, ExG_Home_ATKDEF) * 100,
            2
        )

        # -----------------------------
        # Primeiro Gol
        # -----------------------------

        pg_home, pg_away = calcular_primeiro_gol(
            ExG_Home_ATKDEF,
            ExG_Away_ATKDEF,
            matriz_atkdef,
            row.get("CHM"),
            row.get("CAM")
        )

        resultado["ATKDEF_PrimeiroGol_Home"] = pg_home
        resultado["ATKDEF_PrimeiroGol_Away"] = pg_away

        
        # =====================================================
        # BLOCO 07.4 — POISSON VG
        # =====================================================

        ExG_Home_VG = row.get("Media_VG_H")
        ExG_Away_VG = row.get("Media_VG_A")

        if pd.isna(ExG_Home_VG) or pd.isna(ExG_Away_VG):

            resultado["ExG_Home_VG"] = np.nan
            resultado["ExG_Away_VG"] = np.nan

            resultado["Odd_Justa_Home_VG"] = np.nan
            resultado["Odd_Justa_Draw_VG"] = np.nan
            resultado["Odd_Justa_Away_VG"] = np.nan

            for linha in [0.5, 1.5, 2.5, 3.5, 4.5]:

                resultado[f"VG_Over_{linha}"] = np.nan
                resultado[f"VG_Under_{linha}"] = np.nan

            resultado["VG_BTTS"] = np.nan
            resultado["VG_CleanSheet_Home"] = np.nan
            resultado["VG_CleanSheet_Away"] = np.nan
            resultado["VG_PrimeiroGol_Home"] = np.nan
            resultado["VG_PrimeiroGol_Away"] = np.nan

        else:

            ExG_Home_VG = min(float(ExG_Home_VG), 5)
            ExG_Away_VG = min(float(ExG_Away_VG), 5)

            resultado["ExG_Home_VG"] = round(ExG_Home_VG, 3)
            resultado["ExG_Away_VG"] = round(ExG_Away_VG, 3)

            matriz_vg = matriz_poisson(
                ExG_Home_VG,
                ExG_Away_VG
            )

            pH, pD, pA = calcular_probabilidades_ft(
                ExG_Home_VG,
                ExG_Away_VG,
                MAX_GOLS
            )

            resultado["Odd_Justa_Home_VG"] = safe_odds(pH)
            resultado["Odd_Justa_Draw_VG"] = safe_odds(pD)
            resultado["Odd_Justa_Away_VG"] = safe_odds(pA)

            # -----------------------------
            # Over / Under
            # -----------------------------

            for linha in [0.5, 1.5, 2.5, 3.5, 4.5]:

                over, under = calcular_over_under(
                    ExG_Home_VG,
                    ExG_Away_VG,
                    linha
                )

                resultado[f"VG_Over_{linha}"] = over
                resultado[f"VG_Under_{linha}"] = under

            # -----------------------------
            # BTTS
            # -----------------------------

            resultado["VG_BTTS"] = calcular_btts(
                ExG_Home_VG,
                ExG_Away_VG
            )

            # -----------------------------
            # Clean Sheet
            # -----------------------------

            resultado["VG_CleanSheet_Home"] = round(
                poisson.pmf(0, ExG_Away_VG) * 100,
                2
            )

            resultado["VG_CleanSheet_Away"] = round(
                poisson.pmf(0, ExG_Home_VG) * 100,
                2
            )

            # -----------------------------
            # Primeiro Gol
            # -----------------------------

            pg_home, pg_away = calcular_primeiro_gol(
                ExG_Home_VG,
                ExG_Away_VG,
                matriz_vg,
                row.get("CHM"),
                row.get("CAM")
            )

            resultado["VG_PrimeiroGol_Home"] = pg_home
            resultado["VG_PrimeiroGol_Away"] = pg_away
            

        # =====================================================
        # BLOCO 07.5 — POISSON CONSENSO
        # =====================================================

        ExG_Home_Consenso = np.nanmean([
            resultado["ExG_Home_MGF"],
            resultado["ExG_Home_ATKDEF"],
            resultado["ExG_Home_VG"]
        ])

        ExG_Away_Consenso = np.nanmean([
            resultado["ExG_Away_MGF"],
            resultado["ExG_Away_ATKDEF"],
            resultado["ExG_Away_VG"]
        ])

        ExG_Home_Consenso = min(ExG_Home_Consenso, 5)
        ExG_Away_Consenso = min(ExG_Away_Consenso, 5)

        resultado["ExG_Home_Consenso"] = round(
            ExG_Home_Consenso,
            3
        )

        resultado["ExG_Away_Consenso"] = round(
            ExG_Away_Consenso,
            3
        )

        matriz_consenso = matriz_poisson(
            ExG_Home_Consenso,
            ExG_Away_Consenso
        )

        pH, pD, pA = calcular_probabilidades_ft(
            ExG_Home_Consenso,
            ExG_Away_Consenso,
            MAX_GOLS
        )

        resultado["Odd_Justa_Home"] = safe_odds(pH)
        resultado["Odd_Justa_Draw"] = safe_odds(pD)
        resultado["Odd_Justa_Away"] = safe_odds(pA)

        # ----------------------------------
        # OVER / UNDER
        # ----------------------------------

        for linha in [0.5, 1.5, 2.5, 3.5, 4.5]:

            over, under = calcular_over_under(
                ExG_Home_Consenso,
                ExG_Away_Consenso,
                linha
            )

            resultado[f"Over_{linha}"] = over
            resultado[f"Under_{linha}"] = under

        # ----------------------------------
        # BTTS
        # ----------------------------------

        resultado["BTTS_%"] = calcular_btts(
            ExG_Home_Consenso,
            ExG_Away_Consenso
        )

        # ----------------------------------
        # CLEAN SHEET
        # ----------------------------------

        resultado["Clean_Sheet_Home"] = round(
            poisson.pmf(0, ExG_Away_Consenso) * 100,
            2
        )

        resultado["Clean_Sheet_Away"] = round(
            poisson.pmf(0, ExG_Home_Consenso) * 100,
            2
        )

        # ----------------------------------
        # PRIMEIRO GOL
        # ----------------------------------

        pg_home, pg_away = calcular_primeiro_gol(
            ExG_Home_Consenso,
            ExG_Away_Consenso,
            matriz_consenso,
            row.get("CHM"),
            row.get("CAM")
        )

        resultado["PrimeiroGol_Home"] = pg_home
        resultado["PrimeiroGol_Away"] = pg_away

        # ----------------------------------
        # VALUE
        # ----------------------------------

        resultado["Value_Home"] = round(
            (Odd_H / resultado["Odd_Justa_Home"]) - 1,
            4
        )

        resultado["Value_Draw"] = round(
            (Odd_D / resultado["Odd_Justa_Draw"]) - 1,
            4
        )

        resultado["Value_Away"] = round(
            (Odd_A / resultado["Odd_Justa_Away"]) - 1,
            4
        )

        # ----------------------------------
        # EDGE (%)
        # ----------------------------------

        resultado["Edge_Home"] = round(
            (
                (1 / resultado["Odd_Justa_Home"])
                - (1 / Odd_H)
            ) * 100,
            2
        )

        resultado["Edge_Draw"] = round(
            (
                (1 / resultado["Odd_Justa_Draw"])
                - (1 / Odd_D)
            ) * 100,
            2
        )

        resultado["Edge_Away"] = round(
            (
                (1 / resultado["Odd_Justa_Away"])
                - (1 / Odd_A)
            ) * 100,
            2
        )

        # ----------------------------------
        # INTERPRETAÇÃO
        # ----------------------------------

        resultado["Interpretacao"] = interpretar_forca_mix(

            Team_Home,
            Team_Away,

            pH,
            pA,

            Odd_H,
            Odd_A,

            ExG_Home_Consenso,
            ExG_Away_Consenso,

            resultado["VR01"],
            resultado["COEF_OVER1FT"]

        )

        # ----------------------------------
        # SALVAR LINHA
        # ----------------------------------

        lista_resultados.append(resultado)

    except Exception as erro:

        print(
            f"Erro na linha {idx} - "
            f"{Team_Home} x {Team_Away}: {erro}"
        )

        continue

# ============================================================
# BLOCO 07.6 — FINALIZAÇÃO
# ============================================================

BASE_ML = pd.DataFrame(lista_resultados)

BASE_ML.replace([np.inf, -np.inf], np.nan, inplace=True)

for coluna in BASE_ML.columns:
    if pd.api.types.is_numeric_dtype(BASE_ML[coluna]):
        if not BASE_ML[coluna].isna().all():
            BASE_ML[coluna] = BASE_ML[coluna].fillna(BASE_ML[coluna].median())
    else:
        BASE_ML[coluna] = BASE_ML[coluna].fillna("")

BASE_ML = BASE_ML.drop_duplicates().reset_index(drop=True)

BASE_ML.to_csv(
    "BASE_ML.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("BASE_ML.csv exportada com sucesso.")
