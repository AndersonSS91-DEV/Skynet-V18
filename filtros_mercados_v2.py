# Módulo standalone — import filtros_mercados_v2 as fm  (coloque este arquivo na mesma pasta do app.py)

# -*- coding: utf-8 -*-
"""
Filtros de mercados minerados na base histórica (CSV_LIMPO.csv), restrita a
times do ranking600 jogando em CASA (19.683 jogos, ranking600 = 691 times).

v2 — CORRIGE bug de colisão de nomes: várias colunas (FAH, FAA, FDH, FDA,
Clean_Games_A etc.) existem TANTO no CSV_LIMPO quanto nas planilhas Poisson
já usadas pela Aba IA, com semânticas diferentes. Por isso, aqui TODA coluna
vinda do CSV_LIMPO é lida com o sufixo "__CL" (ex: "FDH__CL"), nunca pelo nome
puro — elimina qualquer ambiguidade, mesmo que o df_clean já tenha uma coluna
"FDH" vinda de outro lugar.
"""

def num(x):
    """Converte string com vírgula decimal (ou já numérico) pra float. None se não der."""
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
            if x == "" or x.lower() == "nan":
                return None
        v = float(x)
        return v
    except (ValueError, TypeError):
        return None


def filtro_over_0_5ht(row):
    """
    Mercado: Over 0,5HT
    Backtest (ranking600, jogando em casa): 69.1% -> 82.4% | n=1309
    """
    try:
        v0 = num(row.get("Odd_Empate_HT__CL"))
        v1 = num(row.get("GF_0-15_Away__CL"))
        v2 = num(row.get("HA_0,5+HT_A__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.54)
            and
            (v1 <= 0.2)
            and
            (v2 <= 60.0)
            and
            (1.06 <= v3 <= 5.16)
            and
            (2.33 <= v4 <= 20.89)
        )
    except Exception:
        return False


def filtro_over_1_5ht(row):
    """
    Mercado: Over 1,5HT
    Backtest (ranking600, jogando em casa): 34.2% -> 57.5% | n=440
    """
    try:
        v0 = num(row.get("Odd_Empate_HT__CL"))
        v1 = num(row.get("Odd_Over_0,5HT__CL"))
        v2 = num(row.get("MCGS_HT_H__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.54)
            and
            (v1 <= 1.17)
            and
            (v2 >= 2.6)
            and
            (1.03 <= v3 <= 5.35)
            and
            (1.78 <= v4 <= 31.85)
        )
    except Exception:
        return False


def filtro_under_1_5ht(row):
    """
    Mercado: Under 1,5HT
    Backtest (ranking600, jogando em casa): 65.8% -> 83.8% | n=712
    """
    try:
        v0 = num(row.get("Odd_Over_0,5HT__CL"))
        v1 = num(row.get("Odd_Empate_HT__CL"))
        v2 = num(row.get("Odd_Visitor_HT__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 1.48)
            and
            (v1 <= 1.92)
            and
            (v2 >= 4.07)
            and
            (1.7 <= v3 <= 2.52)
            and
            (3.23 <= v4 <= 5.39)
        )
    except Exception:
        return False


def filtro_over_1_5ft(row):
    """
    Mercado: Over 1,5FT
    Backtest (ranking600, jogando em casa): 74.0% -> 90.2% | n=691
    """
    try:
        v0 = num(row.get("Odds_Under_2,5FT__CL"))
        v1 = num(row.get("Média_2,5FT_Global__CL"))
        v2 = num(row.get("Odds_Under_1,5FT__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.46)
            and
            (v1 >= 70.0)
            and
            (v2 >= 5.81)
            and
            (1.05 <= v3 <= 8.56)
            and
            (1.43 <= v4 <= 21.05)
        )
    except Exception:
        return False


def filtro_over_2_5ft(row):
    """
    Mercado: Over 2,5FT
    Backtest (ranking600, jogando em casa): 50.9% -> 80.4% | n=352
    """
    try:
        v0 = num(row.get("Odds_Under_2,5FT__CL"))
        v1 = num(row.get("Odds_Over_2,5FT__CL"))
        v2 = num(row.get("GF_76-90_Home__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.46)
            and
            (v1 <= 1.28)
            and
            (v2 <= 0.8)
            and
            (1.02 <= v3 <= 8.79)
            and
            (1.42 <= v4 <= 40.96)
        )
    except Exception:
        return False


def filtro_under_2_5ft(row):
    """
    Mercado: Under 2,5FT
    Backtest (ranking600, jogando em casa): 49.1% -> 74.0% | n=430
    """
    try:
        v0 = num(row.get("Odds_Over_2,5FT__CL"))
        v1 = num(row.get("Odds_Under_1,5FT__CL"))
        v2 = num(row.get("FAA__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.31)
            and
            (v1 <= 2.26)
            and
            (v2 <= 27.0)
            and
            (1.67 <= v3 <= 3.79)
            and
            (2.35 <= v4 <= 5.9)
        )
    except Exception:
        return False


def filtro_btts_sim(row):
    """
    Mercado: BTTS Sim
    Backtest (ranking600, jogando em casa): 52.2% -> 73.5% | n=313
    """
    try:
        v0 = num(row.get("Odd_BTTS_YES__CL"))
        v1 = num(row.get("Odds_Under_1,5FT__CL"))
        v2 = num(row.get("MG_Global__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 <= 1.56)
            and
            (v1 >= 5.7)
            and
            (v2 >= 3.7)
            and
            (1.19 <= v3 <= 5.49)
            and
            (1.55 <= v4 <= 8.36)
        )
    except Exception:
        return False


def filtro_over_3_0ft_asia(row):
    """
    Mercado: Over 3,0FT (Asiático)
    Backtest (ranking600, jogando em casa): 37.5% -> 75.8% | n=310
    Linha asiática 3.0: empate técnico (total=3) é push/anulado, não entra no backtest.
    """
    try:
        v0 = num(row.get("Odds_Under_2,5FT__CL"))
        v1 = num(row.get("Odds_Over_2,5FT__CL"))
        v2 = num(row.get("FDA__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 2.46)
            and
            (v1 <= 1.28)
            and
            (v2 <= 63.0)
            and
            (1.02 <= v3 <= 5.96)
            and
            (1.82 <= v4 <= 42.27)
        )
    except Exception:
        return False


def filtro_lay_goleada_away(row):
    """
    Mercado: Lay Goleada Away
    Backtest (ranking600, jogando em casa): 96.6% -> 99.9% | n=1312
    Lay = aposta CONTRA o evento. "Sim" aqui significa: o jogo tende a NÃO ter goleada do visitante.
    """
    try:
        v0 = num(row.get("CS 0X1__CL"))
        v1 = num(row.get("CS 1X0__CL"))
        v2 = num(row.get("Eficiência_H__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 16.3)
            and
            (v1 <= 8.5)
            and
            (v2 <= 87.0)
            and
            (1.1 <= v3 <= 1.46)
            and
            (6.18 <= v4 <= 18.92)
        )
    except Exception:
        return False


def filtro_lay_empate(row):
    """
    Mercado: Lay Empate
    Backtest (ranking600, jogando em casa): 72.3% -> 91.9% | n=247
    Lay = aposta CONTRA o evento. "Sim" aqui significa: o jogo tende a NÃO terminar empatado.
    """
    try:
        v0 = num(row.get("CS 0X1__CL"))
        v1 = num(row.get("CS 0X0__CL"))
        v2 = num(row.get("Scored_Times_A__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 16.3)
            and
            (v1 >= 25.0)
            and
            (v2 <= 20.0)
            and
            (1.02 <= v3 <= 1.82)
            and
            (4.21 <= v4 <= 42.86)
        )
    except Exception:
        return False


def filtro_lay_away(row):
    """
    Mercado: Lay Away
    Backtest (ranking600, jogando em casa): 77.3% -> 94.9% | n=1480
    Lay = aposta CONTRA o evento. "Sim" aqui significa: o visitante tende a NÃO vencer.
    """
    try:
        v0 = num(row.get("CS 0X1__CL"))
        v1 = num(row.get("Eficiência_H__CL"))
        v2 = num(row.get("PPJA__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 16.3)
            and
            (v1 >= 57.25)
            and
            (v2 <= 1.4)
            and
            (1.06 <= v3 <= 1.62)
            and
            (5.01 <= v4 <= 22.25)
        )
    except Exception:
        return False


def filtro_lay_0x0(row):
    """
    Mercado: Lay 0x0
    Backtest (ranking600, jogando em casa): 90.7% -> 99.6% | n=278
    Lay = aposta CONTRA o evento. "Sim" aqui significa: o jogo tende a NÃO terminar 0x0.
    """
    try:
        v0 = num(row.get("CS 0X1__CL"))
        v1 = num(row.get("CS 0X0__CL"))
        v2 = num(row.get("Eficiência_H__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 16.3)
            and
            (v1 >= 25.0)
            and
            (v2 >= 67.0)
            and
            (1.02 <= v3 <= 1.65)
            and
            (4.23 <= v4 <= 41.19)
        )
    except Exception:
        return False


def filtro_lay_0x1(row):
    """
    Mercado: Lay 0x1
    Backtest (ranking600, jogando em casa): 94.2% -> 99.6% | n=760
    Lay = aposta CONTRA o evento. "Sim" aqui significa: o jogo tende a NÃO terminar 0x1.
    """
    try:
        v0 = num(row.get("CS 0X1__CL"))
        v1 = num(row.get("Clean_Games_A__CL"))
        v2 = num(row.get("Los4_A__CL"))
        v3 = num(row.get("Odds_Casa__CL"))
        v4 = num(row.get("Odds_Visitante__CL"))
        if any(v is None for v in [v0, v1, v2, v3, v4]):
            return False
        return (
            (v0 >= 16.3)
            and
            (v1 <= 0.0)
            and
            (v2 <= 20.0)
            and
            (1.08 <= v3 <= 1.79)
            and
            (4.25 <= v4 <= 19.87)
        )
    except Exception:
        return False


# =========================================
# 📋 COLUNAS NECESSÁRIAS (nomes ORIGINAIS no CSV_LIMPO / df_base)
# Use isso no merge — sempre renomeando com o sufixo __CL
# =========================================
COLUNAS_NECESSARIAS = [
    "CS 0X0",
    "CS 0X1",
    "CS 1X0",
    "Clean_Games_A",
    "Eficiência_H",
    "FAA",
    "FDA",
    "GF_0-15_Away",
    "GF_76-90_Home",
    "HA_0,5+HT_A",
    "Los4_A",
    "MCGS_HT_H",
    "MG_Global",
    "Média_2,5FT_Global",
    "Odd_BTTS_YES",
    "Odd_Empate_HT",
    "Odd_Over_0,5HT",
    "Odd_Visitor_HT",
    "Odds_Casa",
    "Odds_Over_2,5FT",
    "Odds_Under_1,5FT",
    "Odds_Under_2,5FT",
    "Odds_Visitante",
    "PPJA",
    "Scored_Times_A",
]

# =========================================
# 🏷️ SINAIS AGREGADOS
# =========================================
def montar_sinais(row):
    """
    Roda os 13 filtros e devolve uma string com os sinais que bateram, tipo:
    "Over 2,5FT | BTTS | Lay Empate"

    Regra de empilhamento:
      - Grupo "Over Gols FT" (Over 1,5FT / Over 2,5FT / Over 3,0FT asiático):
        mostra só o MAIOR nível batido (3,0 > 2,5 > 1,5) — não empilha.
      - Grupo "Over Gols HT" (Over 0,5HT / Over 1,5HT):
        mostra só o MAIOR nível batido (1,5 > 0,5) — não empilha.
      - Os outros 8 (BTTS, os 4 Lays, Under 2,5FT, Under 1,5HT) são
        independentes e podem aparecer juntos.
    """
    sinais = []

    if filtro_over_3_0ft_asia(row):
        sinais.append("Over 3,0FT (Asiático)")
    elif filtro_over_2_5ft(row):
        sinais.append("Over 2,5FT")
    elif filtro_over_1_5ft(row):
        sinais.append("Over 1,5FT")

    if filtro_over_1_5ht(row):
        sinais.append("Over 1,5HT")
    elif filtro_over_0_5ht(row):
        sinais.append("Over 0,5HT")

    if filtro_btts_sim(row):
        sinais.append("BTTS")
    if filtro_under_2_5ft(row):
        sinais.append("Under 2,5FT")
    if filtro_under_1_5ht(row):
        sinais.append("Under 1,5HT")
    if filtro_lay_goleada_away(row):
        sinais.append("Lay Goleada Away")
    if filtro_lay_empate(row):
        sinais.append("Lay Empate")
    if filtro_lay_away(row):
        sinais.append("Lay Away")
    if filtro_lay_0x0(row):
        sinais.append("Lay 0x0")
    if filtro_lay_0x1(row):
        sinais.append("Lay 0x1")

    return " | ".join(sinais) if sinais else "-"
