# ============================================================
# card_instagram_engine.py  (v2 — visual sóbrio, ícones vetoriais)
# Motor que desenha o card do Instagram como uma imagem PNG real
# (Pillow). Sem emoji colorido "estilo clip-art" — os ícones são
# desenhados vetorialmente na cor de acento de cada seção, pra dar
# uma cara mais editorial/discreta (como o modelo de referência).
#
# Coloque este arquivo na mesma pasta do seu app Streamlit.
# Ainda precisa da pasta fonts/ ao lado (DejaVuSans.ttf e
# DejaVuSans-Bold.ttf) porque o servidor de produção não tem essas
# fontes do sistema. NÃO precisa mais do NotoColorEmoji.ttf.
#
#   seu_projeto/
#     app.py
#     card_instagram_engine.py
#     fonts/
#       DejaVuSans.ttf
#       DejaVuSans-Bold.ttf
# ============================================================
import io
import os
import re
import base64
import functools
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ------------------------------------------------------------------
# Fontes — carregamento resiliente (funciona local e em produção)
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_REGULAR_CANDIDATES = [
    os.path.join(_BASE_DIR, "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_BOLD_CANDIDATES = [
    os.path.join(_BASE_DIR, "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
try:
    import matplotlib
    _mpl_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    _REGULAR_CANDIDATES.append(os.path.join(_mpl_dir, "DejaVuSans.ttf"))
    _BOLD_CANDIDATES.append(os.path.join(_mpl_dir, "DejaVuSans-Bold.ttf"))
except Exception:
    pass


@functools.lru_cache(maxsize=None)
def F(weight, size):
    candidates = _BOLD_CANDIDATES if weight == "bold" else _REGULAR_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ------------------------------------------------------------------
# Paleta — contida, poucas cores com significado fixo
# ------------------------------------------------------------------
BG            = (5, 8, 16)
CARD_BG       = (3, 5, 10)
BORDER        = (47, 111, 237)
INNER_BG      = (8, 11, 18)
INNER_BORDER  = (26, 33, 48)
BRAND         = (127, 179, 255)
WHITE         = (240, 244, 250)
GREY          = (135, 144, 160)
LGREY         = (196, 204, 216)
GREEN         = (74, 222, 128)
RED           = (248, 113, 113)
BLUE          = (74, 157, 255)
ORANGE        = (250, 204, 21)
PURPLE        = (167, 139, 250)

MARKET_PALETTE = [RED, ORANGE, BLUE, PURPLE]


def hexcol(c):
    if isinstance(c, str) and c.startswith("#"):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return c


# ------------------------------------------------------------------
# Helpers de texto/forma
# ------------------------------------------------------------------
def text_w(d, s, fnt):
    b = d.textbbox((0, 0), s, font=fnt)
    return b[2] - b[0]


def ctext(d, cx, cy, s, fnt, fill, anchor="mm"):
    d.text((cx, cy), s, font=fnt, fill=fill, anchor=anchor)


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def wrap_lines(d, text, fnt, max_w, max_lines=2):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_w(d, trial, fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def section_title(d, cx, y, icon_kind, text, color=WHITE, icon_color=None, fnt_size=20):
    """Título de seção: ícone vetorial pequeno + texto, centralizados juntos."""
    icon_color = icon_color or color
    fnt = F("bold", fnt_size)
    tw = text_w(d, text, fnt)
    has_icon = icon_kind is not None
    icon_s = fnt_size
    gap = 9 if has_icon else 0
    total_w = (icon_s + gap if has_icon else 0) + tw
    x0 = cx - total_w / 2
    if has_icon:
        icon(d, icon_kind, x0 + icon_s / 2, y, icon_s, icon_color)
        d.text((x0 + icon_s + gap, y), text, font=fnt, fill=color, anchor="lm")
    else:
        d.text((x0, y), text, font=fnt, fill=color, anchor="lm")


def load_crest(source):
    """Aceita: PIL.Image, bytes, string HTML com <img src="..."> (URL ou
    base64), ou None. Se não conseguir carregar, retorna None (cai no
    fallback de iniciais do time)."""
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    if isinstance(source, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(source)).convert("RGBA")
        except Exception:
            return None
    if isinstance(source, str):
        m = re.search(r'src="data:image/[^;]+;base64,([^"]+)"', source)
        if m:
            try:
                raw = base64.b64decode(m.group(1))
                return Image.open(io.BytesIO(raw)).convert("RGBA")
            except Exception:
                return None
        m = re.search(r'src="(https?://[^"]+)"', source)
        if m:
            try:
                import urllib.request
                with urllib.request.urlopen(m.group(1), timeout=4) as r:
                    return Image.open(io.BytesIO(r.read())).convert("RGBA")
            except Exception:
                return None
    return None


def paste_crest(base_img, crest_img, cx, cy, r, fallback_text, fallback_color):
    d = ImageDraw.Draw(base_img)
    if crest_img is not None:
        size = r * 2
        im = ImageOps.fit(crest_img, (size, size), Image.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
        base_img.paste(im, (int(cx - r), int(cy - r)), mask)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=INNER_BORDER, width=2)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(17, 22, 31),
                  outline=fallback_color, width=3)
        ctext(d, cx, cy, fallback_text, F("bold", int(r * 0.62)), fallback_color)


# ------------------------------------------------------------------
# Ícones vetoriais (desenhados na hora, sem emoji/bitmap)
# ------------------------------------------------------------------
def icon(d, kind, cx, cy, s, color):
    r = s / 2
    w = max(2, round(s / 11))
    if kind == "target":
        for rr, on in [(r, True), (r * 0.62, True), (r * 0.26, False)]:
            if on:
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=color, width=w)
        d.ellipse([cx - r * 0.14, cy - r * 0.14, cx + r * 0.14, cy + r * 0.14], fill=color)
    elif kind == "bars":
        bw = s * 0.22
        heights = [0.45, 0.75, 1.0]
        gap = s * 0.12
        total = bw * 3 + gap * 2
        x0 = cx - total / 2
        base = cy + r * 0.7
        for i, h in enumerate(heights):
            bh = s * 0.9 * h
            x = x0 + i * (bw + gap)
            d.rounded_rectangle([x, base - bh, x + bw, base], radius=bw * 0.25, fill=color)
    elif kind == "trend":
        pts = [(cx - r, cy + r * 0.5), (cx - r * 0.2, cy - r * 0.1), (cx + r * 0.3, cy + r * 0.25), (cx + r, cy - r * 0.7)]
        d.line(pts, fill=color, width=w, joint="curve")
        ax, ay = pts[-1]
        d.polygon([(ax, ay), (ax - r * 0.35, ay), (ax, ay + r * 0.35)], fill=color)
    elif kind == "shield":
        pts = [(cx, cy - r), (cx + r * 0.85, cy - r * 0.55), (cx + r * 0.85, cy + r * 0.15),
               (cx, cy + r), (cx - r * 0.85, cy + r * 0.15), (cx - r * 0.85, cy - r * 0.55)]
        d.polygon(pts, outline=color, width=w)
    elif kind == "person":
        d.ellipse([cx - r * 0.38, cy - r, cx + r * 0.38, cy - r * 0.28], outline=color, width=w)
        body = [cx - r * 0.75, cy - r * 0.05, cx + r * 0.75, cy + r]
        d.pieslice(body, 180, 360, outline=color, width=w)
    elif kind == "speech":
        rrect(d, [cx - r, cy - r * 0.75, cx + r, cy + r * 0.55], r * 0.4, outline=color, width=w)
        d.polygon([(cx - r * 0.35, cy + r * 0.5), (cx - r * 0.05, cy + r * 0.5), (cx - r * 0.3, cy + r * 1.05)], fill=color)
    elif kind == "warning":
        d.polygon([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)], outline=color, width=w)
        d.line([(cx, cy - r * 0.15), (cx, cy + r * 0.28)], fill=color, width=w)
        d.ellipse([cx - w * 0.6, cy + r * 0.5, cx + w * 0.6, cy + r * 0.5 + w * 1.2], fill=color)
    elif kind == "house":
        d.polygon([(cx, cy - r), (cx + r * 0.95, cy - r * 0.05), (cx - r * 0.95, cy - r * 0.05)], outline=color, width=w)
        d.rectangle([cx - r * 0.65, cy - r * 0.1, cx + r * 0.65, cy + r * 0.85], outline=color, width=w)
    elif kind == "plane":
        d.polygon([(cx + r, cy), (cx - r * 0.5, cy - r * 0.55), (cx - r * 0.15, cy - r * 0.05),
                   (cx - r, cy - r * 0.25), (cx - r, cy + r * 0.25), (cx - r * 0.15, cy + r * 0.05),
                   (cx - r * 0.5, cy + r * 0.55)], outline=color, width=max(1, w - 1))
    elif kind == "goal":
        d.rectangle([cx - r, cy - r * 0.75, cx + r, cy + r * 0.75], outline=color, width=w)
        for i in range(1, 4):
            gx = cx - r + i * (2 * r / 4)
            d.line([(gx, cy - r * 0.75), (gx, cy + r * 0.75)], fill=color, width=1)
        for i in range(1, 3):
            gy = cy - r * 0.75 + i * (1.5 * r / 3)
            d.line([(cx - r, gy), (cx + r, gy)], fill=color, width=1)
    elif kind == "boot":
        # silhueta simples de chuteira (perfil), vista da esquerda pra direita
        pts = [
            (cx - r * 0.95, cy + r * 0.15),   # ponta traseira (calcanhar) em cima
            (cx - r * 0.55, cy - r * 0.55),   # sobe pro tornozelo
            (cx + r * 0.05, cy - r * 0.55),   # topo do cano
            (cx + r * 0.05, cy - r * 0.05),   # desce
            (cx + r * 0.6, cy + r * 0.1),     # bico da chuteira
            (cx + r * 0.95, cy + r * 0.42),   # ponta do bico
            (cx + r * 0.55, cy + r * 0.55),   # volta pela sola
            (cx - r * 0.95, cy + r * 0.55),   # sola até o calcanhar
        ]
        d.line(pts + [pts[0]], fill=color, width=max(2, w - 1), joint="curve")
        # cravos na sola
        for i in range(3):
            px = cx - r * 0.55 + i * r * 0.45
            d.line([(px, cy + r * 0.55), (px, cy + r * 0.7)], fill=color, width=max(1, w - 1))
        # bola ao lado
        bcx, bcy, br = cx + r * 1.15, cy + r * 0.25, r * 0.3
        d.ellipse([bcx - br, bcy - br, bcx + br, bcy + br], outline=color, width=max(1, w - 1))
    elif kind == "star":
        pts = []
        for i in range(10):
            ang = math.pi / 2 + i * math.pi / 5
            rad = r if i % 2 == 0 else r * 0.42
            pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
        d.polygon(pts, fill=color)
    elif kind == "heart":
        d.pieslice([cx - r, cy - r * 0.7, cx, cy + r * 0.1], 180, 360, fill=color)
        d.pieslice([cx, cy - r * 0.7, cx + r, cy + r * 0.1], 180, 360, fill=color)
        d.polygon([(cx - r, cy - r * 0.15), (cx + r, cy - r * 0.15), (cx, cy + r)], fill=color)


def icon_for_market(kind):
    k = (kind or "").upper()
    if "GOAL" in k or "0X0" in k or "GH" in k or "GA" in k:
        return "goal"
    return "boot"


# ------------------------------------------------------------------
# Função principal
# ------------------------------------------------------------------
def gerar_card_instagram(
    home, away,
    crest_home=None, crest_away=None,
    odd_casa="-", odd_empate="-", odd_fora="-",
    odd_justa_casa=0.0, odd_justa_empate=0.0, odd_justa_fora=0.0,
    top5=None,                      # list[(gols_home, gols_away, "xx.xx%")]
    over_under=None,                # dict {"0.5": (over_pct, under_pct), ...}
    top4_mercados=None,             # list[(label, valor_0_100, cor_hex, icon_kind, prob_str)]
    titulo_mercados="TOP 4 MERCADOS",
    sinal_texto=None, sinal_cor=None, sinal_estrelas=None, sinal_subtexto=None,
    mostrar_live=False,
    indice_tatico=50, perfil_label="EQUILIBRADO", confianca="MÉDIA",
    brand="@laratodata",
    width=940,
):
    """Retorna os bytes de um PNG pronto para st.image / download_button."""
    top5 = top5 or []
    over_under = over_under or {}
    top4_mercados = top4_mercados or []

    W = width
    pad_out = 36
    pad_in = 30
    cx = W // 2

    H_GUESS = 2400
    img = Image.new("RGB", (W, H_GUESS), BG)
    d = ImageDraw.Draw(img)

    CARD = [pad_out, pad_out, W - pad_out, H_GUESS - pad_out]
    rrect(d, CARD, 26, fill=CARD_BG, outline=BORDER, width=3)

    y = pad_out + 40
    ctext(d, cx, y, brand, F("bold", 20), BRAND)
    y += 34
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 42

    # ---------- cabeçalho times ----------
    r_crest = 42
    left_cx = pad_out + pad_in + r_crest
    right_cx = W - pad_out - pad_in - r_crest
    paste_crest(img, load_crest(crest_home), left_cx, y, r_crest,
                (home or "?")[:2].upper(), GREEN)
    paste_crest(img, load_crest(crest_away), right_cx, y, r_crest,
                (away or "?")[:2].upper(), BLUE)

    name1 = str(home).title()
    name2 = str(away).title()
    fsize = 30
    while fsize > 18:
        tf = F("bold", fsize)
        w1, w2 = text_w(d, name1, tf), text_w(d, name2, tf)
        left_edge = left_cx + r_crest + 18 + w1
        right_edge = right_cx - r_crest - 18 - w2
        if right_edge - left_edge > 50:
            break
        fsize -= 2
    tf = F("bold", fsize)
    w2 = text_w(d, name2, tf)
    name_y = y - 10
    d.text((left_cx + r_crest + 18, name_y), name1, font=tf, fill=WHITE, anchor="lm")
    d.text((right_cx - r_crest - 18 - w2, name_y), name2, font=tf, fill=WHITE, anchor="lm")
    vs_x = (left_edge + right_edge) / 2
    ctext(d, vs_x, name_y, "X", F("bold", 20), GREY)

    tag_y = y + 20
    ex = left_cx + r_crest + 18
    icon(d, "house", ex + 7, tag_y, 14, GREEN)
    d.text((ex + 18, tag_y), "MANDANTE", font=F("bold", 13), fill=GREEN, anchor="lm")

    tag2 = "VISITANTE"
    tag2_w = text_w(d, tag2, F("bold", 13))
    ex2 = right_cx - r_crest - 18
    d.text((ex2 - tag2_w - 20, tag_y), tag2, font=F("bold", 13), fill=BLUE, anchor="lm")
    icon(d, "plane", ex2 - 7, tag_y, 14, BLUE)

    y += 62

    # ---------- ODDS 1X2 ----------
    box_h = 244
    box = [pad_out + pad_in, y, W - pad_out - pad_in, y + box_h]
    rrect(d, box, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)
    section_title(d, cx, box[1] + 28, None, "ODDS 1X2", WHITE, fnt_size=18)

    cols = [("CASA", odd_casa, odd_justa_casa), ("EMPATE", odd_empate, odd_justa_empate),
            ("FORA", odd_fora, odd_justa_fora)]
    colw = (box[2] - box[0]) / 3
    for i, (lbl, real, justa) in enumerate(cols):
        ccx = box[0] + colw * i + colw / 2
        ctext(d, ccx, box[1] + 68, lbl, F("bold", 15), GREY)
        ctext(d, ccx, box[1] + 108, f"{real}", F("bold", 40), WHITE)
        ctext(d, ccx, box[1] + 152, "ODD JUSTA", F("regular", 12), BRAND)
        ctext(d, ccx, box[1] + 172, f"{justa:.2f}", F("bold", 18), BLUE)

    d.line([(box[0] + 20, box[1] + 202), (box[2] - 20, box[1] + 202)], fill=INNER_BORDER, width=2)
    cons = f"CASA: {odd_justa_casa:.2f}   |   EMPATE: {odd_justa_empate:.2f}   |   FORA: {odd_justa_fora:.2f}"
    ctext(d, cx, box[1] + 224, cons, F("bold", 14), LGREY)
    y = box[3] + 22

    # ---------- Top5 placares + Over/Under ----------
    gap = 18
    colw2 = (W - 2 * pad_out - 2 * pad_in - gap) / 2
    b1_h = 44 + max(len(top5), 3) * 40 + 20
    b1 = [pad_out + pad_in, y, pad_out + pad_in + colw2, y + b1_h]
    b2 = [b1[2] + gap, y, W - pad_out - pad_in, b1[3]]
    rrect(d, b1, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)
    rrect(d, b2, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)

    section_title(d, (b1[0] + b1[2]) / 2, b1[1] + 26, "bars", "TOP 5 PLACARES", WHITE, GREEN, 15)
    ry = b1[1] + 62
    for i, (gh, ga, prob) in enumerate(top5[:5]):
        bx = b1[0] + 20
        d.rounded_rectangle([bx, ry - 12, bx + 24, ry + 12], radius=6, fill=GREEN)
        ctext(d, bx + 12, ry, str(i + 1), F("bold", 13), (8, 12, 8))
        d.text((bx + 36, ry), f"{gh} - {ga}", font=F("bold", 18), fill=WHITE, anchor="lm")
        pw = text_w(d, prob, F("bold", 16))
        d.text((b1[2] - 20 - pw, ry), prob, font=F("bold", 16), fill=GREEN, anchor="lm")
        ry += 40

    section_title(d, (b2[0] + b2[2]) / 2, b2[1] + 26, "trend", "OVER / UNDER FT", WHITE, BLUE, 15)
    thy = b2[1] + 56
    col_linha = b2[0] + 20
    col_over = b2[0] + colw2 * 0.52
    col_under = b2[2] - 20
    d.text((col_linha, thy), "LINHA", font=F("bold", 11), fill=GREY, anchor="lm")
    ctext(d, col_over, thy, "OVER", F("bold", 11), GREY, anchor="mm")
    d.text((col_under, thy), "UNDER", font=F("bold", 11), fill=GREY, anchor="rm")
    oy = thy + 30
    for linha in ["0.5", "1.5", "2.5"]:
        over_p, under_p = over_under.get(linha, (None, None))
        d.text((col_linha, oy), linha, font=F("bold", 16), fill=LGREY, anchor="lm")
        over_txt = f"{over_p:.2f}%" if over_p is not None else "—"
        under_txt = f"{under_p:.2f}%" if under_p is not None else "—"
        ctext(d, col_over, oy, over_txt, F("bold", 15), GREEN, anchor="mm")
        d.text((col_under, oy), under_txt, font=F("bold", 15), fill=RED, anchor="rm")
        oy += 36

    y = b1[3] + 22

    # ---------- Top 4 mercados ----------
    n_mkt = len(top4_mercados)
    if n_mkt == 0:
        box3_h = 96
        box3 = [pad_out + pad_in, y, W - pad_out - pad_in, y + box3_h]
        rrect(d, box3, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)
        section_title(d, cx, box3[1] + 26, "shield", titulo_mercados, WHITE, BLUE, 15)
        ctext(d, cx, box3[1] + 66, "Nenhum mercado em destaque para este jogo.", F("regular", 13), GREY)
    else:
        box3_h = 300
        box3 = [pad_out + pad_in, y, W - pad_out - pad_in, y + box3_h]
        rrect(d, box3, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)
        section_title(d, cx, box3[1] + 26, "shield", titulo_mercados, WHITE, BLUE, 15)

        mcolw = (box3[2] - box3[0]) / n_mkt
        inner_pad = 10
        mtop = box3[1] + 50
        mbot = box3[3] - 16
        for i, (label, val, color, icon_kind, prob_txt) in enumerate(top4_mercados):
            color = hexcol(color) if color else MARKET_PALETTE[i % len(MARKET_PALETTE)]
            mx0 = box3[0] + i * mcolw + inner_pad
            mx1 = box3[0] + (i + 1) * mcolw - inner_pad
            mbox = [mx0, mtop, mx1, mbot]
            rrect(d, mbox, 14, fill=(9, 12, 20), outline=INNER_BORDER, width=1)
            mcx = (mx0 + mx1) / 2
            max_label_w = (mx1 - mx0) - 12
            lines = wrap_lines(d, label.upper(), F("bold", 13), max_label_w, max_lines=2)
            ly = mtop + 18
            for ln in lines:
                ctext(d, mcx, ly, ln, F("bold", 13), LGREY)
                ly += 16
            icon_y = mtop + 54
            icon(d, icon_for_market(icon_kind) if icon_kind in (None, "goal", "boot") else icon_kind,
                 mcx, icon_y, 26, color)
            sy = mtop + 92
            d.text((mcx, sy), f"{float(val):.0f}", font=F("bold", 30), fill=color, anchor="mm")
            sw = text_w(d, f"{float(val):.0f}", F("bold", 30))
            d.text((mcx + sw / 2 + 3, sy + 7), "/100", font=F("bold", 13), fill=GREY, anchor="lm")
            bar_y = sy + 26
            bx0, bx1 = mx0 + 12, mx1 - 12
            d.rounded_rectangle([bx0, bar_y, bx1, bar_y + 7], radius=3, fill=(26, 31, 44))
            fx = bx0 + (bx1 - bx0) * max(0, min(float(val), 100)) / 100
            d.rounded_rectangle([bx0, bar_y, fx, bar_y + 7], radius=3, fill=color)
            pl_y = bar_y + 24
            plabel_lines = wrap_lines(d, f"PROB. {label.upper()}", F("regular", 10), max_label_w, max_lines=2)
            py = pl_y
            for ln in plabel_lines:
                ctext(d, mcx, py, ln, F("regular", 10), GREY)
                py += 13
            ctext(d, mcx, py + 12, prob_txt, F("bold", 16), color)

    y = box3[3] + 22

    # ---------- Sinal principal / comentário live (caixas separadas) ----------
    if sinal_texto or mostrar_live:
        n_items = (1 if sinal_texto else 0) + (1 if mostrar_live else 0)
        scolw_full = (W - 2 * pad_out - pad_in * 2 - gap * (n_items - 1)) if n_items > 1 else (W - 2 * pad_out - 2 * pad_in)
        item_w = scolw_full / n_items if n_items else scolw_full

        def build_signal_box(x0, txt, col, label, icon_kind, extra_stars=None, subtext=None, subcolor=None):
            lines = wrap_lines(d, txt, F("bold", 20), item_w - 28, max_lines=2)
            n_lines = len(lines)
            h = 40 + n_lines * 26 + (18 if subtext else 0) + 14
            box = [x0, y, x0 + item_w, y + h]
            rrect(d, box, 16, fill=INNER_BG, outline=INNER_BORDER, width=2)
            section_title(d, (box[0] + box[2]) / 2, box[1] + 22, icon_kind, label, GREY, col, 12)
            ty = box[1] + 50
            for ln in lines:
                ctext(d, (box[0] + box[2]) / 2, ty, ln, F("bold", 20), col)
                ty += 26
            if extra_stars:
                sx0 = (box[0] + box[2]) / 2 - (extra_stars * 16) / 2 + 8
                for si in range(extra_stars):
                    icon(d, "star", sx0 + si * 16, ty + 2, 12, col)
                ty += 22
            if subtext:
                stw = text_w(d, subtext, F("bold", 13))
                heart_x = (box[0] + box[2]) / 2 - stw / 2 - 10
                icon(d, "heart", heart_x, ty + 2, 12, subcolor or PURPLE)
                d.text(((box[0] + box[2]) / 2 - stw / 2 + 4, ty),
                       subtext, font=F("bold", 13), fill=subcolor or PURPLE, anchor="lm")
            return box[3]

        max_bottom = y
        cur_x = pad_out + pad_in
        if sinal_texto:
            bot = build_signal_box(cur_x, sinal_texto, hexcol(sinal_cor) or GREEN, "SINAL PRINCIPAL",
                                    "star", extra_stars=sinal_estrelas, subtext=sinal_subtexto, subcolor=PURPLE)
            max_bottom = max(max_bottom, bot)
            cur_x += item_w + gap
        if mostrar_live:
            bot = build_signal_box(cur_x, "ANALISAR / ACOMPANHAR JOGO LIVE", ORANGE, "COMENTÁRIO ADICIONAL", "speech")
            max_bottom = max(max_bottom, bot)
        y = max_bottom + 22

    # ---------- rodapé ----------
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 36
    conf_color = GREEN if confianca == "ALTA" else (ORANGE if confianca == "MÉDIA" else RED)
    footer = [("ÍNDICE TÁTICO", "target", f"{indice_tatico}/100", GREEN),
              ("PERFIL", "person", perfil_label, BLUE),
              ("CONFIANÇA", "shield", confianca, conf_color)]
    fcolw = (W - 2 * pad_out - 2 * pad_in) / 3
    for i, (lbl, ic, val, col) in enumerate(footer):
        fcx = pad_out + pad_in + fcolw * i + fcolw / 2
        icon(d, ic, fcx, y, 20, col)
        ctext(d, fcx, y + 26, lbl, F("bold", 12), GREY)
        ctext(d, fcx, y + 50, val, F("bold", 18), col)
    y += 76

    final_h = y + pad_out
    img = img.crop((0, 0, W, final_h))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
