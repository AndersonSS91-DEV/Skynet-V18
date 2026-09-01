# ============================================================
# card_instagram_engine.py  (v3 — tipografia premium + robustez)
#
# O que mudou nesta versão:
#   1) Fonte trocada de DejaVu Sans para Inter (a mesma família
#      usada por Stripe, Linear, etc.) — visual muito mais sério
#      e profissional. É uma fonte variável (1 arquivo só cobre
#      Regular / Medium / SemiBold / Bold / ExtraBold).
#   2) Sanitização de texto: qualquer emoji/caractere que a fonte
#      não consiga desenhar (o que causava aqueles quadradinhos
#      pretos "tofu" no SINAL PRINCIPAL) agora é limpo antes de
#      desenhar — estrelas tipo ⭐ viram ★ automaticamente.
#   3) Paleta mais contida/séria (menos neon, mais "dashboard
#      financeiro"), com um dourado discreto como acento premium.
#   4) Tudo dentro de UMA função (render_card_instagram_ui) que já
#      cuida de mostrar a imagem E o botão de download — chame ela
#      com UMA linha dentro do "with tab1:" pra eliminar qualquer
#      risco de indentação vazar pra outras abas.
#
# Precisa da pasta fonts/ ao lado deste arquivo:
#   seu_projeto/
#     app.py
#     card_instagram_engine.py
#     fonts/
#       Inter-Variable.ttf      <- nova, principal
#       DejaVuSans.ttf          <- fallback (opcional, mas recomendado)
#       DejaVuSans-Bold.ttf     <- fallback (opcional, mas recomendado)
# ============================================================
import io
import os
import re
import base64
import functools
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ------------------------------------------------------------------
# Fontes — Inter (variável) como principal, DejaVu como reserva
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_INTER_CANDIDATES = [
    os.path.join(_BASE_DIR, "fonts", "Inter-Variable.ttf"),
    os.path.join(_BASE_DIR, "fonts", "Inter[opsz,wght].ttf"),
    "/usr/share/fonts/truetype/inter/Inter-Variable.ttf",
]
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

_INTER_WEIGHT = {
    "regular": b"Regular",
    "medium": b"Medium",
    "semibold": b"SemiBold",
    "bold": b"Bold",
    "extrabold": b"ExtraBold",
}


@functools.lru_cache(maxsize=None)
def _inter_path():
    for path in _INTER_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@functools.lru_cache(maxsize=None)
def F(weight, size):
    """weight: 'regular' | 'medium' | 'semibold' | 'bold' | 'extrabold'"""
    ipath = _inter_path()
    if ipath:
        try:
            f = ImageFont.truetype(ipath, size)
            f.set_variation_by_name(_INTER_WEIGHT.get(weight, b"Regular"))
            return f
        except Exception:
            pass
    # fallback: DejaVu só tem regular/bold
    candidates = _BOLD_CANDIDATES if weight in ("bold", "extrabold", "semibold") else _REGULAR_CANDIDATES
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
# Sanitização de texto — evita "tofu" (quadrados pretos) quando o
# texto vem com emoji que a fonte não sabe desenhar.
# ------------------------------------------------------------------
_STAR_EQUIVALENTS = "\u2b50\U0001f31f\U0001f320\u2728"  # ⭐ 🌟 🌠 ✨
_HEART_EQUIVALENTS = "\U0001f49c\U0001f496\u2764\U0001f49a\U0001f499"  # 💜 💖 ❤ 💚 💙
_STRIP_RANGES = [
    (0x1F000, 0x1FFFF),   # emoji / símbolos suplementares
    (0x2600, 0x26FF),     # símbolos diversos (exceto tratados abaixo)
    (0x2700, 0x27BF),     # dingbats
    (0xFE00, 0xFE0F),     # seletores de variação
    (0x200D, 0x200D),     # ZWJ
]
_KEEP_EXPLICIT = {0x2605, 0x2606}  # ★ ☆ — Inter/DejaVu sabem desenhar


def clean_text(s):
    """Remove/normaliza caracteres que a fonte não consegue renderizar,
    pra nunca mais aparecer quadradinho preto no card."""
    if not isinstance(s, str):
        return s
    out = []
    for ch in s:
        cp = ord(ch)
        if ch in _STAR_EQUIVALENTS:
            out.append("\u2605")  # normaliza pra ★
            continue
        if ch in _HEART_EQUIVALENTS:
            out.append("\u2665")  # normaliza pra ♥ (Inter tem esse glifo)
            continue
        if cp in _KEEP_EXPLICIT:
            out.append(ch)
            continue
        stripped = False
        for lo, hi in _STRIP_RANGES:
            if lo <= cp <= hi:
                stripped = True
                break
        if stripped:
            continue
        out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------
# Paleta — tom "dashboard financeiro sério", pouco neon
# ------------------------------------------------------------------
BG            = (6, 9, 15)
CARD_BG       = (4, 6, 11)
BORDER        = (43, 78, 145)      # azul profundo, discreto
INNER_BG      = (10, 13, 20)
INNER_BORDER  = (25, 31, 44)
BRAND         = (135, 172, 224)    # azul acinzentado (menos neon)
WHITE         = (238, 241, 246)
GREY          = (140, 148, 163)
LGREY         = (198, 205, 216)
GREEN         = (63, 181, 124)     # verde "financeiro", não neon
RED           = (214, 92, 92)      # vermelho mais seco
BLUE          = (94, 145, 219)
GOLD          = (196, 165, 96)     # acento premium, uso pontual
PURPLE        = (150, 122, 196)

MARKET_PALETTE = [RED, GOLD, BLUE, PURPLE]


def hexcol(c):
    if isinstance(c, str) and c.startswith("#"):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return c


# ------------------------------------------------------------------
# Helpers de texto/forma (todos passam por clean_text)
# ------------------------------------------------------------------
def text_w(d, s, fnt):
    s = clean_text(s)
    b = d.textbbox((0, 0), s, font=fnt)
    return b[2] - b[0]


def dtext(d, xy, s, font, fill, anchor=None):
    s = clean_text(s)
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def ctext(d, cx, cy, s, fnt, fill, anchor="mm"):
    dtext(d, (cx, cy), s, fnt, fill, anchor=anchor)


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def wrap_lines(d, text, fnt, max_w, max_lines=2):
    text = clean_text(text)
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


def section_title(d, cx, y, icon_kind, text, color=WHITE, icon_color=None, fnt_size=18, weight="semibold"):
    icon_color = icon_color or color
    fnt = F(weight, fnt_size)
    tw = text_w(d, text, fnt)
    has_icon = icon_kind is not None
    icon_s = fnt_size
    gap = 9 if has_icon else 0
    total_w = (icon_s + gap if has_icon else 0) + tw
    x0 = cx - total_w / 2
    if has_icon:
        icon(d, icon_kind, x0 + icon_s / 2, y, icon_s, icon_color)
        dtext(d, (x0 + icon_s + gap, y), text, fnt, color, anchor="lm")
    else:
        dtext(d, (x0, y), text, fnt, color, anchor="lm")


def load_crest(source):
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
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 19, 27),
                  outline=fallback_color, width=3)
        ctext(d, cx, cy, fallback_text, F("bold", int(r * 0.6)), fallback_color)


# ------------------------------------------------------------------
# Ícones vetoriais
# ------------------------------------------------------------------
def icon(d, kind, cx, cy, s, color):
    r = s / 2
    w = max(2, round(s / 11))
    if kind == "target":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=w)
        d.ellipse([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6], outline=color, width=w)
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
        d.rounded_rectangle([cx - r, cy + r * 0.15, cx + r * 0.55, cy + r * 0.55],
                             radius=r * 0.18, outline=color, width=max(1, w - 1))
        pts = [(cx - r, cy + r * 0.15), (cx - r * 0.85, cy - r * 0.55),
               (cx - r * 0.1, cy - r * 0.55), (cx + r * 0.05, cy - r * 0.1),
               (cx + r * 0.5, cy + r * 0.05), (cx + r * 0.55, cy + r * 0.15)]
        d.line(pts, fill=color, width=max(1, w - 1), joint="curve")
        for i in range(3):
            px = cx - r * 0.55 + i * r * 0.4
            d.line([(px, cy + r * 0.55), (px, cy + r * 0.68)], fill=color, width=max(1, w - 1))
        bcx, bcy, br = cx + r * 0.78, cy + r * 0.42, r * 0.24
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
# Função principal — desenha o card e retorna bytes PNG
# ------------------------------------------------------------------
def gerar_card_instagram(
    home, away,
    crest_home=None, crest_away=None,
    odd_casa="-", odd_empate="-", odd_fora="-",
    odd_justa_casa=0.0, odd_justa_empate=0.0, odd_justa_fora=0.0,
    top5=None,
    over_under=None,
    top4_mercados=None,
    titulo_mercados="TOP 4 MERCADOS",
    sinal_texto=None, sinal_cor=None, sinal_estrelas=None, sinal_subtexto=None,
    mostrar_live=False,
    indice_tatico=50, perfil_label="EQUILIBRADO", confianca="MÉDIA",
    brand="@laratodata",
    width=880,
):
    """Retorna os bytes de um PNG pronto para st.image / download_button."""
    top5 = top5 or []
    over_under = over_under or {}
    top4_mercados = top4_mercados or []

    W = width
    pad_out = 34
    pad_in = 28
    cx = W // 2

    H_GUESS = 2400
    img = Image.new("RGB", (W, H_GUESS), BG)
    d = ImageDraw.Draw(img)

    CARD = [pad_out, pad_out, W - pad_out, H_GUESS - pad_out]
    rrect(d, CARD, 22, fill=CARD_BG, outline=BORDER, width=3)

    y = pad_out + 38
    ctext(d, cx, y, clean_text(brand), F("semibold", 19), BRAND)
    y += 32
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 40

    # ---------- cabeçalho times ----------
    r_crest = 40
    left_cx = pad_out + pad_in + r_crest
    right_cx = W - pad_out - pad_in - r_crest
    paste_crest(img, load_crest(crest_home), left_cx, y, r_crest,
                (home or "?")[:2].upper(), GOLD)
    paste_crest(img, load_crest(crest_away), right_cx, y, r_crest,
                (away or "?")[:2].upper(), BLUE)

    name1 = str(home).title()
    name2 = str(away).title()
    fsize = 27
    while fsize > 17:
        tf = F("bold", fsize)
        w1, w2 = text_w(d, name1, tf), text_w(d, name2, tf)
        left_edge = left_cx + r_crest + 16 + w1
        right_edge = right_cx - r_crest - 16 - w2
        if right_edge - left_edge > 46:
            break
        fsize -= 1
    tf = F("bold", fsize)
    w2 = text_w(d, name2, tf)
    name_y = y - 9
    dtext(d, (left_cx + r_crest + 16, name_y), name1, tf, WHITE, anchor="lm")
    dtext(d, (right_cx - r_crest - 16 - w2, name_y), name2, tf, WHITE, anchor="lm")
    vs_x = (left_edge + right_edge) / 2
    ctext(d, vs_x, name_y, "x", F("medium", 17), GREY)

    tag_y = y + 19
    ex = left_cx + r_crest + 16
    icon(d, "house", ex + 6, tag_y, 13, GREEN)
    dtext(d, (ex + 16, tag_y), "MANDANTE", F("semibold", 12), GREEN, anchor="lm")

    tag2 = "VISITANTE"
    tag2_w = text_w(d, tag2, F("semibold", 12))
    ex2 = right_cx - r_crest - 16
    dtext(d, (ex2 - tag2_w - 18, tag_y), tag2, F("semibold", 12), BLUE, anchor="lm")
    icon(d, "plane", ex2 - 6, tag_y, 13, BLUE)

    y += 58

    # ---------- ODDS 1X2 ----------
    box_h = 226
    box = [pad_out + pad_in, y, W - pad_out - pad_in, y + box_h]
    rrect(d, box, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)
    section_title(d, cx, box[1] + 26, None, "ODDS 1X2", WHITE, fnt_size=16)

    cols = [("CASA", odd_casa, odd_justa_casa), ("EMPATE", odd_empate, odd_justa_empate),
            ("FORA", odd_fora, odd_justa_fora)]
    colw = (box[2] - box[0]) / 3
    for i, (lbl, real, justa) in enumerate(cols):
        ccx = box[0] + colw * i + colw / 2
        ctext(d, ccx, box[1] + 64, lbl, F("semibold", 13), GREY)
        ctext(d, ccx, box[1] + 100, f"{real}", F("bold", 36), WHITE)
        ctext(d, ccx, box[1] + 140, "ODD JUSTA", F("medium", 11), BRAND)
        ctext(d, ccx, box[1] + 159, f"{justa:.2f}", F("bold", 17), GOLD)

    d.line([(box[0] + 18, box[1] + 187), (box[2] - 18, box[1] + 187)], fill=INNER_BORDER, width=2)
    cons = f"CASA: {odd_justa_casa:.2f}   |   EMPATE: {odd_justa_empate:.2f}   |   FORA: {odd_justa_fora:.2f}"
    ctext(d, cx, box[1] + 207, cons, F("semibold", 12.5) if False else F("semibold", 13), LGREY)
    y = box[3] + 20

    # ---------- Top5 placares + Over/Under ----------
    gap = 16
    colw2 = (W - 2 * pad_out - 2 * pad_in - gap) / 2
    b1_h = 40 + max(len(top5), 3) * 37 + 18
    b1 = [pad_out + pad_in, y, pad_out + pad_in + colw2, y + b1_h]
    b2 = [b1[2] + gap, y, W - pad_out - pad_in, b1[3]]
    rrect(d, b1, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)
    rrect(d, b2, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)

    section_title(d, (b1[0] + b1[2]) / 2, b1[1] + 24, "bars", "TOP 5 PLACARES", WHITE, GOLD, 14)
    ry = b1[1] + 58
    for i, (gh, ga, prob) in enumerate(top5[:5]):
        bx = b1[0] + 18
        d.rounded_rectangle([bx, ry - 11, bx + 22, ry + 11], radius=5, fill=GOLD)
        ctext(d, bx + 11, ry, str(i + 1), F("bold", 12), (12, 12, 10))
        dtext(d, (bx + 32, ry), f"{gh} - {ga}", F("semibold", 16), WHITE, anchor="lm")
        pw = text_w(d, prob, F("bold", 15))
        dtext(d, (b1[2] - 18 - pw, ry), prob, F("bold", 15), GREEN, anchor="lm")
        ry += 37

    section_title(d, (b2[0] + b2[2]) / 2, b2[1] + 24, "trend", "OVER / UNDER FT", WHITE, BLUE, 14)
    thy = b2[1] + 52
    col_linha = b2[0] + 18
    col_over = b2[0] + colw2 * 0.52
    col_under = b2[2] - 18
    dtext(d, (col_linha, thy), "LINHA", F("semibold", 10.5) if False else F("semibold", 11), GREY, anchor="lm")
    ctext(d, col_over, thy, "OVER", F("semibold", 11), GREY, anchor="mm")
    dtext(d, (col_under, thy), "UNDER", F("semibold", 11), GREY, anchor="rm")
    oy = thy + 28
    for linha in ["0.5", "1.5", "2.5"]:
        over_p, under_p = over_under.get(linha, (None, None))
        dtext(d, (col_linha, oy), linha, F("semibold", 15), LGREY, anchor="lm")
        over_txt = f"{over_p:.2f}%" if over_p is not None else "—"
        under_txt = f"{under_p:.2f}%" if under_p is not None else "—"
        ctext(d, col_over, oy, over_txt, F("bold", 14), GREEN, anchor="mm")
        dtext(d, (col_under, oy), under_txt, F("bold", 14), RED, anchor="rm")
        oy += 33

    y = b1[3] + 20

    # ---------- Top 4 mercados ----------
    n_mkt = len(top4_mercados)
    if n_mkt == 0:
        box3_h = 88
        box3 = [pad_out + pad_in, y, W - pad_out - pad_in, y + box3_h]
        rrect(d, box3, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)
        section_title(d, cx, box3[1] + 24, "shield", titulo_mercados, WHITE, BLUE, 14)
        ctext(d, cx, box3[1] + 60, "Nenhum mercado em destaque para este jogo.", F("regular", 12), GREY)
    else:
        box3_h = 276
        box3 = [pad_out + pad_in, y, W - pad_out - pad_in, y + box3_h]
        rrect(d, box3, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)
        section_title(d, cx, box3[1] + 24, "shield", titulo_mercados, WHITE, BLUE, 14)

        mcolw = (box3[2] - box3[0]) / n_mkt
        inner_pad = 9
        mtop = box3[1] + 46
        mbot = box3[3] - 14
        for i, (label, val, color, icon_kind, prob_txt) in enumerate(top4_mercados):
            color = hexcol(color) if color else MARKET_PALETTE[i % len(MARKET_PALETTE)]
            mx0 = box3[0] + i * mcolw + inner_pad
            mx1 = box3[0] + (i + 1) * mcolw - inner_pad
            mbox = [mx0, mtop, mx1, mbot]
            rrect(d, mbox, 12, fill=(8, 10, 17), outline=INNER_BORDER, width=1)
            mcx = (mx0 + mx1) / 2
            max_label_w = (mx1 - mx0) - 10
            lines = wrap_lines(d, label.upper(), F("semibold", 12), max_label_w, max_lines=2)
            ly = mtop + 16
            for ln in lines:
                ctext(d, mcx, ly, ln, F("semibold", 12), LGREY)
                ly += 15
            icon_y = mtop + 50
            icon(d, icon_for_market(icon_kind) if icon_kind in (None, "goal", "boot") else icon_kind,
                 mcx, icon_y, 24, color)
            sy = mtop + 84
            dtext(d, (mcx, sy), f"{float(val):.0f}", F("bold", 27), color, anchor="mm")
            sw = text_w(d, f"{float(val):.0f}", F("bold", 27))
            dtext(d, (mcx + sw / 2 + 3, sy + 6), "/100", F("semibold", 12), GREY, anchor="lm")
            bar_y = sy + 24
            bx0, bx1 = mx0 + 10, mx1 - 10
            d.rounded_rectangle([bx0, bar_y, bx1, bar_y + 6], radius=3, fill=(24, 28, 40))
            fx = bx0 + (bx1 - bx0) * max(0, min(float(val), 100)) / 100
            d.rounded_rectangle([bx0, bar_y, fx, bar_y + 6], radius=3, fill=color)
            pl_y = bar_y + 22
            plabel_lines = wrap_lines(d, f"PROB. {label.upper()}", F("regular", 9.5) if False else F("regular", 10), max_label_w, max_lines=2)
            py = pl_y
            for ln in plabel_lines:
                ctext(d, mcx, py, ln, F("regular", 10), GREY)
                py += 12
            ctext(d, mcx, py + 12, prob_txt, F("bold", 15), color)

    y = box3[3] + 20

    # ---------- Sinal principal / comentário live ----------
    if sinal_texto or mostrar_live:
        n_items = (1 if sinal_texto else 0) + (1 if mostrar_live else 0)
        item_w = ((W - 2 * pad_out - pad_in * 2 - gap * (n_items - 1)) if n_items > 1
                  else (W - 2 * pad_out - 2 * pad_in)) / n_items

        def build_signal_box(x0, txt, col, label, icon_kind, extra_stars=None, subtext=None, subcolor=None):
            txt_clean = clean_text(txt)
            lines = wrap_lines(d, txt_clean, F("bold", 18), item_w - 24, max_lines=2)
            n_lines = len(lines)
            h = 36 + n_lines * 23 + (16 if subtext else 0) + (20 if extra_stars else 0) + 12
            box = [x0, y, x0 + item_w, y + h]
            rrect(d, box, 14, fill=INNER_BG, outline=INNER_BORDER, width=2)
            section_title(d, (box[0] + box[2]) / 2, box[1] + 20, icon_kind, label, GREY, col, 11)
            ty = box[1] + 46
            for ln in lines:
                ctext(d, (box[0] + box[2]) / 2, ty, ln, F("bold", 18), col)
                ty += 23
            if extra_stars:
                sx0 = (box[0] + box[2]) / 2 - (extra_stars * 15) / 2 + 7
                for si in range(extra_stars):
                    icon(d, "star", sx0 + si * 15, ty + 2, 11, GOLD)
                ty += 20
            if subtext:
                stc = clean_text(subtext)
                stw = text_w(d, stc, F("semibold", 12))
                hx = (box[0] + box[2]) / 2 - stw / 2 - 9
                icon(d, "heart", hx, ty + 2, 11, subcolor or PURPLE)
                dtext(d, ((box[0] + box[2]) / 2 - stw / 2 + 4, ty), stc, F("semibold", 12), subcolor or PURPLE, anchor="lm")
            return box[3]

        max_bottom = y
        cur_x = pad_out + pad_in
        if sinal_texto:
            bot = build_signal_box(cur_x, sinal_texto, hexcol(sinal_cor) or GREEN, "SINAL PRINCIPAL",
                                    "star", extra_stars=sinal_estrelas, subtext=sinal_subtexto, subcolor=PURPLE)
            max_bottom = max(max_bottom, bot)
            cur_x += item_w + gap
        if mostrar_live:
            bot = build_signal_box(cur_x, "ANALISAR / ACOMPANHAR JOGO LIVE", GOLD, "COMENTÁRIO ADICIONAL", "speech")
            max_bottom = max(max_bottom, bot)
        y = max_bottom + 20

    # ---------- rodapé ----------
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 32
    conf_color = GREEN if confianca == "ALTA" else (GOLD if confianca == "MÉDIA" else RED)
    footer = [("ÍNDICE TÁTICO", "target", f"{indice_tatico}/100", GOLD),
              ("PERFIL", "person", perfil_label, BLUE),
              ("CONFIANÇA", "shield", confianca, conf_color)]
    fcolw = (W - 2 * pad_out - 2 * pad_in) / 3
    for i, (lbl, ic, val, col) in enumerate(footer):
        fcx = pad_out + pad_in + fcolw * i + fcolw / 2
        icon(d, ic, fcx, y, 18, col)
        ctext(d, fcx, y + 23, lbl, F("semibold", 11), GREY)
        ctext(d, fcx, y + 45, val, F("bold", 17), col)
    y += 68

    final_h = y + pad_out
    img = img.crop((0, 0, W, final_h))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------------
# Wrapper de UI — mostra a imagem + botão de download.
# Chame ISSO (uma linha só) dentro do "with tabX:" do seu app.
# Assim nenhuma indentação interna deste arquivo pode vazar pra
# outras abas — só o que está DENTRO desta função roda, e ela é
# chamada de um único ponto.
# ------------------------------------------------------------------
def render_card_instagram_ui(st, *, largura_exibicao=420, **kwargs):
    """
    st: o módulo streamlit (passe `st` do seu app).
    largura_exibicao: largura em pixels que a imagem ocupa NA TELA
        (não afeta a resolução do PNG baixado, que continua nítida).
        Reduza esse valor se o card ainda parecer grande demais perto
        de outros elementos da página.
    **kwargs: todos os parâmetros de gerar_card_instagram (home, away,
        odd_casa, top5, over_under, top4_mercados, etc.)
    """
    png = gerar_card_instagram(**kwargs)
    st.image(png, width=largura_exibicao)
    home = kwargs.get("home", "time1")
    away = kwargs.get("away", "time2")
    st.download_button(
        label="📥 Baixar card para Instagram",
        data=png,
        file_name=f"card_{home}_{away}.png".replace(" ", "_"),
        mime="image/png",
        use_container_width=True,
        key=f"download_card_{home}_{away}",
    )
    return png
