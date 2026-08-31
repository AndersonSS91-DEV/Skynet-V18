# ============================================================
# card_instagram_engine.py
# Motor que desenha o card do Instagram como uma imagem PNG real
# (Pillow), em alta resolução, com fontes grandes e emojis
# coloridos de verdade — pronto para downloadar e postar.
#
# Coloque este arquivo na mesma pasta do seu app Streamlit e
# importe com:
#     from card_instagram_engine import gerar_card_instagram
# ============================================================
import io
import re
import base64
import functools
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ------------------------------------------------------------------
# Fontes
# ------------------------------------------------------------------
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
EMOJI_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_NATIVE_SIZE = 109  # tamanho fixo do bitmap do Noto Color Emoji


@functools.lru_cache(maxsize=None)
def F(weight, size):
    name = "DejaVuSans-Bold.ttf" if weight == "bold" else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size)


@functools.lru_cache(maxsize=None)
def _emoji_font():
    return ImageFont.truetype(EMOJI_FONT_PATH, EMOJI_NATIVE_SIZE)


_VS = "\ufe0f\u200d"


@functools.lru_cache(maxsize=None)
def _render_emoji_glyph(ch, size):
    """Renderiza 1 emoji em resolução nativa e reduz p/ o tamanho pedido."""
    tmp = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    try:
        td.text((15, 8), ch, font=_emoji_font(), embedded_color=True)
    except Exception:
        return None
    bbox = tmp.getbbox()
    if not bbox:
        return None
    tmp = tmp.crop(bbox)
    tmp.thumbnail((size, size), Image.LANCZOS)
    return tmp


def draw_emoji_str(base_img, s, xy, size, anchor="mm", gap=2):
    """Desenha uma sequência curta de emojis centralizada em xy."""
    chars = [c for c in s if c not in _VS]
    glyphs = []
    for c in chars:
        g = _render_emoji_glyph(c, size)
        if g is not None:
            glyphs.append(g)
    if not glyphs:
        return 0
    total_w = sum(g.width for g in glyphs) + gap * (len(glyphs) - 1)
    x, y = xy
    if anchor[0] == "m":
        x0 = x - total_w / 2
    elif anchor[0] == "r":
        x0 = x - total_w
    else:
        x0 = x
    max_h = max(g.height for g in glyphs)
    if anchor[1] == "m":
        y0 = y - max_h / 2
    elif anchor[1] == "b":
        y0 = y - max_h
    else:
        y0 = y
    cx = x0
    for g in glyphs:
        base_img.paste(g, (int(cx), int(y0 + (max_h - g.height) / 2)), g)
        cx += g.width + gap
    return total_w


# ------------------------------------------------------------------
# Paleta
# ------------------------------------------------------------------
BG            = (5, 8, 16)
CARD_BG       = (3, 5, 10)
BORDER        = (47, 111, 237)
INNER_BG      = (8, 11, 18)
INNER_BORDER  = (28, 36, 51)
BRAND         = (127, 179, 255)
WHITE         = (245, 247, 250)
GREY          = (138, 147, 163)
LGREY         = (200, 207, 216)
GREEN         = (74, 222, 128)
RED           = (248, 113, 113)
BLUE          = (74, 157, 255)
ORANGE        = (250, 204, 21)


# ------------------------------------------------------------------
# Helpers de desenho
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


def load_crest(source):
    """Aceita: PIL.Image, bytes, uma string HTML com <img src="..."> (URL ou
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
# Função principal
# ------------------------------------------------------------------
def gerar_card_instagram(
    home, away,
    crest_home=None, crest_away=None,
    odd_casa="-", odd_empate="-", odd_fora="-",
    odd_justa_casa=0.0, odd_justa_empate=0.0, odd_justa_fora=0.0,
    top5=None,                      # list[(gols_home:int, gols_away:int, prob_str:str)]
    over_under=None,                # dict {"0.5": (over_pct, under_pct), "1.5": (...), "2.5": (...)}
    top4_mercados=None,             # list[(label, valor_0_100, cor_hex, emoji, prob_str)]
    sinal_texto=None, sinal_cor=None,
    mostrar_live=False,
    indice_tatico=50, perfil_label="EQUILIBRADO", confianca="MÉDIA",
    brand="@laratodata",
    width=1080,
):
    """Retorna os bytes de um PNG pronto para st.image / download_button."""
    top5 = top5 or []
    over_under = over_under or {}
    top4_mercados = top4_mercados or []

    def hexcol(c):
        if isinstance(c, str) and c.startswith("#"):
            c = c.lstrip("#")
            return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        return c

    W = width
    pad_out = 44
    pad_in = 42
    cx = W // 2

    H_GUESS = 2600
    img = Image.new("RGB", (W, H_GUESS), BG)
    d = ImageDraw.Draw(img)

    CARD = [pad_out, pad_out, W - pad_out, H_GUESS - pad_out]
    rrect(d, CARD, 32, fill=CARD_BG, outline=BORDER, width=3)

    y = pad_out + 52
    ctext(d, cx, y, brand, F("bold", 26), BRAND)
    y += 46
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 58

    # ---------- cabeçalho times ----------
    r_crest = 52
    left_cx = pad_out + pad_in + r_crest
    right_cx = W - pad_out - pad_in - r_crest
    paste_crest(img, load_crest(crest_home), left_cx, y, r_crest,
                (home or "?")[:2].upper(), GREEN)
    paste_crest(img, load_crest(crest_away), right_cx, y, r_crest,
                (away or "?")[:2].upper(), BLUE)

    name1 = str(home).title()
    name2 = str(away).title()
    fsize = 40
    while fsize > 24:
        tf = F("bold", fsize)
        w1, w2 = text_w(d, name1, tf), text_w(d, name2, tf)
        left_edge = left_cx + r_crest + 24 + w1
        right_edge = right_cx - r_crest - 24 - w2
        if right_edge - left_edge > 70:
            break
        fsize -= 2
    tf = F("bold", fsize)
    w2 = text_w(d, name2, tf)
    name_y = y - 14
    d.text((left_cx + r_crest + 24, name_y), name1, font=tf, fill=WHITE, anchor="lm")
    d.text((right_cx - r_crest - 24 - w2, name_y), name2, font=tf, fill=WHITE, anchor="lm")
    vs_x = (left_edge + right_edge) / 2
    ctext(d, vs_x, name_y, "X", F("bold", 28), GREY)

    tag_y = y + 24
    ex = left_cx + r_crest + 24
    ew = draw_emoji_str(img, "🏠", (ex, tag_y), 18, anchor="lm")
    d.text((ex + ew + 6, tag_y), "MANDANTE", font=F("bold", 17), fill=GREEN, anchor="lm")

    tag2 = "VISITANTE"
    tag2_w = text_w(d, tag2, F("bold", 17))
    ex2 = right_cx - r_crest - 24
    d.text((ex2 - tag2_w - 22, tag_y), tag2, font=F("bold", 17), fill=BLUE, anchor="lm")
    draw_emoji_str(img, "✈️", (ex2 - 8, tag_y), 18, anchor="lm")

    y += 78

    # ---------- ODDS 1X2 ----------
    box_h = 300
    box = [pad_out + pad_in, y, W - pad_out - pad_in, y + box_h]
    rrect(d, box, 20, fill=INNER_BG, outline=INNER_BORDER, width=2)
    title_y = box[1] + 40
    icon_w = 26
    total_title_w = icon_w + 10 + text_w(d, "ODDS 1X2", F("bold", 24))
    tstart = cx - total_title_w / 2
    draw_emoji_str(img, "🎯", (tstart, title_y), icon_w, anchor="lm")
    d.text((tstart + icon_w + 10, title_y), "ODDS 1X2", font=F("bold", 24), fill=ORANGE, anchor="lm")

    cols = [("CASA", odd_casa, odd_justa_casa, GREEN),
            ("EMPATE", odd_empate, odd_justa_empate, LGREY),
            ("FORA", odd_fora, odd_justa_fora, BLUE)]
    colw = (box[2] - box[0]) / 3
    for i, (lbl, real, justa, col) in enumerate(cols):
        ccx = box[0] + colw * i + colw / 2
        ctext(d, ccx, box[1] + 88, lbl, F("bold", 21), GREY)
        ctext(d, ccx, box[1] + 138, f"{real}", F("bold", 54), col)
        ctext(d, ccx, box[1] + 194, "ODD JUSTA", F("regular", 16), BRAND)
        ctext(d, ccx, box[1] + 220, f"{justa:.2f}", F("bold", 22), BLUE)

    d.line([(box[0] + 24, box[1] + 254), (box[2] - 24, box[1] + 254)],
           fill=INNER_BORDER, width=2)
    cons = f"CASA: {odd_justa_casa:.2f}   |   EMPATE: {odd_justa_empate:.2f}   |   FORA: {odd_justa_fora:.2f}"
    ctext(d, cx, box[1] + 282, cons, F("bold", 18), LGREY)
    y = box[3] + 28

    # ---------- Top5 placares + Over/Under ----------
    gap = 22
    colw2 = (W - 2 * pad_out - 2 * pad_in - gap) / 2
    b1 = [pad_out + pad_in, y, pad_out + pad_in + colw2, y + 44 + max(len(top5), 3) * 52 + 30]
    b2 = [b1[2] + gap, y, W - pad_out - pad_in, b1[3]]
    rrect(d, b1, 20, fill=INNER_BG, outline=INNER_BORDER, width=2)
    rrect(d, b2, 20, fill=INNER_BG, outline=INNER_BORDER, width=2)

    ctext(d, (b1[0] + b1[2]) / 2, b1[1] + 34, "TOP 5 PLACARES", F("bold", 22), BRAND)
    badge_colors = [ORANGE, LGREY, (205, 127, 50), BLUE, (192, 132, 252)]
    ry = b1[1] + 78
    for i, (gh, ga, prob) in enumerate(top5[:5]):
        bx = b1[0] + 26
        bcol = badge_colors[i] if i < len(badge_colors) else GREY
        d.rounded_rectangle([bx, ry - 15, bx + 30, ry + 15], radius=7, fill=bcol)
        ctext(d, bx + 15, ry, str(i + 1), F("bold", 17), (10, 10, 15))
        d.text((bx + 46, ry), f"{gh} - {ga}", font=F("bold", 22), fill=WHITE, anchor="lm")
        pw = text_w(d, prob, F("bold", 21))
        d.text((b1[2] - 26 - pw, ry), prob, font=F("bold", 21), fill=GREEN, anchor="lm")
        ry += 52

    ctext(d, (b2[0] + b2[2]) / 2, b2[1] + 34, "OVER / UNDER FT", F("bold", 22), BRAND)
    oy = b2[1] + 78
    for linha in ["0.5", "1.5", "2.5"]:
        over_p, under_p = over_under.get(linha, (None, None))
        d.text((b2[0] + 26, oy), f"Linha {linha}", font=F("bold", 21), fill=LGREY, anchor="lm")
        over_txt = f"{over_p:.2f}%" if over_p is not None else "—"
        under_txt = f"{under_p:.2f}%" if under_p is not None else "—"
        ow = text_w(d, over_txt, F("bold", 20))
        sep_w = text_w(d, "  ·  ", F("bold", 20))
        uw = text_w(d, under_txt, F("bold", 20))
        sx = b2[2] - 26 - (ow + sep_w + uw)
        d.text((sx, oy), over_txt, font=F("bold", 20), fill=GREEN, anchor="lm")
        d.text((sx + ow, oy), "  ·  ", font=F("bold", 20), fill=GREY, anchor="lm")
        d.text((sx + ow + sep_w, oy), under_txt, font=F("bold", 20), fill=RED, anchor="lm")
        oy += 52

    y = b1[3] + 28

    # ---------- Top 4 mercados ----------
    n_mkt = max(len(top4_mercados), 1)
    box3_h = 400
    box3 = [pad_out + pad_in, y, W - pad_out - pad_in, y + box3_h]
    rrect(d, box3, 20, fill=INNER_BG, outline=INNER_BORDER, width=2)
    ctext(d, cx, box3[1] + 36, "TOP 4 MERCADOS", F("bold", 22), BRAND)

    mcolw = (box3[2] - box3[0]) / n_mkt
    inner_pad = 12
    mtop = box3[1] + 66
    mbot = box3[3] - 20
    for i, (label, val, color, emoji, prob_txt) in enumerate(top4_mercados):
        color = hexcol(color)
        mx0 = box3[0] + i * mcolw + inner_pad
        mx1 = box3[0] + (i + 1) * mcolw - inner_pad
        mbox = [mx0, mtop, mx1, mbot]
        rrect(d, mbox, 16, fill=(9, 12, 20), outline=INNER_BORDER, width=1)
        mcx = (mx0 + mx1) / 2
        max_label_w = (mx1 - mx0) - 16
        lines = wrap_lines(d, label.upper(), F("bold", 15), max_label_w, max_lines=2)
        ly = mtop + 24
        for ln in lines:
            ctext(d, mcx, ly, ln, F("bold", 15), LGREY)
            ly += 19
        icon_y = mtop + 70
        draw_emoji_str(img, emoji, (mcx, icon_y), 30, anchor="mm")
        sy = mtop + 118
        d.text((mcx, sy), f"{val:.0f}", font=F("bold", 38), fill=color, anchor="mm")
        sw = text_w(d, f"{val:.0f}", F("bold", 38))
        d.text((mcx + sw / 2 + 4, sy + 9), "/100", font=F("bold", 16), fill=GREY, anchor="lm")
        bar_y = sy + 34
        bx0, bx1 = mx0 + 14, mx1 - 14
        d.rounded_rectangle([bx0, bar_y, bx1, bar_y + 9], radius=4, fill=(28, 33, 47))
        fx = bx0 + (bx1 - bx0) * max(0, min(val, 100)) / 100
        d.rounded_rectangle([bx0, bar_y, fx, bar_y + 9], radius=4, fill=color)
        pl_y = bar_y + 30
        plabel_lines = wrap_lines(d, f"PROB. {label.upper()}", F("regular", 12), max_label_w, max_lines=2)
        py = pl_y
        for ln in plabel_lines:
            ctext(d, mcx, py, ln, F("regular", 12), GREY)
            py += 15
        ctext(d, mcx, py + 14, prob_txt, F("bold", 19), color)

    y = box3[3] + 28

    # ---------- Sinal principal / comentário live ----------
    if sinal_texto or mostrar_live:
        items = []
        if sinal_texto:
            items.append(("SINAL PRINCIPAL", "⭐", sinal_texto, hexcol(sinal_cor) or GREEN))
        if mostrar_live:
            items.append(("COMENTÁRIO ADICIONAL", "⚠️", "ANALISAR / ACOMPANHAR JOGO LIVE", ORANGE))
        scolw_probe = (W - 2 * pad_out - 2 * pad_in) / len(items)
        max_lines = 1
        wrapped_cache = []
        for (lbl, emoji, txt, col) in items:
            lns = wrap_lines(d, txt, F("bold", 24), scolw_probe - 30, max_lines=2)
            wrapped_cache.append(lns)
            max_lines = max(max_lines, len(lns))
        sbox_h = 70 + max_lines * 30 + 20
        sbox = [pad_out + pad_in, y, W - pad_out - pad_in, y + sbox_h]
        rrect(d, sbox, 20, fill=INNER_BG, outline=INNER_BORDER, width=2)
        scolw = (sbox[2] - sbox[0]) / len(items)
        for i, (lbl, emoji, txt, col) in enumerate(items):
            scx = sbox[0] + scolw * i + scolw / 2
            lab_w = text_w(d, lbl, F("bold", 15))
            ic_w = 20
            lstart = scx - (ic_w + 8 + lab_w) / 2
            draw_emoji_str(img, emoji, (lstart, sbox[1] + 32), ic_w, anchor="lm")
            d.text((lstart + ic_w + 8, sbox[1] + 32), lbl, font=F("bold", 15), fill=GREY, anchor="lm")
            ty = sbox[1] + 70
            for ln in wrapped_cache[i]:
                ctext(d, scx, ty, ln, F("bold", 24), col)
                ty += 30
        y = sbox[3] + 28

    # ---------- rodapé ----------
    d.line([(pad_out + pad_in, y), (W - pad_out - pad_in, y)], fill=INNER_BORDER, width=2)
    y += 46
    footer = [("ÍNDICE TÁTICO", "🎯", f"{indice_tatico}/100", GREEN),
              ("PERFIL", "👤", perfil_label, BLUE),
              ("CONFIANÇA", "🛡️", confianca,
               GREEN if confianca == "ALTA" else (ORANGE if confianca == "MÉDIA" else RED))]
    fcolw = (W - 2 * pad_out - 2 * pad_in) / 3
    for i, (lbl, emoji, val, col) in enumerate(footer):
        fcx = pad_out + pad_in + fcolw * i + fcolw / 2
        draw_emoji_str(img, emoji, (fcx, y), 22, anchor="mm")
        ctext(d, fcx, y + 30, lbl, F("bold", 15), GREY)
        ctext(d, fcx, y + 60, val, F("bold", 22), col)
    y += 90

    final_h = y + pad_out
    img = img.crop((0, 0, W, final_h))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
