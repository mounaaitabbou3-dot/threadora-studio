"""Synthetic regression designs. Drawn here so tests need no customer artwork.

Each design stresses one family of conversion defects: lettering, thin outlines,
small high-contrast details (eyes, highlights), negative space, many colors,
transparent and opaque-white backgrounds. Every function returns an RGBA PIL image.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parents[1] / "threadora" / "data" / "fonts"


def _font(size, bold=True, serif=False):
    name = "DejaVuSerif-Bold.ttf" if serif else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(FONTS / name), size)


def _canvas(w, h, bg=(0, 0, 0, 0)):
    im = Image.new("RGBA", (w, h), bg)
    return im, ImageDraw.Draw(im)


def memorial_lettering():
    """Heart + 'In Loving Memory' / name / year: the memorial use case."""
    im, d = _canvas(520, 560)
    d.polygon([(260, 250), (120, 130), (130, 60), (200, 40), (260, 95), (320, 40), (390, 60), (400, 130)],
              fill=(178, 24, 43, 255), outline=(90, 10, 20, 255))
    d.ellipse([175, 70, 205, 100], fill=(250, 230, 230, 255))           # highlight
    d.text((260, 330), "In Loving Memory", font=_font(44, serif=True), fill=(40, 40, 40, 255), anchor="mm")
    d.text((260, 410), "ROSE", font=_font(72), fill=(120, 30, 90, 255), anchor="mm")
    d.text((260, 490), "1950 - 2026", font=_font(40), fill=(40, 40, 40, 255), anchor="mm")
    return im


def christmas_ornament():
    im, d = _canvas(420, 480)
    d.rectangle([185, 20, 235, 60], fill=(200, 170, 60, 255))           # cap
    d.ellipse([200, 0, 220, 24], outline=(200, 170, 60, 255), width=4)  # hook ring (hole)
    d.ellipse([40, 60, 380, 400], fill=(20, 110, 60, 255), outline=(10, 60, 30, 255), width=6)
    for y, col in ((150, (220, 30, 40, 255)), (230, (240, 220, 120, 255)), (310, (220, 30, 40, 255))):
        d.rectangle([50, y, 370, y + 22], fill=col)
    for x in range(90, 360, 55):
        d.ellipse([x, 190, x + 16, 206], fill=(255, 255, 255, 255))     # white dots on green
    d.ellipse([110, 100, 140, 130], fill=(200, 240, 210, 255))          # shine
    return im


def teddy_bear():
    im, d = _canvas(440, 480)
    brown, dark, light = (150, 95, 50, 255), (70, 40, 20, 255), (225, 190, 150, 255)
    d.ellipse([60, 30, 150, 120], fill=brown); d.ellipse([290, 30, 380, 120], fill=brown)   # ears
    d.ellipse([85, 55, 125, 95], fill=light); d.ellipse([315, 55, 355, 95], fill=light)
    d.ellipse([90, 60, 350, 300], fill=brown, outline=dark, width=5)                         # head
    d.ellipse([170, 170, 270, 260], fill=light)                                               # muzzle
    d.ellipse([160, 130, 185, 155], fill=(15, 15, 15, 255)); d.ellipse([255, 130, 280, 155], fill=(15, 15, 15, 255))
    d.ellipse([167, 136, 174, 143], fill=(255, 255, 255, 255)); d.ellipse([262, 136, 269, 143], fill=(255, 255, 255, 255))
    d.ellipse([205, 185, 235, 207], fill=dark)                                                # nose
    d.arc([195, 195, 245, 240], 20, 160, fill=dark, width=4)                                  # mouth
    d.ellipse([110, 280, 330, 470], fill=brown, outline=dark, width=5)                        # body
    d.ellipse([170, 330, 270, 440], fill=light)
    return im


def animal_cat():
    im, d = _canvas(460, 420)
    gray, dark = (120, 120, 130, 255), (40, 40, 45, 255)
    d.polygon([(90, 40), (150, 140), (60, 160)], fill=gray, outline=dark)
    d.polygon([(370, 40), (310, 140), (400, 160)], fill=gray, outline=dark)
    d.ellipse([60, 90, 400, 400], fill=gray, outline=dark, width=6)
    for x in (150, 270):
        d.ellipse([x, 190, x + 50, 240], fill=(110, 190, 70, 255), outline=dark, width=3)
        d.ellipse([x + 20, 195, x + 30, 235], fill=(10, 10, 10, 255))     # slit pupil
    d.polygon([(215, 270), (245, 270), (230, 290)], fill=(230, 140, 150, 255))
    for s in (-1, 1):                                                      # whiskers: thin lines
        for dy in (-12, 0, 12):
            d.line([(230 + s * 40, 290 + dy), (230 + s * 160, 275 + dy * 2)], fill=(250, 250, 250, 255), width=3)
    return im


def floral():
    im, d = _canvas(480, 480)
    cx, cy = 240, 240
    for i in range(8):
        a = i * math.pi / 4
        x, y = cx + 120 * math.cos(a), cy + 120 * math.sin(a)
        d.ellipse([x - 60, y - 40, x + 60, y + 40], fill=(235, 120, 170, 255), outline=(160, 40, 100, 255), width=3)
    d.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(250, 200, 40, 255), outline=(180, 120, 0, 255), width=4)
    for i in range(12):
        a = i * math.pi / 6
        x, y = cx + 30 * math.cos(a), cy + 30 * math.sin(a)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(120, 70, 10, 255))
    for x, y in ((40, 40), (440, 40), (40, 440), (440, 440)):                # detached leaves
        d.ellipse([x - 30, y - 15, x + 30, y + 15], fill=(60, 140, 60, 255))
    return im


def house():
    im, d = _canvas(480, 440)
    d.polygon([(40, 200), (240, 40), (440, 200)], fill=(170, 40, 40, 255), outline=(90, 20, 20, 255))
    d.rectangle([80, 200, 400, 420], fill=(235, 215, 170, 255), outline=(90, 70, 40, 255), width=4)
    d.rectangle([210, 300, 270, 420], fill=(100, 60, 30, 255))                                # door
    d.ellipse([255, 355, 263, 363], fill=(240, 200, 60, 255))                                 # knob
    for x in (110, 310):                                                                      # windows with muntins
        d.rectangle([x, 240, x + 60, 300], fill=(150, 200, 240, 255), outline=(255, 255, 255, 255), width=5)
        d.line([(x + 30, 240), (x + 30, 300)], fill=(255, 255, 255, 255), width=4)
        d.line([(x, 270), (x + 60, 270)], fill=(255, 255, 255, 255), width=4)
    d.rectangle([320, 80, 360, 150], fill=(120, 120, 120, 255))                               # chimney
    return im


def fine_typography():
    im, d = _canvas(640, 300)
    d.text((320, 70), "Home Sweet Home", font=_font(52, serif=True), fill=(30, 60, 120, 255), anchor="mm")
    d.text((320, 160), "EST. 2026", font=_font(46), fill=(160, 30, 50, 255), anchor="mm")
    d.text((320, 240), "family & friends", font=_font(40, bold=False), fill=(30, 30, 30, 255), anchor="mm")
    return im


def transparent_wreath():
    """Ring with a transparent centre and transparent gaps: NO_STITCH negative space."""
    im, d = _canvas(460, 460)
    d.ellipse([30, 30, 430, 430], fill=(40, 120, 60, 255))
    d.ellipse([130, 130, 330, 330], fill=(0, 0, 0, 0))                     # transparent centre
    for i in range(10):
        a = i * math.pi / 5
        x, y = 230 + 150 * math.cos(a), 230 + 150 * math.sin(a)
        d.ellipse([x - 16, y - 16, x + 16, y + 16], fill=(210, 20, 40, 255))  # berries
        d.ellipse([x - 6, y - 10, x + 2, y - 2], fill=(255, 255, 255, 255))  # highlight
    d.rectangle([220, 20, 240, 70], fill=(0, 0, 0, 0))                     # cut gap
    return im


def complex_multicolor():
    """Smooth gradients plus hard color boundaries: many DMC colors, bleeding risk."""
    h, w = 400, 400
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8)
    g = (yy / h * 255).astype(np.uint8)
    b = (255 - (xx + yy) / (w + h) * 255).astype(np.uint8)
    a = np.full((h, w), 255, np.uint8)
    arr = np.dstack([r, g, b, a])
    im = Image.fromarray(arr, "RGBA")
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 180, 180], fill=(255, 255, 255, 255))
    d.rectangle([220, 60, 340, 180], fill=(0, 0, 0, 255))
    d.ellipse([120, 220, 280, 380], fill=(250, 210, 0, 255), outline=(200, 0, 0, 255), width=8)
    return im


def opaque_white_background():
    """JPEG-like opaque image on white with WHITE details inside the design (snowman)."""
    im = Image.new("RGBA", (420, 480), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 40, 360, 440], fill=(30, 60, 130, 255))                # night-sky panel
    d.ellipse([140, 240, 280, 400], fill=(255, 255, 255, 255))              # white snowman (detail)
    d.ellipse([160, 140, 260, 250], fill=(255, 255, 255, 255))
    d.ellipse([190, 175, 202, 187], fill=(0, 0, 0, 255)); d.ellipse([218, 175, 230, 187], fill=(0, 0, 0, 255))
    d.polygon([(208, 195), (240, 205), (208, 212)], fill=(240, 120, 20, 255))
    for x, y in ((90, 80), (320, 100), (100, 380), (330, 330)):
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255))   # snowflakes
    return im


DESIGNS = {
    "memorial_lettering": (memorial_lettering, {"phrase": "In Loving Memory", "name": "Rose", "year": "2026"}),
    "christmas_ornament": (christmas_ornament, {}),
    "teddy_bear": (teddy_bear, {}),
    "animal_cat": (animal_cat, {}),
    "floral": (floral, {}),
    "house": (house, {}),
    "fine_typography": (fine_typography, {"phrase": "Home Sweet Home", "name": "family friends", "year": "2026"}),
    "transparent_wreath": (transparent_wreath, {}),
    "complex_multicolor": (complex_multicolor, {}),
    "opaque_white_background": (opaque_white_background, {}),
}


def save(name: str, folder: Path) -> Path:
    fn, _ = DESIGNS[name]
    p = Path(folder) / f"{name}.png"
    im = fn()
    if name == "opaque_white_background":
        im.convert("RGB").save(p)          # no alpha channel at all
    else:
        im.save(p)
    return p
