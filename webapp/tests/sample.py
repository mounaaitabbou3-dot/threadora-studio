"""Synthetic test artwork (made here, so no customer image is needed in tests)."""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parents[2] / "threadora" / "data" / "fonts" / "DejaVuSans-Bold.ttf"


def sample_png(size=480) -> bytes:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 480
    d.ellipse([60 * s, 40 * s, 420 * s, 400 * s], fill=(200, 30, 45, 255), outline=(60, 20, 20, 255), width=int(8 * s))
    d.rectangle([150 * s, 150 * s, 330 * s, 290 * s], fill=(250, 245, 235, 255))
    d.ellipse([180 * s, 175 * s, 215 * s, 210 * s], fill=(20, 20, 20, 255))
    d.ellipse([265 * s, 175 * s, 300 * s, 210 * s], fill=(20, 20, 20, 255))
    font = ImageFont.truetype(str(FONT), int(56 * s))
    d.text((240 * s, 445 * s), "LOVE", font=font, fill=(30, 60, 140, 255), anchor="mm")
    buf = BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
