"""Unique, high-contrast chart symbols."""
from __future__ import annotations

import os
from pathlib import Path

# DejaVu fonts are bundled (data/fonts, free license) so the PDF is identical on Windows,
# macOS and Linux. THREADORA_FONT_DIR can point elsewhere if ever needed.
FONT_DIR = Path(os.environ.get("THREADORA_FONT_DIR") or Path(__file__).resolve().parent / "data" / "fonts")
SYMBOL_FONT_FILE = FONT_DIR / "DejaVuSans-Bold.ttf"

# Ordered for maximum distinctness; look-alike pairs (◆/♦, ○/◯, O/0, I/1/l) excluded.
SYMBOL_POOL = ("●", "■", "▲", "◆", "★", "♥", "✚", "✖", "○", "□", "△", "◇", "☆", "♣",
               "♠", "▼", "◐", "✿", "⊕", "⊞", "▣", "♪", "⌂", "A", "C", "E", "H", "K",
               "M", "N", "S", "T", "V", "X", "Z", "2", "3", "4", "5", "7", "8", "9")


def assign_symbols(n: int) -> list[str]:
    if n > len(SYMBOL_POOL):
        raise ValueError(f"{n} colors exceed {len(SYMBOL_POOL)} distinct symbols")
    return list(SYMBOL_POOL[:n])


def rel_luminance(rgb) -> float:
    def ch(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a, b) -> float:
    la, lb = rel_luminance(a), rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def symbol_ink(bg_rgb) -> tuple[int, int, int]:
    """Black or white, whichever contrasts more with the cell color (always >= 4.58:1)."""
    black, white = (0, 0, 0), (255, 255, 255)
    return black if contrast_ratio(bg_rgb, black) >= contrast_ratio(bg_rgb, white) else white
