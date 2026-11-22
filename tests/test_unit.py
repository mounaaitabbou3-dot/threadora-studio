"""Engine unit and regression tests (no order needed)."""
import numpy as np
import pytest
from PIL import Image

from threadora.convert import NO_STITCH, cell_fields, remove_confetti
from threadora.dmc import DATA_SHA256, ciede2000_matrix, dataset_sha256, dmc_by_code, dmc_lab, load_dmc, nearest, to_lab
from threadora.pdf import MIN_CELL_PT, plan_tiles
from threadora.qa import labels_collide
from threadora.sizing import FABRICS, fit_grid, to_inches
from threadora.source import load_source
from threadora.symbols import SYMBOL_POOL, SYMBOL_FONT_FILE, assign_symbols, contrast_ratio, symbol_ink


def test_dmc_dataset_integrity():
    assert dataset_sha256() == DATA_SHA256
    fl = load_dmc()
    assert len(fl) >= 450
    codes = [f.code for f in fl]
    assert len(codes) == len(set(codes))
    known = {"310": "Black", "B5200": "Snow White", "Blanc": "White", "666": "Bright Red", "321": "Red",
             "3865": "Winter White", "Ecru": "Ecru"}
    by = dmc_by_code()
    for code, name in known.items():
        assert by[code].name == name


def test_ciede2000_reference_pair():
    # Sharma et al. (2005) test data pair 1: dE00 = 2.0425
    a = np.array([[50.0, 2.6772, -79.7751]]); b = np.array([[50.0, 0.0, -82.7485]])
    assert abs(ciede2000_matrix(a, b)[0, 0] - 2.0425) < 1e-3


def test_nearest_exact_dmc_color_maps_to_itself():
    fl = load_dmc()
    for f in fl[::37]:
        i, d = nearest(to_lab(np.array([f.rgb], np.uint8)), dmc_lab())
        assert d[0] < 1e-6
        assert fl[i[0]].rgb == f.rgb


@pytest.mark.parametrize("w,h,units", [(3, 3, "in"), (5, 5, "in"), (7, 7, "in"), (9, 9, "in"), (12, 12, "in"),
                                       (16, 16, "in"), (20, 20, "in"), (8, 10, "in"), (20, 30, "cm")])
@pytest.mark.parametrize("src", [(1000, 1000), (1200, 1235), (1600, 900), (700, 1400)])
def test_fit_grid_never_exceeds_request_and_keeps_aspect(w, h, units, src):
    wi, hi = to_inches(w, h, units)
    for fab in FABRICS:
        p = fit_grid(src[0], src[1], wi, hi, fab)
        assert p.cols >= 1 and p.rows >= 1
        assert p.actual_w_in <= wi + 1e-9 and p.actual_h_in <= hi + 1e-9
        assert p.total_cells == p.cols * p.rows
        # aspect error at most one stitch of rounding
        assert abs(p.cols / p.rows - src[0] / src[1]) <= 1.0 / min(p.cols, p.rows) + 1e-9
        # one side is maximal (no wasted size)
        assert p.cols == int(wi * fab.stitches_per_inch + 1e-9) or p.rows == int(hi * fab.stitches_per_inch + 1e-9)


def test_cm_conversion():
    assert to_inches(25.4, 12.7, "cm") == pytest.approx((10, 5))


def test_symbols_unique_in_font_and_contrast():
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(SYMBOL_FONT_FILE)).getBestCmap()
    assert len(set(SYMBOL_POOL)) == len(SYMBOL_POOL)
    assert all(ord(s) in cmap for s in SYMBOL_POOL)
    assert len(assign_symbols(30)) == 30
    with pytest.raises(ValueError):
        assign_symbols(len(SYMBOL_POOL) + 1)
    for f in load_dmc():
        assert contrast_ratio(f.rgb, symbol_ink(f.rgb)) >= 4.5


@pytest.mark.parametrize("cols,rows", [(30, 30), (49, 49), (99, 101), (136, 140), (175, 180), (196, 196),
                                       (220, 280), (330, 330)])
def test_tiles_partition_every_cell_exactly_once(cols, rows):
    tiles, cs = plan_tiles(cols, rows)
    assert cs >= MIN_CELL_PT
    cover = np.zeros((rows, cols), int)
    for t in tiles:
        cover[t.r0:t.r1, t.c0:t.c1] += 1
    assert (cover == 1).all()
    assert len({t.id for t in tiles}) == len(tiles)


def test_label_collision_detector():
    boxes = [("col", 99, 10, 0, 18, 6), ("col", 100, 15, 0, 25, 6)]
    assert labels_collide(boxes)
    boxes = [("col", 195, 0, 0, 10, 6), ("col", 196, 10.5, 0, 20, 6)]
    assert not labels_collide(boxes)


def test_confetti_removal_keeps_high_contrast_detail():
    pal = to_lab(np.array([[200, 30, 30], [205, 35, 35], [255, 255, 255]], np.uint8))
    m = np.zeros((7, 7), np.int32)
    m[3, 3] = 1          # near-identical isolated stitch -> noise
    m[1, 1] = 2          # isolated white highlight -> intentional
    out, removed, kept = remove_confetti(m, pal, np.zeros_like(m, bool))
    assert out[3, 3] == 0 and out[1, 1] == 2 and removed == 1 and kept >= 1


def test_alpha_is_no_stitch_not_black(tmp_path):
    a = np.zeros((40, 40, 4), np.uint8)       # transparent black background
    a[10:30, 10:30] = (0, 0, 0, 255)           # opaque black square = real stitches
    p = tmp_path / "a.png"
    Image.fromarray(a).save(p)
    s = load_source(p)
    assert s.has_alpha and s.opaque[20, 20] and not s.opaque[2, 2]
    f = cell_fields(s.rgba[..., :3], s.opaque, 4, 4)
    stitch = f.coverage >= 0.5
    assert stitch[1:3, 1:3].all() and not stitch[0, 0]


def test_exif_orientation_applied(tmp_path):
    im = Image.new("RGB", (40, 20), "white")
    exif = im.getexif(); exif[0x0112] = 6       # rotate 90
    p = tmp_path / "o.jpg"
    im.save(p, exif=exif)
    s = load_source(p)
    assert (s.width, s.height) == (20, 40)


def test_chart_detector_flags_printed_grid():
    g = np.full((400, 400, 3), 255, np.uint8)
    for i in range(0, 400, 10):
        g[i, :] = 0; g[:, i] = 0
    from threadora.source import detect_chart_grid
    assert detect_chart_grid(g)["is_chart"]
    assert not detect_chart_grid(np.full((400, 400, 3), 200, np.uint8))["is_chart"]


def test_regression_saturated_red_not_matched_to_brown():
    """Bug found on the Cecilia order: crimson lettering (128,1,3) snapped to
    3777 Terra Cotta (brown) by a 0.16 dE00 margin. Near-ties now keep chroma."""
    from threadora.dmc import nearest_dmc
    i, d = nearest_dmc(to_lab(np.array([[128, 1, 3]], np.uint8)))
    assert load_dmc()[i[0]].code in {"815", "814", "816"}
    # an exact DMC color still maps to itself
    f = dmc_by_code()["3777"]
    i, d = nearest_dmc(to_lab(np.array([f.rgb], np.uint8)))
    assert load_dmc()[i[0]].code == "3777"


def test_regression_fonts_bundled_not_linux_path():
    """Windows support: fonts come from the package, not /usr/share/fonts."""
    from threadora.symbols import FONT_DIR
    assert "usr" not in FONT_DIR.parts
    for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif-Bold.ttf"):
        assert (FONT_DIR / f).is_file()


def test_tesseract_cmd_override(monkeypatch, tmp_path):
    import pytesseract
    from threadora import features
    fake = tmp_path / "tesseract.exe"
    fake.write_text("")
    old = pytesseract.pytesseract.tesseract_cmd
    monkeypatch.setenv("THREADORA_TESSERACT_CMD", str(fake))
    try:
        assert features.configure_tesseract() == str(fake)
        assert pytesseract.pytesseract.tesseract_cmd == str(fake)
        assert features.ocr_status()["available"] is False      # not a real binary -> reported, not hidden
    finally:
        monkeypatch.delenv("THREADORA_TESSERACT_CMD")
        pytesseract.pytesseract.tesseract_cmd = old
        features.configure_tesseract()
