"""Regenerate PitchLab's logo, icons and share image (outputs are committed).

    pip install playwright pillow && python tools/make_assets.py

The logo is a small football pitch drawn in cyan on a black tile: outline,
halfway line, centre circle and both penalty boxes. It is plain SVG, so the
same drawing is used inline in the page header, as the favicon, and rendered
to PNG for app icons and the share image.
"""
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "site" / "static"
FONT = (STATIC / "fonts" / "inter-latin-wght-normal.woff2").as_uri()

BG = "#0A0C0E"       # page background
TILE = "#14181C"     # logo tile
CYAN = "#3CC8DC"     # accent
AMBER = "#F0A35E"    # away colour (share image bar)
GREY = "#8C969F"     # draw colour (share image bar)


def logo_svg(tile: bool = True, stroke: float = 1.8) -> str:
    """32x32 pitch mark. tile=False draws the pitch alone (for a custom background)."""
    t = f'<rect width="32" height="32" rx="8" fill="{TILE}"/>' if tile else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">{t}'
            f'<g fill="none" stroke="{CYAN}" stroke-width="{stroke}" stroke-linejoin="round">'
            f'<rect x="5" y="8" width="22" height="16" rx="2"/>'
            f'<path d="M16 8v16"/><circle cx="16" cy="16" r="3.4"/>'
            f'<path d="M5 12.2h3.8v7.6H5M27 12.2h-3.8v7.6H27"/></g></svg>')


BASE = f"""<style>
@font-face{{font-family:Inter;src:url({FONT}) format('woff2');font-weight:100 900}}
*{{margin:0;box-sizing:border-box}} html,body{{background:transparent}}
body{{font-family:Inter,sans-serif;-webkit-font-smoothing:antialiased}}
svg{{display:block}}
</style>"""


def icon_html(size: int, pad: float = 0.0, square: bool = False) -> str:
    """App icon. pad > 0 puts the mark inside a solid field (maskable safe zone)."""
    inner = size * (1 - 2 * pad)
    svg = logo_svg()
    if square:                     # iOS rounds the corners itself
        svg = svg.replace('rx="8" fill', 'rx="0" fill')
    return BASE + f"""<div style="width:{size}px;height:{size}px;display:grid;place-items:center;
      background:{TILE if pad else 'transparent'}">
      <div style="width:{inner}px;height:{inner}px">{svg.replace('<svg ', f'<svg width="{inner}" height="{inner}" ')}</div>
    </div>"""


OG = BASE + f"""<div style="width:1200px;height:630px;background:{BG};color:#F2F5F7;
  padding:72px 80px;display:flex;flex-direction:column;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:22px">
    {logo_svg().replace('<svg ', '<svg width="84" height="84" ')}
    <div style="font-weight:800;font-size:56px;letter-spacing:-.02em">PitchLab</div>
  </div>
  <div>
    <div style="font-weight:800;font-size:68px;line-height:1.05;letter-spacing:-.03em;max-width:980px">
      Football predictions,<br><span style="color:{CYAN}">scored in public.</span></div>
    <div style="margin-top:22px;font-size:27px;color:#B1BAC3;max-width:960px;line-height:1.4">
      Match, player, corner and card probabilities, logged before kickoff.</div>
  </div>
  <div>
    <div style="display:flex;height:16px;border-radius:9px;overflow:hidden;width:100%">
      <i style="width:52%;background:{CYAN}"></i><i style="width:24%;background:{GREY}"></i>
      <i style="width:24%;background:{AMBER}"></i></div>
    <div style="margin-top:18px;font-size:23px;color:#87929D;letter-spacing:.02em">
      Premier League · La Liga · Serie A · Bundesliga · Ligue 1</div>
  </div>
</div>"""


async def shot(page, html, path, w, h):
    await page.set_viewport_size({"width": w, "height": h})
    await page.set_content(html)
    await page.evaluate("document.fonts.ready")
    await page.wait_for_timeout(150)
    await page.screenshot(path=str(path), omit_background=True,
                          clip={"x": 0, "y": 0, "width": w, "height": h})


async def main():
    icons = STATIC / "icons"
    icons.mkdir(parents=True, exist_ok=True)
    (icons / "favicon.svg").write_text(logo_svg() + "\n")
    (icons / "logo.svg").write_text(logo_svg() + "\n")
    exe = os.environ.get("CHROME")
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe \
            else await p.chromium.launch()
        pg = await b.new_page()
        await shot(pg, OG, STATIC / "og.png", 1200, 630)
        await shot(pg, icon_html(32), icons / "favicon-32.png", 32, 32)
        await shot(pg, icon_html(180, square=True), icons / "apple-touch-icon.png", 180, 180)
        await shot(pg, icon_html(192), icons / "icon-192.png", 192, 192)
        await shot(pg, icon_html(512), icons / "icon-512.png", 512, 512)
        # maskable: art inside the 80% safe zone on a solid field
        await shot(pg, icon_html(512, pad=0.14), icons / "icon-maskable-512.png", 512, 512)
        await b.close()
    _shrink()


def _shrink():
    """Palette-compress the PNGs (flat artwork, so no visible loss)."""
    from PIL import Image
    for f in [STATIC / "og.png", *sorted((STATIC / "icons").glob("*.png"))]:
        im = Image.open(f).convert("RGBA")
        im.quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(f, optimize=True)


if __name__ == "__main__":
    asyncio.run(main())
