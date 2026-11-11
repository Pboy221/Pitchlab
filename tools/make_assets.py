"""Regenerate PitchLab's icons and share image (run once; outputs are committed).

    pip install playwright && python tools/make_assets.py

Renders small HTML pages in headless Chromium so the artwork uses the same
Inter font and navy/gold palette as the site.
"""
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "site" / "static"
FONT = (STATIC / "fonts" / "inter-latin-wght-normal.woff2").as_uri()
NAVY, GOLD, GOLD_DIM, BLUE, GREY = "#0A1628", "#F5B82E", "#C8901A", "#6CB8FF", "#94A3BC"

BASE = f"""<style>
@font-face{{font-family:Inter;src:url({FONT}) format('woff2');font-weight:100 900}}
*{{margin:0;box-sizing:border-box}} html,body{{background:transparent}}
body{{font-family:Inter,sans-serif;-webkit-font-smoothing:antialiased}}
</style>"""


def icon_html(size: int, pad: float, radius: float) -> str:
    inner = size * (1 - 2 * pad)
    return BASE + f"""<div style="width:{size}px;height:{size}px;display:grid;place-items:center;
      background:{NAVY if pad else 'transparent'}">
      <div style="width:{inner}px;height:{inner}px;border-radius:{inner * radius}px;
        background:linear-gradient(145deg,{GOLD},{GOLD_DIM});display:grid;place-items:center;
        font-weight:800;font-size:{inner * 0.62}px;color:{NAVY};letter-spacing:-.04em;
        line-height:1;padding-bottom:{inner * 0.02}px">P</div></div>"""


OG = BASE + f"""<div style="width:1200px;height:630px;background:{NAVY};color:#F5F7FB;
  padding:72px 80px;display:flex;flex-direction:column;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:22px">
    <div style="width:84px;height:84px;border-radius:22px;background:linear-gradient(145deg,{GOLD},{GOLD_DIM});
      display:grid;place-items:center;font-weight:800;font-size:54px;color:{NAVY}">P</div>
    <div style="font-weight:800;font-size:56px;letter-spacing:-.02em">PitchLab</div>
  </div>
  <div>
    <div style="font-weight:800;font-size:68px;line-height:1.05;letter-spacing:-.03em;max-width:980px">
      Football predictions,<br><span style="color:{GOLD}">scored in public.</span></div>
    <div style="margin-top:22px;font-size:27px;color:#BAC6D9;max-width:960px;line-height:1.4">
      Match, player, corner and card probabilities, logged before kickoff.</div>
  </div>
  <div>
    <div style="display:flex;height:16px;border-radius:9px;overflow:hidden;width:100%">
      <i style="width:52%;background:{GOLD}"></i><i style="width:24%;background:{GREY}"></i>
      <i style="width:24%;background:{BLUE}"></i></div>
    <div style="margin-top:18px;font-size:23px;color:#8E9FBB;letter-spacing:.02em">
      Premier League · La Liga · Serie A · Bundesliga · Ligue 1</div>
  </div>
</div>"""

FAVICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="{GOLD}"/><stop offset="1" stop-color="{GOLD_DIM}"/></linearGradient></defs>
<rect width="64" height="64" rx="15" fill="url(#g)"/>
<path d="M22 50V14h14.5c7.6 0 12.5 4.6 12.5 11.6S44.1 37.2 36.5 37.2H29.8V50zm7.8-19.3h5.9
c3.6 0 5.6-1.9 5.6-5.1s-2-5.1-5.6-5.1h-5.9z" fill="{NAVY}"/></svg>
"""


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
    (icons / "favicon.svg").write_text(FAVICON_SVG)
    exe = os.environ.get("CHROME")
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe \
            else await p.chromium.launch()
        pg = await b.new_page()
        await shot(pg, OG, STATIC / "og.png", 1200, 630)
        await shot(pg, icon_html(32, 0, 0.24), icons / "favicon-32.png", 32, 32)
        await shot(pg, icon_html(180, 0.0, 0.0), icons / "apple-touch-icon.png", 180, 180)
        await shot(pg, icon_html(192, 0, 0.24), icons / "icon-192.png", 192, 192)
        await shot(pg, icon_html(512, 0, 0.24), icons / "icon-512.png", 512, 512)
        # maskable: art inside the 80% safe zone on a solid navy field
        await shot(pg, icon_html(512, 0.18, 0.24), icons / "icon-maskable-512.png", 512, 512)
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
