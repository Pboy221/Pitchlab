"""Everything the site needs besides index.html: static assets, the web app
manifest, the service worker, a 404 page, robots.txt and sitemap.xml."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "site" / "static"

DESCRIPTION = ("Free, transparent football forecasts for the Premier League, La Liga, "
               "Serie A, Bundesliga and Ligue 1.")

MANIFEST = {
    "name": "PitchLab",
    "short_name": "PitchLab",
    "description": DESCRIPTION,
    "start_url": "./",
    "scope": "./",
    "display": "standalone",
    "background_color": "#0A0C0E",
    "theme_color": "#0A0C0E",
    "categories": ["sports"],
    "icons": [
        {"src": "icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "maskable"},
    ],
}

# Network first for the page and data (forecasts must be current), cache as the
# offline fallback; cache first for fonts and icons, which never change in place.
SERVICE_WORKER = """// PitchLab service worker, build __BUILD__
const PAGES = 'pitchlab-pages-v1';
const ASSETS = 'pitchlab-assets-v1';
const PRECACHE = ['./', 'fonts/inter-latin-wght-normal.woff2', 'icons/icon-192.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(ASSETS).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys
    .filter(k => ![PAGES, ASSETS].includes(k)).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  const isAsset = /\\/(fonts|icons)\\//.test(req.url);
  if (isAsset) {
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
      const copy = res.clone(); caches.open(ASSETS).then(c => c.put(req, copy)); return res;
    })));
    return;
  }
  e.respondWith(fetch(req).then(res => {
    if (res.ok) { const copy = res.clone(); caches.open(PAGES).then(c => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req).then(hit => hit || caches.match('./'))));
});
"""

NOT_FOUND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Page not found · PitchLab</title>
<meta name="robots" content="noindex">
<link rel="icon" href="__SITE__icons/favicon.svg" type="image/svg+xml">
<style>
@font-face{font-family:Inter;src:url(__SITE__fonts/inter-latin-wght-normal.woff2) format('woff2');font-weight:100 900;font-display:swap}
html{background:#0A0C0E}body{margin:0;min-height:100vh;display:grid;place-items:center;
font-family:Inter,system-ui,sans-serif;color:#F2F5F7;text-align:center;padding:24px}
.m{width:56px;height:56px;border-radius:15px;margin:0 auto 22px;display:grid;place-items:center;
background:#1B2026;box-shadow:inset 0 0 0 1px rgba(60,200,220,.45);color:#3CC8DC;font-weight:800;font-size:32px}
h1{font-size:26px;margin:0 0 10px;letter-spacing:-.02em}p{color:#B1BAC3;margin:0 0 24px;font-size:16px}
a{display:inline-block;background:#3CC8DC;color:#0A0C0E;font-weight:700;text-decoration:none;
padding:12px 22px;border-radius:11px}a:focus-visible{outline:2px solid #F2F5F7;outline-offset:3px}
</style></head><body><main><div class="m">P</div><h1>Offside.</h1>
<p>That page doesn't exist. The forecasts are this way.</p>
<a href="__SITE__">Back to PitchLab</a></main></body></html>
"""


def write(out: Path, build_id: str, today: str) -> None:
    if STATIC.exists():
        shutil.copytree(STATIC, out, dirs_exist_ok=True)
    site = C.SITE_URL
    (out / "manifest.webmanifest").write_text(json.dumps(MANIFEST, indent=1))
    (out / "sw.js").write_text(SERVICE_WORKER.replace("__BUILD__", build_id))
    (out / "404.html").write_text(NOT_FOUND.replace("__SITE__", site))
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {site}sitemap.xml\n")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{site}</loc><lastmod>{today}</lastmod><changefreq>daily</changefreq></url>\n"
        "</urlset>\n")
    (out / ".nojekyll").write_text("")
