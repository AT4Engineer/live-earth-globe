#!/usr/bin/env python3
"""One-shot build: Live geostationary globe: fetch full-disk RGB images, reproject to equirectangular, blend, serve."""
import io, json, math, os, threading, time, urllib.request, datetime, traceback
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("OUT_DIR", os.path.join(ROOT, "site")); os.makedirs(CACHE, exist_ok=True)
W, H = 4096, 2048
REFRESH_S = int(os.environ.get("REFRESH_S", 600))
PORT = int(os.environ.get("PORT", 8080))
SLIDER = "https://slider.cira.colostate.edu"
UA = {"User-Agent": "live-globe/1.0 (educational viewer)"}
BASE_URL = "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg"

SATS = [  # name, sub-sat lon, source
  dict(id="goes19", name="GOES-19 (GOES-East)", lon=-75.2, src="star", path="GOES19"),
  dict(id="goes18", name="GOES-18 (GOES-West)", lon=-137.0, src="star", path="GOES18"),
  dict(id="himawari", name="Himawari-9", lon=140.7, src="slider", path="himawari"),
  dict(id="gk2a", name="GK-2A", lon=128.2, src="slider", path="gk2a"),
  dict(id="meteosat0", name="Meteosat 0° (MTG/MSG)", lon=0.0, src="slider", path="meteosat-0deg"),
  dict(id="meteosat9", name="Meteosat-9 IODC (45.5°E)", lon=45.5, src="slider", path="meteosat-9"),
]
status = {"composite_time": None, "sats": {}}
lock = threading.Lock()

def get(url, timeout=60):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
    return r.read(), r.headers

def fetch(s):
    if s["src"] == "star":
        data, hd = get(f"https://cdn.star.nesdis.noaa.gov/{s['path']}/ABI/FD/GEOCOLOR/1808x1808.jpg")
        t = parsedate_to_datetime(hd["Last-Modified"]).astimezone(datetime.timezone.utc)
        return Image.open(io.BytesIO(data)).convert("RGB"), t.strftime("%Y-%m-%dT%H:%M:%SZ") + " (CDN upload)"
    js, _ = get(f"{SLIDER}/data/json/{s['path']}/full_disk/geocolor/latest_times.json")
    ts = str(json.loads(js)["timestamps_int"][0])
    tiles = {}
    def tile(rc):
        r, c = rc
        u = f"{SLIDER}/data/imagery/{ts[:4]}/{ts[4:6]}/{ts[6:8]}/{s['path']}---full_disk/geocolor/{ts}/01/{r:03d}_{c:03d}.png"
        tiles[rc] = Image.open(io.BytesIO(get(u)[0])).convert("RGB")
    with ThreadPoolExecutor(4) as ex: list(ex.map(tile, [(0,0),(0,1),(1,0),(1,1)]))
    n = tiles[(0,0)].size[0]; img = Image.new("RGB", (2*n, 2*n))
    for (r, c), t in tiles.items(): img.paste(t, (c*n, r*n))
    return img, f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"

# lat/lon grid
lon = (np.arange(W) + 0.5) / W * 360 - 180
lat = 90 - (np.arange(H) + 0.5) / H * 180
LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
REQ, RPOL, HS = 6378.137, 6356.7523, 42164.16
E2 = 1 - RPOL**2 / REQ**2
XEDGE = math.asin(REQ / HS); YEDGE = math.asin(RPOL / HS)

def reproject(img, sublon, star=False):
    a = np.asarray(img).astype(np.float32); h, w, _ = a.shape
    # disk extent from the middle row (images are square & centred)
    row = a[h // 2].max(axis=1); nz = np.where(row > 12)[0]
    x0, x1 = nz[0], nz[-1]; cx = (x0 + x1) / 2; R = (x1 - x0) / 2; cy = (h - 1) / 2
    phic = np.arctan(RPOL**2 / REQ**2 * np.tan(LAT))
    rc = RPOL / np.sqrt(1 - E2 * np.cos(phic)**2)
    dl = LON - math.radians(sublon)
    sx = HS - rc * np.cos(phic) * np.cos(dl); sy = -rc * np.cos(phic) * np.sin(dl); sz = rc * np.sin(phic)
    vis = HS * (HS - sx) >= sy**2 + (REQ**2 / RPOL**2) * sz**2
    x = np.arcsin(np.clip(-sy / np.sqrt(sx**2 + sy**2 + sz**2), -1, 1)); y = np.arctan(sz / sx)
    col = cx + x / XEDGE * R; rowp = cy - y / YEDGE * R
    ok = vis & (col >= 0) & (col < w - 1) & (rowp >= 0) & (rowp < h - 1)
    ci = np.clip(col, 0, w - 1.001); ri = np.clip(rowp, 0, h - 1.001)
    c0 = ci.astype(int); r0 = ri.astype(int); fc = (ci - c0)[..., None]; fr = (ri - r0)[..., None]
    out = (a[r0, c0] * (1-fc) * (1-fr) + a[r0, c0+1] * fc * (1-fr) + a[r0+1, c0] * (1-fc) * fr + a[r0+1, c0+1] * fc * fr)
    if star: ok &= (rowp < cy + 0.86 * R) & (rowp > cy - 0.92 * R)  # mask STAR text banner/logos
    cosd = np.cos(LAT) * np.cos(dl)  # cos of great-circle distance from sub-satellite point
    wgt = np.where(ok, np.clip((cosd - 0.12) / 0.88, 0, 1) ** 2, 0).astype(np.float32)
    return out, wgt

base = None
def load_base():
    global base
    p = os.path.join(CACHE, "bluemarble.jpg")
    if not os.path.exists(p):
        open(p, "wb").write(get(BASE_URL)[0])
    base = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)).astype(np.float32)

def build():
    acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
    def one(s):
        try:
            img, t = fetch(s); out, wg = reproject(img, s["lon"], s["src"] == "star")
            return s, out, wg, t, None
        except Exception as e:
            traceback.print_exc(); return s, None, None, None, f"{type(e).__name__}: {e}"
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, SATS))
    sats = {}
    for s, out, wg, t, err in res:
        prev = status["sats"].get(s["id"], {})
        if err:
            sats[s["id"]] = dict(name=s["name"], lon=s["lon"], ok=False, error=err, time=prev.get("time")); continue
        acc += out * wg[..., None]; wsum += wg
        sats[s["id"]] = dict(name=s["name"], lon=s["lon"], ok=True, time=t, error=None)
    blend = np.clip(wsum / 0.15, 0, 1)[..., None]  # fade to Blue Marble where coverage is thin (poles)
    comp = acc / np.maximum(wsum, 1e-6)[..., None] * blend + base * (1 - blend)
    im = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))
    tmp = os.path.join(CACHE, "composite.tmp.jpg"); im.save(tmp, quality=88); os.replace(tmp, os.path.join(CACHE, "composite.jpg"))
    with lock:
        status["sats"] = sats
        status["composite_time"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json.dump(status, open(os.path.join(CACHE, "status.json"), "w"))
    print("composite built", status["composite_time"], {k: v["ok"] for k, v in sats.items()}, flush=True)


if __name__ == "__main__":
    sp = os.path.join(CACHE, "status.json")
    if os.path.exists(sp):
        try: status.update(json.load(open(sp)))
        except Exception: pass
    load_base(); build()
