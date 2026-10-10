#!/usr/bin/env python3
"""One-shot build: fetch max-res geostationary full disks, reproject to a high-res
equirectangular composite (strip by strip), and write a WebP tile pyramid
(GeographicTilingScheme: level 0 = 2x1 tiles, y=0 at north) + status.json into site/."""
import io, json, math, os, shutil, time, urllib.request, datetime, traceback
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "site"))
WORK = os.environ.get("WORK_DIR", os.path.join(ROOT, "work"))
MAXLEVEL = int(os.environ.get("MAXLEVEL", 5))          # 5 -> 32768 x 16384 composite
TILE = 512
W, H = 2 * TILE * 2**MAXLEVEL, TILE * 2**MAXLEVEL
STRIP = TILE
MINCOS = 0.08      # ignore view zenith beyond ~85 deg (limb)
FEATHER = 3e-4     # softmax temperature on cos(distance): seam feather of only a few px
SLIDER = "https://slider.cira.colostate.edu"
UA = {"User-Agent": "live-earth-globe/2.0 (github.com/AT4Engineer/live-earth-globe)"}
BASE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x10800.jpg"
SATS = [
  dict(id="goes19", name="GOES-19 (GOES-East)", lon=-75.0,  # STAR GOES-19 imagery is gridded at nominal 75.0W (verified vs coastlines)
       src="star", path="GOES19"),
  dict(id="goes18", name="GOES-18 (GOES-West)", lon=-137.0, src="star", path="GOES18"),
  dict(id="himawari", name="Himawari-9", lon=140.7, src="slider", path="himawari", zoom=4),
  dict(id="gk2a", name="GK-2A", lon=128.2, src="slider", path="gk2a", zoom=4),
  dict(id="meteosat0", name="Meteosat 0° (MTG/MSG)", lon=0.0, src="slider", path="meteosat-0deg", zoom=3, R=0.4885, bonus=-0.05),
  dict(id="meteosat9", name="Meteosat-9 IODC (45.5°E)", lon=45.5, src="slider", path="meteosat-9", zoom=3, R=0.4885, bonus=-0.05),
]
REQ, RPOL, HS = 6378.137, 6356.7523, 42164.16
E2 = 1 - RPOL**2 / REQ**2
XEDGE = math.asin(REQ / HS); YEDGE = math.asin(RPOL / HS)

def get(url, timeout=120, tries=3):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
            return r.read(), r.headers
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 + 3 * i)

def fetch(s):
    if s["src"] == "star":
        data, hd = get(f"https://cdn.star.nesdis.noaa.gov/{s['path']}/ABI/FD/GEOCOLOR/10848x10848.jpg", timeout=300)
        t = parsedate_to_datetime(hd["Last-Modified"]).astimezone(datetime.timezone.utc)
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), t.strftime("%Y-%m-%dT%H:%M:%SZ") + " (CDN upload)"
    js, _ = get(f"{SLIDER}/data/json/{s['path']}/full_disk/geocolor/latest_times.json")
    ts = str(json.loads(js)["timestamps_int"][0]); z = s["zoom"]; n = 2**z
    base = f"{SLIDER}/data/imagery/{ts[:4]}/{ts[4:6]}/{ts[6:8]}/{s['path']}---full_disk/geocolor/{ts}/{z:02d}"
    first = Image.open(io.BytesIO(get(f"{base}/000_000.png")[0])).convert("RGB"); ts_ = first.size[0]
    img = np.zeros((n * ts_, n * ts_, 3), np.uint8); img[:ts_, :ts_] = np.asarray(first)
    def tile(rc):
        r, c = rc
        t = Image.open(io.BytesIO(get(f"{base}/{r:03d}_{c:03d}.png")[0])).convert("RGB")
        img[r*ts_:(r+1)*ts_, c*ts_:(c+1)*ts_] = np.asarray(t)
    with ThreadPoolExecutor(16) as ex: list(ex.map(tile, [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]))
    return img, f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"

def disk_geom(a, star, R=None):
    h, w, _ = a.shape
    if R: return dict(cx=(w - 1) / 2, cy=(h - 1) / 2, R=R * w, star=star)
    row = a[h // 2].max(axis=1); nz = np.where(row > 12)[0]
    x0, x1 = nz[0], nz[-1]
    return dict(cx=(x0 + x1) / 2, cy=(h - 1) / 2, R=(x1 - x0) / 2, star=star)

LONS = np.radians((np.arange(W) + 0.5) / W * 360 - 180).astype(np.float32)

def sample(a, g, sublon, lat1d):
    """Return (rgb float32, weight) for strip rows lat1d and the columns this sat can see."""
    dl_all = (np.degrees(LONS) - sublon + 180) % 360 - 180
    cols = np.where(np.abs(dl_all) < 82)[0]
    if len(cols) == 0: return None
    LAT = np.radians(lat1d).astype(np.float32)[:, None]; dl = np.radians(dl_all[cols]).astype(np.float32)[None, :]
    phic = np.arctan(RPOL**2 / REQ**2 * np.tan(LAT))
    rc = RPOL / np.sqrt(1 - E2 * np.cos(phic)**2)
    cp = rc * np.cos(phic)
    sx = HS - cp * np.cos(dl); sy = -cp * np.sin(dl); sz = np.broadcast_to(rc * np.sin(phic), sx.shape)
    vis = HS * (HS - sx) >= sy**2 + (REQ**2 / RPOL**2) * sz**2
    x = np.arcsin(np.clip(-sy / np.sqrt(sx**2 + sy**2 + sz**2), -1, 1)); y = np.arctan(sz / sx)
    h, w, _ = a.shape
    col = g["cx"] + x / XEDGE * g["R"]; rowp = g["cy"] - y / XEDGE * g["R"]  # square pixels in scan-angle space
    ok = vis & (col >= 0) & (col < w - 1) & (rowp >= 0) & (rowp < h - 1)
    if g["star"]: ok &= (rowp < g["cy"] + 0.86 * g["R"]) & (rowp > g["cy"] - 0.92 * g["R"])
    cosd = np.cos(LAT) * np.cos(dl)
    score = np.where(ok & (cosd > MINCOS), cosd, -np.inf).astype(np.float32)
    if not np.isfinite(score).any(): return None
    ci = np.clip(col, 0, w - 1.001); ri = np.clip(rowp, 0, h - 1.001)
    c0 = ci.astype(np.int32); r0 = ri.astype(np.int32)
    fc = (ci - c0).astype(np.float32)[..., None]; fr = (ri - r0).astype(np.float32)[..., None]
    out = (a[r0, c0] * (1 - fc) + a[r0, c0 + 1] * fc) * (1 - fr) + (a[r0 + 1, c0] * (1 - fc) + a[r0 + 1, c0 + 1] * fc) * fr
    return cols, np.clip(out + 0.5, 0, 255).astype(np.uint8), score

def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    status = {"sats": {}}
    sp = os.path.join(OUT, "status.json")
    if os.path.exists(sp):
        try: status = json.load(open(sp))
        except Exception: pass
    bp = os.path.join(WORK, "bluemarble.jpg")
    if not os.path.exists(bp): open(bp, "wb").write(get(BASE_URL, timeout=300)[0])
    base = np.asarray(Image.open(bp).convert("RGB")); bh, bw, _ = base.shape

    def one(s):
        try:
            a, t = fetch(s); return s, a, disk_geom(a, s["src"] == "star", s.get("R")), t, None
        except Exception as e:
            traceback.print_exc(); return s, None, None, None, f"{type(e).__name__}: {e}"
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, SATS))
    t_fetch = time.time() - t0
    sats = {}
    for s, a, g, t, err in res:
        prev = status.get("sats", {}).get(s["id"], {})
        sats[s["id"]] = dict(name=s["name"], lon=s["lon"], ok=err is None, time=t or prev.get("time"), error=err,
                             src_px=None if a is None else f"{a.shape[1]}x{a.shape[0]}")
        print(s["id"], "ok" if err is None else err, sats[s["id"]]["src_px"], flush=True)

    comp = np.empty((H, W, 3), np.uint8)
    bx = ((np.arange(W) + 0.5) / W * bw).astype(np.int32).clip(0, bw - 1)
    def strip(r0):
        rows = np.arange(r0, r0 + STRIP)
        lat1d = 90 - (rows + 0.5) / H * 180
        parts = []
        best = np.full((STRIP, W), -np.inf, np.float32); raw = np.full((STRIP, W), -1.0, np.float32)
        for s, a, g, t, err in res:
            if a is None: continue
            r = sample(a, g, s["lon"], lat1d)
            if r is None: continue
            cols, out, sc = r
            raw[:, cols] = np.maximum(raw[:, cols], np.where(np.isfinite(sc), sc, -1))
            sc += s.get("bonus", 0.0)
            parts.append((cols, out, sc)); best[:, cols] = np.maximum(best[:, cols], sc)
        acc = np.zeros((STRIP, W, 3), np.float32); ws = np.zeros((STRIP, W), np.float32)
        fb = np.where(np.isfinite(best), best, 0)
        for cols, out, sc in parts:   # single best satellite per pixel, narrow softmax feather at seams
            wg = np.exp(np.minimum(sc - fb[:, cols], 0) / FEATHER); wg[~np.isfinite(sc)] = 0
            acc[:, cols] += out * wg[..., None]; ws[:, cols] += wg
        del parts
        by = ((rows + 0.5) / H * bh).astype(np.int32).clip(0, bh - 1)
        bm = base[by][:, bx].astype(np.float32)
        blend = np.clip((raw - MINCOS) / 0.03, 0, 1)[..., None]  # Blue Marble only beyond satellite coverage
        comp[r0:r0 + STRIP] = np.clip(acc / np.maximum(ws, 1e-6)[..., None] * blend + bm * (1 - blend), 0, 255).astype(np.uint8)
    with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex: list(ex.map(strip, range(0, H, STRIP)))
    del res, base
    t_proj = time.time() - t0 - t_fetch
    print(f"reprojected {W}x{H} in {t_proj:.0f}s", flush=True)

    tiles = os.path.join(OUT, "tiles"); tmpd = tiles + ".new"
    shutil.rmtree(tmpd, ignore_errors=True)
    img = Image.fromarray(comp); del comp
    def save_level(z, im):
        nx, ny = im.size[0] // TILE, im.size[1] // TILE
        def w(x):
            os.makedirs(f"{tmpd}/{z}/{x}", exist_ok=True)
            for y in range(ny):
                im.crop((x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE)).save(f"{tmpd}/{z}/{x}/{y}.webp", quality=82, method=3)
        with ThreadPoolExecutor(os.cpu_count()) as ex: list(ex.map(w, range(nx)))
    for z in range(MAXLEVEL, -1, -1):
        save_level(z, img)
        if z: img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS if img.size[0] <= 8192 else Image.BILINEAR)
        if z == 3: img.save(os.path.join(OUT, "preview.jpg"), quality=85)  # 4096x2048 fallback
    shutil.rmtree(tiles, ignore_errors=True); os.replace(tmpd, tiles)
    t_tiles = time.time() - t0 - t_fetch - t_proj
    status = dict(sats=sats, composite_time=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  width=W, height=H, maxLevel=MAXLEVEL, tileSize=TILE,
                  build_seconds=dict(fetch=round(t_fetch), reproject=round(t_proj), tiles=round(t_tiles), total=round(time.time() - t0)))
    json.dump(status, open(sp, "w"), indent=1)
    print("done", json.dumps(status["build_seconds"]), flush=True)

if __name__ == "__main__":
    main()
