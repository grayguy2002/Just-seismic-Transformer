"""Download KiK-net event ZIPs via Clash proxy. Simple, reliable, single-threaded."""
import base64, time, urllib.request
from pathlib import Path

ZIP_DIR = Path("/Volumes/MyPassport/EEW/data/kiknet_measured_vs30_pwave_v1/raw_zips")
PROXY = "http://127.0.0.1:7897"
TOKEN = base64.b64encode(b"garyguy2002:5865147Gg").decode("ascii")

def get_events():
    import pandas as pd
    df = pd.read_csv("/Volumes/MyPassport/EEW/data/kiknet_measured_vs30_pwave_v1/nied_search/nied_kiknet_record_candidates.csv")
    events = df[["eqid", "source_magnitude"]].drop_duplicates()
    events["eqid_str"] = events["eqid"].apply(lambda x: str(int(x)))
    events["mag_float"] = pd.to_numeric(events["source_magnitude"], errors="coerce")
    events = events.sort_values("mag_float", ascending=False)
    print(f"{len(events)} unique events to download")
    return events

def download_one(eqid):
    out = ZIP_DIR / f"{eqid}_ascii.zip"
    if out.exists():
        return "exists", out.stat().st_size
    year, month = eqid[:4], eqid[4:6]
    url = f"https://www.kyoshin.bosai.go.jp/kyoshin/download/kik/zip/{year}/{month}/{eqid}/{eqid}_ascii.zip"
    proxy_handler = urllib.request.ProxyHandler({"https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    req = urllib.request.Request(url, headers={
        "User-Agent": "EEW-KiKnet/1.0", "Authorization": f"Basic {TOKEN}"})
    t0 = time.time()
    try:
        resp = opener.open(req, timeout=300)
        data = resp.read()
    except urllib.error.HTTPError as e:
        return f"HTTP_{e.code}", 0
    tmps = str(out) + ".part"
    with open(tmps, "wb") as f:
        f.write(data)
    Path(tmps).rename(out)
    elapsed = time.time() - t0
    return "ok", len(data), elapsed

events = get_events()
errors = 0
total_bytes = 0
t_start = time.time()
total = len(events)

for i, (_, row) in enumerate(events.iterrows()):
    eqid = row["eqid_str"]
    mag = row["mag_float"]
    try:
        status = download_one(eqid)
    except Exception as e:
        print(f"[{i+1}/{total}] M{mag:.1f} {eqid}  ERROR: {e}")
        errors += 1
        continue
    if isinstance(status, tuple) and len(status) == 3 and status[0] == "ok":
        st, sz, et = status
        total_bytes += sz
        done = i + 1
        total = len(events)
        elapsed = time.time() - t_start
        rate = total_bytes / max(elapsed, 1)
        eta = (total - done) * (elapsed / max(done, 1))
        print(f"[{done}/{total}] M{mag:.1f} {eqid}  {sz//1024}KB {et:.0f}s  "
              f"{rate/1024:.0f}KB/s  ETA {eta/60:.0f}min")
    else:
        errors += 1
        if errors > 5:
            print(f"Too many errors ({errors}), stopping")
            break
    time.sleep(0.3)

print(f"\nDone: {i+1-errors}/{len(events)} ok, {errors} errors, "
      f"{total_bytes/1024/1024:.0f}MB in {(time.time()-t_start)/60:.0f}min")
