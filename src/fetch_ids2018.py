"""Fetch CSE-CIC-IDS2018 processed flow CSVs from the public S3 bucket.

No registration required (AWS Open Data). Records SHA-256 + size for every
file so the manifest can state exactly which bytes the experiments ran on.
"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

BUCKET = "https://cse-cic-ids2018.s3.amazonaws.com/"
PREFIX = "Processed Traffic Data for ML Algorithms/"
FILES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
]

def main():
    out = Path("data/raw/cse-cic-ids2018")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name in FILES:
        dest = out / name
        if name in manifest and dest.exists() and dest.stat().st_size == manifest[name]["bytes"]:
            print(f"[skip] {name}", flush=True)
            continue
        url = BUCKET + urllib.parse.quote(PREFIX + name)
        print(f"[get ] {name}", flush=True)
        t0 = time.time()
        h = hashlib.sha256()
        n = 0
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk); h.update(chunk); n += len(chunk)
        manifest[name] = {
            "bytes": n, "sha256": h.hexdigest(),
            "url": url, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[ok  ] {name}  {n/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)

    print("DONE", len(manifest), "files", flush=True)

if __name__ == "__main__":
    sys.exit(main())
