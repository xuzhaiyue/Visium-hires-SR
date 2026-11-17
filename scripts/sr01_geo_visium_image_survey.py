"""Step 01 — survey of H&E image availability in human Visium series on GEO (reads file lists only; no data are downloaded).

Question: how many public Visium studies deposit only the downsampled tissue_hires_image.png and no full-resolution image?
Method:
  1. Query GEO via E-utilities for all GSE matching "visium" + Homo sapiens;
  2. for each GSE read the suppl/ directory listing and filelist.txt (files inside RAW.tar with sizes);
  3. classify image files by name / extension / size and assign each GSE a summary category:
       fullres      full-resolution image: header readable with longest side >= 10000 px, or header unreadable but single file >= 100 MB
       fullres_unc  possibly full resolution: longest side 4000-10000 px, or header unreadable and 20-100 MB
       hires_only   only tissue_hires/lowres png, CytAssist images or small images
       bundle_only  images may be packed in tar.gz / zip; cannot be judged from the listing
       no_image     listing contains only expression matrices etc., no image
  For candidate full-resolution images, fetch only the first 1 MB with an HTTP Range request and read the pixel size from the header (TIFFs with the IFD at the end cannot be read; fall back to size).
  On a 6.5 mm capture area a longest side of 10000 px corresponds to about <= 0.65 um/pixel.
Note: full-resolution images outside GEO (Zenodo, figshare, 10x website, etc.) are not counted; GSEs whose summary mentions such links are flagged separately.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import struct
import zlib
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"
OUT = Path(__file__).resolve().parents[1] / "results" / "sr_survey"

FULLRES_EXT = (".tif", ".tiff", ".btf", ".svs", ".ndpi", ".qptiff", ".scn", ".czi", ".vsi", ".mrxs", ".jp2")
RASTER_EXT = (".jpg", ".jpeg", ".png")
LOWRES_PAT = re.compile(r"(hires|lowres|fiducial|detected_tissue|aligned_tissue|cytassist)", re.I)
EXTERNAL_PAT = re.compile(r"(?:zenodo|figshare|mendeley|dryad|synapse|10xgenomics\.com|github\.com|huggingface)", re.I)
SIZE_UNITS = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def get(url: str, retries: int = 4) -> str:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return ""


def search_gse(term: str) -> list[str]:
    q = urllib.parse.urlencode({"db": "gds", "term": term, "retmax": 10000, "retmode": "json"})
    ids = json.loads(get(f"{EUTILS}/esearch.fcgi?{q}"))["esearchresult"]["idlist"]
    rows = []
    for i in range(0, len(ids), 200):
        q = urllib.parse.urlencode({"db": "gds", "id": ",".join(ids[i:i + 200]), "retmode": "json"})
        res = json.loads(get(f"{EUTILS}/esummary.fcgi?{q}"))["result"]
        for uid in res["uids"]:
            d = res[uid]
            rows.append({"gse": "GSE" + str(d["gse"]), "title": d["title"], "summary": d["summary"],
                         "n_samples": d["n_samples"], "pdat": d["pdat"], "gpl": d["gpl"]})
        time.sleep(0.4)
    return rows


def gse_dir(gse: str) -> str:
    stub = gse[:-3] + "nnn" if len(gse) > 6 else "GSEnnn"
    return f"{FTP}/{stub}/{gse}/suppl"


def list_files(gse: str) -> list[tuple[str, float]]:
    """Return (file name, bytes): directory listing plus files inside RAW.tar from filelist.txt."""
    base = gse_dir(gse)
    html = get(base + "/")
    files = []
    for name, size in re.findall(r'<a href="([^"/?]+)">[^<]+</a>\s+\S+ \S+\s+([\d.]+[KMGT]?)', html):
        m = re.match(r"([\d.]+)([KMGT]?)", size)
        files.append((name, float(m.group(1)) * SIZE_UNITS[m.group(2)]))
    if any(n == "filelist.txt" for n, _ in files):
        for line in get(base + "/filelist.txt").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0] == "File":
                files.append((parts[1], float(parts[3])))
    return [(n, s) for n, s in files if n != "filelist.txt"]


def classify_file(name: str, size: float) -> str | None:
    low = name.lower().removesuffix(".gz")
    if low.endswith(FULLRES_EXT + RASTER_EXT):
        if LOWRES_PAT.search(low):
            return "hires_png"
        return "candidate" if size > 2e6 else "other_img"
    if low.endswith((".tar.gz", ".tgz", ".zip", ".tar")) and not low.endswith("_raw.tar") \
            and not re.search(r"(matrix|barcodes|features|h5ad|rds|loupe)", low):
        return "bundle"
    return None


def file_url(gse: str, name: str) -> str:
    m = re.match(r"(GSM\d+)_", name)
    if not m:
        return f"{gse_dir(gse)}/{name}"
    gsm = m.group(1)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{urllib.parse.quote(name)}"


def _tiff_dims(b: bytes) -> tuple[int, int] | None:
    bo = {b"II": "<", b"MM": ">"}.get(b[:2])
    if bo is None:
        return None
    big = struct.unpack(bo + "H", b[2:4])[0] == 43
    off = struct.unpack(bo + ("Q" if big else "I"), b[8:16] if big else b[4:8])[0]
    hdr, ent = (8, 20) if big else (2, 12)
    if off + hdr > len(b):
        return None
    n = struct.unpack(bo + ("Q" if big else "H"), b[off:off + hdr])[0]
    tags = {}
    for i in range(n):
        e = b[off + hdr + ent * i: off + hdr + ent * (i + 1)]
        if len(e) < ent:
            return None
        tag, typ = struct.unpack(bo + "HH", e[:4])
        val = e[12:] if big else e[8:]
        tags[tag] = struct.unpack(bo + "H", val[:2])[0] if typ == 3 else struct.unpack(bo + "I", val[:4])[0]
    return (tags[256], tags[257]) if 256 in tags and 257 in tags else None


def probe_dims(url: str) -> int | None:
    """Fetch only the first 1 MB and return the longest side in pixels; None if unreadable."""
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        b = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw) if url.endswith(".gz") else raw
        d = _tiff_dims(b)
        if d is None and b[:4] != b"II*\x00" and b[:4] != b"MM\x00*":
            Image.MAX_IMAGE_PIXELS = None
            d = Image.open(io.BytesIO(b)).size
        return max(d) if d else None
    except Exception:
        return None


def judge(longest: int | None, size: float) -> str:
    if longest is not None:
        return "fullres" if longest >= 10000 else "fullres_unc" if longest >= 4000 else "small"
    return "fullres" if size >= 100e6 else "fullres_unc" if size >= 20e6 else "small"


def survey(gse: str) -> dict:
    files = list_files(gse)
    cls = [(n, s, classify_file(n, s)) for n, s in files]
    cands = sorted([(n, s) for n, s, c in cls if c == "candidate"], key=lambda x: -x[1])
    # probe at most the 3 largest candidate files per GSE and keep the best call
    probed = [(n, s, probe_dims(file_url(gse, n))) for n, s in cands[:3]]
    verdicts = [judge(px, s) for n, s, px in probed]
    cnt = {k: sum(1 for *_, c in cls if c == k) for k in ("candidate", "hires_png", "bundle")}
    if "fullres" in verdicts:
        cat = "fullres"
    elif "fullres_unc" in verdicts:
        cat = "fullres_unc"
    elif cnt["hires_png"] or cands or any(c == "other_img" for *_, c in cls):
        cat = "hires_only"
    elif cnt["bundle"]:
        cat = "bundle_only"
    else:
        cat = "no_image"
    longest = [px for *_, px in probed if px]
    return {"gse": gse, "category": cat, "n_files": len(files), **{f"n_{k}": v for k, v in cnt.items()},
            "max_candidate_MB": cands[0][1] / 1e6 if cands else None,
            "max_longest_px": max(longest) if longest else None,
            "probed": "; ".join(f"{n}|{s/1e6:.0f}MB|{px}px" for n, s, px in probed)[:400],
            "bundle_examples": "; ".join(n for n, _, c in cls if c == "bundle")[:300],
            "max_bundle_MB": max([s for _, s, c in cls if c == "bundle"], default=0) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default='(visium) AND "Homo sapiens"[Organism] AND gse[Entry Type]')
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.DataFrame(search_gse(args.term)).drop_duplicates("gse")
    print(f"retrieved {len(meta)} GSE")
    with ThreadPoolExecutor(args.workers) as ex:
        res = pd.DataFrame(list(ex.map(survey, meta.gse)))
    df = meta.merge(res, on="gse")
    text = df.title + " " + df.summary
    df["mentions_visium"] = text.str.contains(r"visium|spatial transcriptom", case=False)
    df["external_link"] = df.summary.str.contains(EXTERNAL_PAT.pattern, case=False)
    df["pdac"] = text.str.contains(r"pancrea|PDAC", case=False)
    df.drop(columns="summary").to_csv(OUT / "geo_visium_image_survey.csv", index=False)

    v = df[df.mentions_visium]
    print(f"\ntitle/summary explicitly mentions Visium/spatial transcriptomics: {len(v)} GSE")
    print(v.category.value_counts().to_string())
    print("\nof which pancreas-related:")
    print(v[v.pdac].category.value_counts().to_string())
    print("\nhires_only with an external repository mentioned in the summary:", int(v[v.category == "hires_only"].external_link.sum()))


if __name__ == "__main__":
    main()
