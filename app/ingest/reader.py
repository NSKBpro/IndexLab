# app/ingest/reader.py
from pathlib import Path
import os, subprocess, tempfile, shutil
import pandas as pd
from bs4 import BeautifulSoup

MAX_CHM_FILES = 10000          # hard safety cap
MAX_CHM_BYTES = 100 * 1024**2  # 100 MB of HTML processed

def _html_to_text(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        txt = soup.get_text(separator="\n", strip=True)
        # collapse extra blanks
        lines = [ln.strip() for ln in txt.splitlines()]
        return "\n".join([ln for ln in lines if ln])
    except Exception:
        return html_bytes.decode("utf-8", errors="ignore")

# ---------- CHM via Windows hh.exe ----------
def _find_hh_exe() -> str | None:
    # Typical locations; allow override
    cand = [
        os.environ.get("HH_EXE") or "",
        r"C:\Windows\hh.exe",
        r"C:\Windows\System32\hh.exe",
        r"C:\Windows\SysWOW64\hh.exe",
    ]
    for c in cand:
        if c and os.path.exists(c):
            return c
    # Try via PATH
    for name in ["hh.exe"]:
        try:
            p = shutil.which(name)
            if p:
                return p
        except Exception:
            pass
    return None

def _read_chm_with_hh(path: Path) -> pd.DataFrame | None:
    hh = _find_hh_exe()
    if not hh:
        return None
    outdir = Path(tempfile.mkdtemp(prefix="chm_hh_"))
    try:
        # Syntax: hh.exe -decompile <outdir> <file.chm>
        # (No stdout on success; returns immediately)
        proc = subprocess.run(
            [hh, "-decompile", str(outdir), str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        # Even if return code isn't perfect, files are usually dumped
        rows, count, size_acc = [], 0, 0
        for root, _dirs, files in os.walk(outdir):
            for fn in files:
                if count >= MAX_CHM_FILES:
                    break
                low = fn.lower()
                if not (low.endswith(".htm") or low.endswith(".html") or low.endswith(".hhc") or low.endswith(".hhk")):
                    continue
                p = Path(root) / fn
                try:
                    b = p.read_bytes()
                except Exception:
                    continue
                size_acc += len(b)
                if size_acc > MAX_CHM_BYTES:
                    break
                text = _html_to_text(b)
                if text.strip():
                    rel = str(Path(root).relative_to(outdir) / fn)
                    rows.append({"path": rel, "text": text})
                    count += 1
            if count >= MAX_CHM_FILES or size_acc > MAX_CHM_BYTES:
                break
        return pd.DataFrame(rows) if rows else pd.DataFrame({"text": []})
    finally:
        # Clean up extracted files
        shutil.rmtree(outdir, ignore_errors=True)

# ---------- CHM via 7-Zip fallback (Windows/macOS/Linux) ----------
def _which_7z() -> str | None:
    env = os.environ.get("SEVEN_ZIP") or os.environ.get("SEVENZIP") or os.environ.get("Z7")
    if env and os.path.exists(env):
        return env
    # PATH search + common Windows paths
    for name in ["7z", "7za", "7zz", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
        p = shutil.which(name) if os.path.basename(name) == name else name
        try:
            if p and subprocess.run([p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2):
                return p
        except Exception:
            pass
    return None

def _read_chm_with_7z(path: Path) -> pd.DataFrame | None:
    seven = _which_7z()
    if not seven:
        return None
    outdir = Path(tempfile.mkdtemp(prefix="chm_7z_"))
    try:
        subprocess.run([seven, "x", str(path), f"-o{outdir}", "-y"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        rows, count, size_acc = [], 0, 0
        for root, _dirs, files in os.walk(outdir):
            for fn in files:
                if count >= MAX_CHM_FILES:
                    break
                low = fn.lower()
                if not (low.endswith(".htm") or low.endswith(".html") or low.endswith(".hhc") or low.endswith(".hhk")):
                    continue
                p = Path(root) / fn
                try:
                    b = p.read_bytes()
                except Exception:
                    continue
                size_acc += len(b)
                if size_acc > MAX_CHM_BYTES:
                    break
                text = _html_to_text(b)
                if text.strip():
                    rel = str(Path(root).relative_to(outdir) / fn)
                    rows.append({"path": rel, "text": text})
                    count += 1
            if count >= MAX_CHM_FILES or size_acc > MAX_CHM_BYTES:
                break
        return pd.DataFrame(rows) if rows else pd.DataFrame({"text": []})
    finally:
        shutil.rmtree(outdir, ignore_errors=True)

# ---------- PUBLIC: read_any ----------
def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".chm":
        df = _read_chm_with_hh(path)
        if df is None:
            df = _read_chm_with_7z(path)
        if df is None:
            raise ValueError(
                "CHM parsing requires Windows 'hh.exe' (built-in) or 7-Zip on PATH/SEVEN_ZIP. "
                "Install 7-Zip or set HH_EXE to the full path of hh.exe."
            )
        return df  # has 'text' (and 'path')

    # --- your existing handlers below ---
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".txt", ".md"):
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        return pd.DataFrame({"text": [txt]})

    raise ValueError(f"Unsupported file type: {suffix}")
