"""galaxia-rs engine of the Galaxia fleet runner (docs/runbooks/galaxia-fleet.md, section "galaxia-rs engine").

galaxia-rs is the Rust port of Galaxia (``~/exonit/galaxia-rs``).  Its ``batch`` mode loads the ~5 GB of
resident data ONCE and then generates many 0.1 deg^2 fields (the same field definition as ``fpp_background.generate_raw``:
photoSys DCMC, J / J-Ks, intrinsic J <= 22, seed ``seed_for_tic``):

    galaxia-rs batch --csv <tic,ra,dec> --out-dir <dir> --data-dir <GalaxiaData> --backend cpu|cuda|opencl [--tmag-ceiling X]

What it writes (read from ``src/batch.rs`` / ``src/generator.rs::write_output_ebf``, not guessed):

* ``<out-dir>/gx_<tic>.ebf`` per field, EBF v1.1 (``src/ebf.rs``): a sequence of datasets, each a header (56 fixed bytes,
  ``rank`` int64 dims, name, unit, padding up to ``headersize``) followed by little-endian data.  Dataset names are
  lower-case.  Per star (all float32 unless noted): px py pz vx vy vz feh alpha smass age rad mag0 mag1 mag2
  popid(i32) satid(i32) fieldid(i32) partid(i32) lum teff grav mact mtip dcmc_i dcmc_j dcmc_h dcmc_ks exbv_schlegel
  exbv_solar exbv_schlegel_inf glon glat; plus /log (char) and /center (f64[6]).  rad is in kpc ("1000 parsec"),
  teff = log10 K, grav = log g (cgs), DCMC magnitudes are ABSOLUTE and intrinsic, exbv_solar is the E(B-V) of the
  3-D extinction map: exactly the nine ``fb.RAW_COLUMNS`` (``ebv`` <- ``exbv_solar``).
* stdout, one line per field: ``[ i/N] TIC <tic> (l=.., b=.., seed=..): spawned <kept> stars (<raw> raw) in <s>s | ...``
  (``raw`` = all J <= 22 stars before the ceiling = the C++ engine's ``n_galaxia``); a failed field prints
  ``[batch] ERROR: Field TIC <tic> ... FAILED: <why>. Continuing batch...`` on stderr and the batch goes on.  The exit status
  is 0 even when fields failed, so success is judged per field from BOTH the line and a complete file.

Ceiling: ``--tmag-ceiling`` cuts on f64 values (``log10(rad*100)``) while ``fb.apply_tmag_ceiling`` cuts on the stored float32
values (``5*log10(rad*1000)-5``); the two can differ in the 7th digit.  So the binary is run with the fleet ceiling PLUS
``CEILING_MARGIN`` (a safe superset) and the fleet's own ``fb.apply_tmag_ceiling`` is applied on the read-back arrays: the stored
rows are the fleet's definition, identical to the C++ engine's, whatever galaxia-rs does at the edge.

Precision: galaxia-rs does not report it (``--version`` is static; the batch banner prints the OpenCL device NAME only).  cpu and
cuda are fp64.  For opencl the runner asks ``clinfo`` for the device galaxia-rs will pick (first GPU of the first platform that
has one, like ``OpenClBackend::new``) and requires ``cl_khr_fp64`` / ``cl_amd_fp64`` in its extensions; anything it cannot prove
is treated as fp32.  An fp32 backend is refused unless ``allow_fp32`` (test runs only).  The device name printed by every batch is
compared with the probed one, so a different device cannot slip in.

Stdlib + numpy only (compute hosts have no exonit install).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                              # repo layout
    from exonit import fpp_background as fb
except ImportError:                               # flat bundle (hosts without the exonit package)
    import fpp_background as fb                   # type: ignore

BACKENDS = ("cpu", "cuda", "opencl")
SLICE_TICS = 100                  # TICs per batch process: small so progress, heartbeats and resume keep working
DEFAULT_CEILING = 22.5            # the plain-mapping Tmag ceiling of the rs population (+C22.5)
CEILING_MARGIN = 0.05             # galaxia-rs runs at ceiling + margin; the fleet's own cut then makes the exact selection
RESIDENT_GB = 7.0                 # one batch process: ~6.5 GB peak RSS measured (<host> 6.47, Mac 6.84) incl. a field
LOAD_WAIT_S = 600.0               # a batch process must have loaded its resident data within this
POLL_S = 0.5

RS_COLUMN_MAP = {"rad": "rad", "mact": "mact", "grav": "grav", "teff": "teff", "feh": "feh",
                 "dcmc_j": "dcmc_j", "dcmc_h": "dcmc_h", "dcmc_ks": "dcmc_ks", "ebv": "exbv_solar"}   # RAW column <- dataset
# every per-star dataset the writer emits (truncation guard: a file cut between two datasets lacks the last ones)
RS_STAR_DATASETS = ("px", "py", "pz", "vx", "vy", "vz", "feh", "alpha", "smass", "age", "rad", "mag0", "mag1", "mag2",
                    "popid", "satid", "fieldid", "partid", "lum", "teff", "grav", "mact", "mtip", "dcmc_i", "dcmc_j",
                    "dcmc_h", "dcmc_ks", "exbv_schlegel", "exbv_solar", "exbv_schlegel_inf", "glon", "glat")


class RsError(RuntimeError):
    """A galaxia-rs configuration or output problem."""


class RsFatal(RsError):
    """Not retryable and not a per-TIC failure (wrong device, unreadable binary): the runner must stop."""


# ==========================================================================
# EBF v1.1 reader (numpy only)
# ==========================================================================

_EBF_SIG = bytes([0x8A, 0x45, 0x42, 0x46, 0xAE, 0x2B, 0xAF, 0x0A])
_EBF_ENDIAN = 0x64636261
_EBF_DTYPES = {2: "<i4", 3: "<i8", 4: "<f4", 5: "<f8"}


def read_ebf(path: str, wanted: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """``({name: array} for the wanted datasets, {name: n_elements} for every dataset)``.  The whole file must parse: a
    bad signature / endianness, or a header or data block running past the end of the file (garbage after the last dataset
    is read as a header and fails the same way) -> ``RsError``."""
    size = os.path.getsize(path)
    arrays: Dict[str, np.ndarray] = {}
    lengths: Dict[str, int] = {}
    want = {w.lstrip("/").lower() for w in wanted}
    off = 0
    with open(path, "rb") as f:
        while off < size:
            f.seek(off)
            h = f.read(56)
            if len(h) < 56 or h[:8] != _EBF_SIG:
                raise RsError(f"{path}: not an EBF v1.1 dataset header at byte {off}")
            if h[8:10] != b"\x01\x01":
                raise RsError(f"{path}: EBF version {h[8]}.{h[9]} at byte {off}, only 1.1 is written by galaxia-rs")
            endian, headersize, namesize, dtype, datasize, rank, unitsize, sdefsize = struct.unpack("<8i", h[12:44])
            (capacity,) = struct.unpack("<q", h[48:56])
            if endian != _EBF_ENDIAN:
                raise RsError(f"{path}: unexpected EBF endianness marker at byte {off}")
            fixed = 56 + 8 * rank + namesize + unitsize + sdefsize
            if not (0 < rank <= 8 and fixed <= headersize <= 4096 and capacity >= 0):
                raise RsError(f"{path}: inconsistent EBF header at byte {off}")
            rest = f.read(headersize - 56)
            if len(rest) != headersize - 56:
                raise RsError(f"{path}: truncated EBF header at byte {off}")
            dims = struct.unpack(f"<{rank}q", rest[:8 * rank])
            name = rest[8 * rank:8 * rank + namesize].decode("ascii", "replace").lstrip("/").lower()
            data_at = off + headersize
            if data_at + capacity > size:
                raise RsError(f"{path}: dataset /{name} runs past the end of the file (truncated)")
            n = 1
            for d in dims:
                n *= d
            if capacity != n * datasize:
                raise RsError(f"{path}: dataset /{name}: {capacity} bytes for {n} elements of {datasize} bytes")
            if name in lengths:
                raise RsError(f"{path}: dataset /{name} appears twice")
            lengths[name] = n
            if name in want:
                dt = _EBF_DTYPES.get(dtype)
                if dt is None or np.dtype(dt).itemsize != datasize:
                    raise RsError(f"{path}: dataset /{name} has unsupported type code {dtype}")
                f.seek(data_at)
                buf = f.read(capacity)
                if len(buf) != capacity:
                    raise RsError(f"{path}: truncated data of /{name}")
                arrays[name] = np.frombuffer(buf, dtype=dt, count=n).copy()
            off = data_at + capacity
    return arrays, lengths


def read_rs_field(path: str) -> Dict[str, np.ndarray]:
    """One ``gx_<tic>.ebf`` -> the nine ``fb.RAW_COLUMNS`` as float64 arrays (the float32 values widened, like the
    C++ engine's reader).  Every per-star dataset must be present with the same length: a half-written file fails."""
    arrays, lengths = read_ebf(path, list(RS_COLUMN_MAP.values()))
    missing = [d for d in RS_STAR_DATASETS if d not in lengths]
    if missing:
        raise RsError(f"{path}: datasets missing from the output: {missing}")
    ns = {lengths[d] for d in RS_STAR_DATASETS}
    if len(ns) != 1:
        raise RsError(f"{path}: per-star datasets of different length {sorted(ns)}")
    out = {}
    for col, ds in RS_COLUMN_MAP.items():
        a = arrays[ds]
        if a.dtype != np.dtype("<f4"):
            raise RsError(f"{path}: /{ds} is {a.dtype}, expected float32")
        out[col] = a.astype(np.float64)
    return out


# ==========================================================================
# Identity: version, source, data, precision
# ==========================================================================

def os_tag() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    raise RsError(f"unsupported OS {sys.platform!r}")


def _sha256_file12(path: str) -> str:
    return fb.file_sha256(path)[:12]


def _clinfo_raw() -> str:
    exe = shutil.which("clinfo")
    if not exe:
        raise RsError("clinfo is not installed: cannot prove that the OpenCL device has fp64")
    r = subprocess.run([exe, "--raw"], capture_output=True, text=True, timeout=60, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        raise RsError(f"clinfo --raw failed (exit {r.returncode}): {r.stderr.strip()[:200]}")
    return r.stdout


def opencl_gpu_device() -> dict:
    """``{"name", "fp64"}`` of the device ``OpenClBackend::new`` selects: the FIRST GPU device of the FIRST platform that
    has one, parsed from ``clinfo --raw`` (``[PLATFORM/INDEX] CL_DEVICE_NAME ...`` lines, in platform order)."""
    devs: Dict[str, dict] = {}
    order: List[str] = []
    for line in _clinfo_raw().splitlines():
        m = re.match(r"\[([^\]]+)/(\d+)\]\s+(CL_DEVICE_NAME|CL_DEVICE_TYPE|CL_DEVICE_EXTENSIONS)\s+(.*)$", line)
        if not m:
            continue
        key = f"{m.group(1)}/{m.group(2)}"
        if key not in devs:
            devs[key] = {"platform": m.group(1), "index": int(m.group(2))}
            order.append(key)
        devs[key][m.group(3)] = m.group(4).strip()
    for key in order:
        d = devs[key]
        if "GPU" in d.get("CL_DEVICE_TYPE", ""):
            ext = d.get("CL_DEVICE_EXTENSIONS", "").split()
            return {"name": d.get("CL_DEVICE_NAME", ""), "fp64": "cl_khr_fp64" in ext or "cl_amd_fp64" in ext}
    raise RsError("clinfo reports no OpenCL GPU device")


def _hash_tree_or_provenance(data_dir: str) -> str:
    """Data-tree sha256 of ``data_dir``: the sibling ``provenance-fleet.json`` of a standard install
    (``<root>/GalaxiaData`` next to it, schema 2, this CPU class) when it matches, else hashed live."""
    root = os.path.dirname(os.path.abspath(data_dir))
    prov = os.path.join(root, "provenance-fleet.json")
    if os.path.basename(os.path.abspath(data_dir)) == "GalaxiaData" and os.path.exists(prov):
        try:
            with open(prov) as f:
                p = json.load(f)
            if p.get("schema") == 2 and p.get("platform") == fb.platform_tag() and p.get("data_tree_sha256"):
                return p["data_tree_sha256"]
        except (OSError, ValueError):
            pass
    return fb.tree_sha256(data_dir)


def _git(args: Sequence[str], cwd: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", cwd] + list(args), capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


class RsInstall:
    """The galaxia-rs binary + its data directory + the backend to run on.  Duck-types ``fb.GalaxiaInstall`` where the
    runner needs it (``raw_version``, ``available``, ``provenance``, ``platform``)."""

    engine = "rs"

    def __init__(self, binary: str, data_dir: str, backend: str, allow_fp32: bool = False):
        if backend not in BACKENDS:
            raise RsError(f"--rs-backend must be one of {BACKENDS}, got {backend!r}")
        self.binary = os.path.abspath(os.path.expanduser(binary))
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.backend = backend
        self.allow_fp32 = bool(allow_fp32)
        self._memo: Dict[str, object] = {}

    # -- cached probes
    def _once(self, key: str, fn: Callable[[], object]):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def available(self) -> bool:
        return os.path.isfile(self.binary) and os.path.isdir(self.data_dir)

    def binary_sha256(self) -> str:
        return self._once("bin", lambda: fb.file_sha256(self.binary))        # type: ignore[return-value]

    def cargo_version(self) -> str:
        """``X.Y.Z`` from ``galaxia-rs --version --backend <b>`` (``galaxia-rs-X.Y.Z-<b>``; no device is initialised)."""
        def probe():
            try:
                r = subprocess.run([self.binary, "--version", "--backend", self.backend], capture_output=True, text=True,
                                   timeout=60, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RsError(f"cannot run {self.binary} --version: {exc}") from exc
            out = r.stdout.strip()
            # <= 0.2.1: "galaxia-rs-X.Y.Z-<b>";  >= 0.2.2: "galaxia-rs X.Y.Z (git <sha>[-dirty]) backend=<b>[-fp64] tmag_ceiling=.."
            m = re.fullmatch(r"galaxia-rs-(\d+\.\d+\.\d+[0-9A-Za-z.+~-]*?)-(cpu|cuda|opencl)", out)
            m2 = re.fullmatch(r"galaxia-rs (\d+\.\d+\.\d+[0-9A-Za-z.+~-]*) \(git ([0-9A-Za-z.-]+)\) "
                              r"backend=(cpu|cuda|opencl)(?:-fp64)? tmag_ceiling=\S+", out)
            if r.returncode != 0 or not (m or m2):
                raise RsError(f"unexpected `{self.binary} --version` output: {out[:120]!r}")
            if m2 and m2.group(2).endswith("-dirty"):
                raise RsError(f"{self.binary} was built from a modified tree (git {m2.group(2)}): not a production binary")
            version, backend = (m.group(1), m.group(2)) if m else (m2.group(1), m2.group(3))
            if backend != self.backend:
                raise RsError(f"binary reports backend {backend!r}, asked for {self.backend!r}")
            return version
        return self._once("cargo", probe)                                   # type: ignore[return-value]

    def src_id(self) -> str:
        """sha12 of the git HEAD of the galaxia-rs checkout the binary lives in; sha256-12 of the binary when there is no
        git checkout (or its tracked files are modified, since HEAD then does not describe the binary)."""
        def probe():
            d = os.path.dirname(os.path.realpath(self.binary))
            for _ in range(6):
                if os.path.isdir(os.path.join(d, ".git")) and os.path.exists(os.path.join(d, "Cargo.toml")):
                    head = _git(["rev-parse", "HEAD"], d)
                    dirty = _git(["status", "--porcelain", "--untracked-files=no"], d)
                    if head and re.fullmatch(r"[0-9a-f]{40,64}", head) and dirty == "":
                        return head[:12]
                    if head and dirty:
                        print(f"WARNING galaxia-rs checkout {d} has modified tracked files: the version carries the binary "
                              f"hash instead of the git sha", file=sys.stderr, flush=True)
                    break
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
            return self.binary_sha256()[:12]
        return self._once("src", probe)                                     # type: ignore[return-value]

    def data_id(self) -> str:
        return self._once("data", lambda: _hash_tree_or_provenance(self.data_dir)[:12])   # type: ignore[return-value]

    def opencl_device(self) -> dict:
        return self._once("cldev", opencl_gpu_device)                       # type: ignore[return-value]

    def precision(self) -> str:
        """``fp64`` / ``fp32``; an fp32 backend raises unless ``allow_fp32`` (see the module doc for how it is established)."""
        def probe():
            if self.backend in ("cpu", "cuda"):
                return "fp64"
            try:
                return "fp64" if self.opencl_device()["fp64"] else "fp32"
            except RsError as exc:
                if not self.allow_fp32:
                    raise RsError(f"opencl precision cannot be established ({exc}); refusing "
                                  f"(--allow-fp32 treats it as fp32, test runs only)") from exc
                return "fp32"
        prec = self._once("prec", probe)
        if prec == "fp32" and not self.allow_fp32:
            raise RsError(f"backend {self.backend} runs in fp32 (device {self._memo.get('cldev', {}).get('name', '?')!r}): "
                          f"its output is not production grade; refusing (--allow-fp32 for test runs only)")
        return prec                                                         # type: ignore[return-value]

    def plat(self) -> str:
        return f"{os_tag()}-{fb.platform_tag()}"

    @property
    def platform(self) -> str:
        """Label sent to the coordinator at claim time (informational)."""
        return f"rs-{fb.platform_tag()}"

    def family(self) -> str:
        return f"galaxia-rs-{self.cargo_version()}"

    def class_key(self) -> str:
        """Host class of the conformance gate: everything that decides the bits except area / magnitude limit / ceiling."""
        return (f"{self.family()}+src.{self.src_id()}+data.{self.data_id()}+plat.{self.plat()}"
                f"+be.{self.backend}-{self.precision()}")

    def raw_version(self, tmag_ceiling: Optional[float] = None) -> str:
        v = f"{self.class_key()}+A{fb.FIELD_AREA_DEG2}+M{fb.GALAXIA_MAG_LIMIT:g}"
        return v if tmag_ceiling is None else f"{v}+C{tmag_ceiling:g}"

    def provenance(self) -> dict:
        return {"engine": "rs", "binary_sha256": self.binary_sha256(), "cargo_version": self.cargo_version(),
                "src": self.src_id(), "data_tree": self.data_id(), "platform": self.plat(), "backend": self.backend,
                "precision": self.precision(), "class_key": self.class_key()}


# ==========================================================================
# One batch process
# ==========================================================================

_OK_RE = re.compile(r"\]\s*TIC\s+(\d+)\s+\(.*?\):\s*spawned\s+(\d+)\s+stars\s+\((\d+)\s+raw\)\s+in\s+([0-9.]+)s")
_ERR_RE = re.compile(r"\[batch\] ERROR: Field TIC (\d+) .*? FAILED: (.*?)\. Continuing batch")
_DEV_RE = re.compile(r"^Accelerator backend: (.*?)\s*$", re.M)


def _reap(p: subprocess.Popen):
    """``(finished, returncode, rss_mb)`` without blocking; rss from ``wait4`` on POSIX."""
    if sys.platform == "win32":
        rc = p.poll()
        return rc is not None, rc, None
    pid, status, ru = os.wait4(p.pid, os.WNOHANG)
    if pid == 0:
        return False, None, None
    p.returncode = os.waitstatus_to_exitcode(status)
    rss = ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0
    return True, p.returncode, rss


def _kill(p: subprocess.Popen) -> None:
    try:
        p.kill()
    except OSError:
        pass


def _csv_text(items: Sequence[Tuple[str, float, float]]) -> str:
    # the same rounding as generate_raw: the stored ra/dec regenerate the field exactly
    return "tic,ra,dec\n" + "".join(f"{t},{round(float(ra), 6):.6f},{round(float(dec), 6):.6f}\n" for t, ra, dec in items)


def _tail(path: str, n: int = 400) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace").strip().replace("\n", " | ")
    except OSError:
        return "no log"


class Once:
    """Outcome of one batch process over some TICs (fields already emitted are not repeated here)."""
    def __init__(self):
        self.ok: Dict[str, dict] = {}              # tic -> {"n_raw", "dt"}
        self.errors: Dict[str, str] = {}           # tic -> reason (explicit ERROR line or unreadable file)
        self.rc: Optional[int] = None
        self.crashed = False                       # non-zero exit, killed or stalled
        self.stalled = False
        self.rss_mb: Optional[float] = None
        self.device: Optional[str] = None
        self.log_tail = ""
        self.interrupted = False                   # gate/abort refused to start or killed it


def run_batch_once(install: RsInstall, items: Sequence[Tuple[str, float, float]], scratch: str, *,
                   cpu: Optional[object], nice: int, stall_s: float, ceiling: Optional[float],
                   on_field: Callable[[Tuple[str, float, float], dict, int, float], None],
                   abort: Optional[threading.Event] = None, start_lock=None,
                   gate_wait: Optional[Callable[[], bool]] = None, expect_device: Optional[str] = None,
                   threads: int = 1) -> Once:
    """Run ``galaxia-rs batch`` over ``items`` in a private scratch directory.  Every field that the binary reports done is
    read back, cut with the fleet's ceiling and handed to ``on_field(item, raw, n_galaxia, dt)`` AS SOON AS its line appears (and
    its file is removed), so a slice never piles up gigabytes of EBF.  A field the binary reports as failed, or whose file is
    absent / cut / inconsistent, is NOT emitted and lands in ``Once.errors``.  ``stall_s``: the process is killed when its log
    does not grow for that long.  ``start_lock`` is held from the memory-gate check until the resident data is loaded (or the
    process is gone), so that two starts never overlap their ~5 GB load.  ``expect_device``: the device banner must equal it
    (``RsFatal`` otherwise, before any field is read)."""
    out = Once()
    by_tic = {t: (t, ra, dec) for t, ra, dec in items}
    work = tempfile.mkdtemp(prefix="rs_", dir=scratch)
    held = False
    p = None
    try:
        csv_path = os.path.join(work, "slice.csv")
        out_dir = os.path.join(work, "out")
        log_path = os.path.join(work, "batch.log")
        os.makedirs(out_dir)
        with open(csv_path, "w", newline="\n") as f:
            f.write(_csv_text(items))
        cmd: List[str] = []
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x00004000                       # BELOW_NORMAL_PRIORITY_CLASS
        else:
            cmd += ["nice", "-n", str(nice)]
            if cpu is not None and sys.platform.startswith("linux") and shutil.which("taskset"):
                cmd += ["taskset", "-c", str(cpu)]
        cmd += [install.binary, "batch", "--csv", csv_path, "--out-dir", out_dir, "--data-dir", install.data_dir,
                "--backend", install.backend]
        if ceiling is not None:
            cmd += ["--tmag-ceiling", f"{float(ceiling) + CEILING_MARGIN:.6f}"]
        if threads > 1:                                   # not passed for 1: older binaries have no --threads; output is bit-identical for any N
            cmd += ["--threads", str(int(threads))]
        if start_lock is not None:
            start_lock.acquire()
            held = True
        if gate_wait is not None and not gate_wait():
            out.interrupted = True
            return out
        t_start = time.time()
        with open(log_path, "wb") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=work, stdin=subprocess.DEVNULL, **kw)

        offset, last_change, text_tail = 0, time.time(), ""
        loaded = False
        while True:
            finished, rc, rss = _reap(p)
            try:
                with open(log_path, "rb") as lf:
                    lf.seek(offset)
                    chunk = lf.read()
                offset += len(chunk)
            except OSError:
                chunk = b""
            if chunk:
                last_change = time.time()
            lines = (text_tail + chunk.decode("utf-8", "replace")).split("\n")
            text_tail = lines.pop()
            if finished and text_tail:
                lines.append(text_tail)
                text_tail = ""
            for line in lines:
                if "Resident data loaded" in line:
                    loaded = True
                m = _DEV_RE.search(line)
                if m:
                    out.device = m.group(1)
                    if expect_device is not None and out.device != expect_device:
                        raise RsFatal(f"galaxia-rs initialised device {out.device!r}, the precision probe saw "
                                      f"{expect_device!r}: refusing")
                m = _ERR_RE.search(line)
                if m and m.group(1) in by_tic:
                    out.errors[m.group(1)] = f"galaxia-rs reported the field as failed: {m.group(2)[:300]}"
                    continue
                m = _OK_RE.search(line)
                if m and m.group(1) in by_tic and m.group(1) not in out.ok:
                    tic, n_kept, n_raw, dt = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
                    path = os.path.join(out_dir, f"gx_{tic}.ebf")
                    try:
                        arrays = read_rs_field(path)
                        n_file = len(arrays["rad"])
                        if n_file != n_kept:
                            raise RsError(f"file holds {n_file} stars, the batch said {n_kept}")
                        if ceiling is not None:
                            arrays = fb.apply_tmag_ceiling(arrays, ceiling)
                        item = by_tic[tic]
                        ra, dec = round(float(item[1]), 6), round(float(item[2]), 6)
                        l, b = fb.galactic_lb(ra, dec)
                        raw = {"tic_id": str(tic), "ra": ra, "dec": dec, "gal_l": l, "gal_b": b,
                               "seed": fb.seed_for_tic(tic), "arrays": arrays, "n_galaxia": n_raw}
                    except (RsError, OSError) as exc:
                        out.errors[tic] = f"{type(exc).__name__}: {exc}"[:400]
                        continue
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(path)
                    on_field(item, raw, n_raw, dt)
                    out.ok[tic] = {"n_raw": n_raw, "dt": dt}
            if held and (loaded or finished or time.time() - t_start > min(stall_s, LOAD_WAIT_S)):
                start_lock.release()
                held = False
            if finished:
                out.rc, out.rss_mb = rc, rss
                out.crashed = rc != 0
                break
            if abort is not None and abort.is_set():
                _kill(p)
                _reap_block(p)
                out.crashed = out.interrupted = True
                break
            if time.time() - last_change > stall_s:
                _kill(p)
                _reap_block(p)
                out.crashed = out.stalled = True
                break
            time.sleep(POLL_S)
        out.log_tail = _tail(log_path)
        return out
    except BaseException:
        if p is not None and p.returncode is None:
            _kill(p)
            _reap_block(p)
        raise
    finally:
        if held:
            start_lock.release()
        shutil.rmtree(work, ignore_errors=True)


def _reap_block(p: subprocess.Popen) -> None:
    try:
        p.wait(timeout=30)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


# ==========================================================================
# A slice with retries
# ==========================================================================

def generate_slice(install: RsInstall, items: Sequence[Tuple[str, float, float]], scratch: str, *, retries: int, nice: int,
                   stall_s: float, ceiling: Optional[float], cpu: Optional[object] = None,
                   threads: int = 1, emit: Callable[[Tuple[str, float, float], Optional[dict], Optional[str], int, float, Optional[float]], None],
                   stop: Optional[threading.Event] = None, abort: Optional[threading.Event] = None,
                   start_lock=None, gate_wait: Optional[Callable[[], bool]] = None,
                   expect_device: Optional[str] = None, log: Callable[[str], None] = lambda m: None) -> dict:
    """Generate ``items`` with as many batch processes as needed: each TIC gets ``retries + 1`` attempts.

    ``emit(item, raw|None, error|None, attempts, dt, rss_mb)`` is called once per TIC: a raw field, an error text after the
    last attempt, or ``"stopped"`` when the runner is stopping.  Attempts are charged only to a TIC the binary reported as
    failed, or the one a crashed / stalled process was working on (the first TIC with no line); the TICs behind it are
    retried free of charge.  ``retries + 1`` rounds without a single finished field end the slice (a broken install must not
    spin).  Returns ``{"rounds", "rss_mb"}``."""
    pending = list(items)
    attempts = {t: 0 for t, _, _ in items}
    last_err = {t: "no output" for t, _, _ in items}
    rounds, rss_max, barren = 0, None, 0

    def finish_ok(item, raw, n_raw, dt):
        attempts[item[0]] += 1
        emit(item, raw, None, attempts[item[0]], dt, None)
    while pending:
        if (stop is not None and stop.is_set()) or (abort is not None and abort.is_set()):
            break
        rounds += 1
        res = run_batch_once(install, pending, scratch, cpu=cpu, nice=nice, stall_s=stall_s, ceiling=ceiling,
                             on_field=finish_ok, abort=abort, start_lock=start_lock, gate_wait=gate_wait,
                             expect_device=expect_device, threads=threads)
        if res.interrupted:
            break
        if res.rss_mb is not None:
            rss_max = res.rss_mb if rss_max is None else max(rss_max, res.rss_mb)
        if expect_device is not None and res.device is None and (res.ok or res.rc == 0):
            raise RsFatal("galaxia-rs printed no device banner: cannot confirm the OpenCL device")
        nxt = []
        blamed = False
        for it in pending:
            t = it[0]
            if t in res.ok:
                continue
            charge = True
            if t in res.errors:
                last_err[t] = res.errors[t]
            elif res.crashed:
                if blamed:
                    charge = False                                   # never reached: free retry
                else:
                    blamed = True
                    why = "stalled (no output)" if res.stalled else f"exit code {res.rc}"
                    last_err[t] = f"galaxia-rs batch process died ({why}) while on this TIC; log tail: {res.log_tail}"[:600]
            else:
                last_err[t] = f"no output for the TIC (batch exited 0 without a result line); log tail: {res.log_tail}"[:600]
            if charge:
                attempts[t] += 1
            if attempts[t] > retries:
                emit(it, None, last_err[t], attempts[t], 0.0, rss_max)
            else:
                nxt.append(it)
        barren = 0 if res.ok else barren + 1
        if barren > retries and nxt:                       # a broken install: stop spinning
            why = f"batch produced no field in {barren} rounds; log tail: {res.log_tail}"[:600]
            for it in nxt:
                emit(it, None, why, max(1, attempts[it[0]]), 0.0, rss_max)
            nxt = []
        pending = nxt
        log(f"slice round {rounds}: {len(res.ok)} ok, {len(res.errors)} reported failed, rc={res.rc}"
            + (" STALLED" if res.stalled else ""))
    for it in pending:                                                    # left when the runner was stopping
        emit(it, None, "stopped", attempts[it[0]], 0.0, rss_max)
    return {"rounds": rounds, "rss_mb": rss_max}
