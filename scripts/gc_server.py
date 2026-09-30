"""Grand Challenge /health and /invoke server for Photon and Proton CT/MR."""

from __future__ import annotations
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import torch
import prismx.runtime as inf

MODALITY = os.environ.get("PRISMX_MODALITY", "photon")
IMAGE_TYPE = os.environ.get("PRISMX_IMAGE_TYPE", "mri")
INPUT = Path(os.environ.get("INPUT_PATH", "/input"))
OUTPUT = Path(os.environ.get("OUTPUT_PATH", "/output"))
DEVICE = os.environ.get("PRISMX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", "4743"))
_models = None
_ready = threading.Event()
_lock = threading.Lock()


def _load_models() -> None:
    global _models
    print(
        f"[gc_server] loading {MODALITY}/{IMAGE_TYPE} models on {DEVICE} ...",
        flush=True,
    )
    _models = inf.Models(
        MODALITY, IMAGE_TYPE, DEVICE, inf._default_cfg(MODALITY, IMAGE_TYPE)
    )
    _ready.set()
    print("[gc_server] models ready", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: str = "") -> None:
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(
                200 if _ready.is_set() else 503,
                "ready" if _ready.is_set() else "loading",
            )
        else:
            self._send(404)

    def do_POST(self):
        if self.path.rstrip("/") != "/invoke":
            self._send(404)
            return
        if not _ready.is_set():
            self._send(503, "models not loaded")
            return
        try:
            with _lock:
                inf.run(INPUT, OUTPUT, _models)
            self._send(201, "ok")
        except Exception as e:
            traceback.print_exc()
            self._send(500, f"{type(e).__name__}: {e}")


def main() -> None:
    _load_models()
    print(
        f"[gc_server] serving {MODALITY}/{IMAGE_TYPE} on :{PORT} (input={INPUT} output={OUTPUT})",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
