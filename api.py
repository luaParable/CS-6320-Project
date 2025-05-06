from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile, os, socket
from transcribe_md import generate_markdown

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # pv2/
UI_DIR = ROOT / "ui" / "frontend"  # <-- fixed
UI_DIR.mkdir(parents=True, exist_ok=True)  # create if missing

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="ATC-Whisper API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def root(): return RedirectResponse("/ui/")


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...)
):
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read());
        tmp_path = Path(tmp.name)
    try:
        md = generate_markdown(tmp_path, compare=False)
        # --- convert MD → JSON (first body paragraph = transcript)
        lines = md.splitlines()
        text = next((l for l in lines if l and not l.startswith('#')), "")
        ent_rows = [l for l in lines if l.startswith('|')][2:]
        entities = [tuple(c.strip() for c in r.split('|')[1:3]) for r in ent_rows]
        return {"text": text, "entities": entities}
    finally:
        os.unlink(tmp_path)


app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="frontend")


# ------------------------------------------------------------------
# pick first free port starting at 8000
# ------------------------------------------------------------------
def _pick_port(start=8000, tries=20):
    port = start
    for _ in range(tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)):  # 0 = busy
                return port
        port += 1
    raise RuntimeError("no free port")


if __name__ == "__main__":
    import uvicorn, sys

    port = _pick_port()
    print(f"🚀  http://127.0.0.1:{port}/ui/")
    uvicorn.run("api:app", host="127.0.0.1", port=port,
                reload="--reload" in sys.argv)
