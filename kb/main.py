import os, shutil, tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
# NOTE: form.items() yields Starlette's UploadFile, not FastAPI's subclass -> import Starlette's.
from starlette.datastructures import UploadFile

from .llm import LLM
from .ingest import index_zip
from .retrieve import Engine

app = FastAPI(title="Smart Knowledge Base for Agents")
llm = LLM()
engine = Engine(llm)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def get_ui():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Smart Knowledge Base API: POST /index, POST /query"}


class Q(BaseModel):
    collection_name: str = ""
    query: str = ""

    def __init__(self, **data):
        if "collection" in data and not data.get("collection_name"):
            data["collection_name"] = data["collection"]
        if "question" in data and not data.get("query"):
            data["query"] = data["question"]
        super().__init__(**data)


@app.post("/index")
async def index(request: Request):
    """multipart/form-data: collection_name + a zip upload under ANY field name."""
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(400, f"could not parse multipart form: {e}")

    coll = None
    for key in ("collection_name", "collection", "name", "collectionName"):
        v = form.get(key)
        if isinstance(v, str) and v.strip():
            coll = v.strip()
            break
    if not coll:
        coll = request.query_params.get("collection_name") or request.query_params.get("collection")
    if not coll:
        raise HTTPException(400, "collection_name (or collection) is required")

    target = None
    for _, v in form.multi_items():
        if isinstance(v, UploadFile) and getattr(v, "filename", None):
            target = v
            break
    if target is None:
        raise HTTPException(400, "No zip file uploaded in request (checked all form fields)")

    fd, tmp = tempfile.mkstemp(suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(target.file, f)
        # run the blocking indexer in a worker thread so /query and /docs stay responsive
        stats = await run_in_threadpool(index_zip, tmp, coll, llm=llm)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"indexing failed: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"status": "success", "collection_name": coll, **stats}


@app.post("/query")
def query(q: Q):
    coll = q.collection_name
    query_str = q.query
    if not coll or not query_str:
        raise HTTPException(400, "Both collection_name and query are required")
    try:
        return engine.query(coll, query_str)
    except KeyError as e:
        raise HTTPException(404, str(e))
