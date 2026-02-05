"""REST API server for CoLibri - enables integration with Microsoft Copilot and other HTTP clients.

Provides semantic search over a personal library of technical books and notes via REST endpoints.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from colibri.config import LANCEDB_DIR, LIBRARY_PATH

# --- Pydantic Models for OpenAPI Schema ---


class SearchResult(BaseModel):
    """A single search result from the library."""

    text: str = Field(description="The matched text content")
    file: str = Field(description="Source file path relative to library root")
    title: str = Field(description="Document title")
    type: str = Field(description="Document type (book, note, etc.)")
    folder: str = Field(description="Folder containing the document")
    score: float | None = Field(description="Similarity score (higher is better)")


class SearchResponse(BaseModel):
    """Response containing search results."""

    query: str = Field(description="The original search query")
    results: list[SearchResult] = Field(description="List of matching documents")
    count: int = Field(description="Number of results returned")


class BookInfo(BaseModel):
    """Information about an indexed book."""

    title: str = Field(description="Book title")
    chunks: int = Field(description="Number of indexed chunks")
    file: str | None = Field(default=None, description="Source file path relative to library root")
    author: str | None = Field(default=None, description="Book author")
    language: str | None = Field(default=None, description="Language code (e.g. 'en', 'de')")
    tags: list[str] | None = Field(default=None, description="Document tags")
    original_title: str | None = Field(
        default=None, description="Original title before translation"
    )
    translated_from: str | None = Field(
        default=None, description="Language the book was translated from"
    )


class BooksResponse(BaseModel):
    """Response containing list of indexed books."""

    books: list[BookInfo] = Field(description="List of indexed books")
    count: int = Field(description="Total number of books")


class NoteMetadata(BaseModel):
    """Metadata from a note's frontmatter."""

    title: str | None = None
    type: str | None = None
    author: str | None = None
    tags: list[str] | None = None

    class Config:
        extra = "allow"


class NoteResponse(BaseModel):
    """Response containing a single note."""

    path: str = Field(description="Note path relative to library root")
    title: str = Field(description="Note title")
    metadata: dict = Field(description="Frontmatter metadata")
    content: str = Field(description="Note content (markdown)")


class LinksResponse(BaseModel):
    """Response containing linked notes."""

    path: str = Field(description="Source note path")
    links: list[str] = Field(description="List of linked note names")
    count: int = Field(description="Number of links")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    library_path: str = Field(description="Configured library path")
    library_exists: bool = Field(description="Whether library directory exists")
    index_exists: bool = Field(description="Whether search index exists")


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(description="Error message")


class SourceProfile(BaseModel):
    """A source profile for folder indexing."""

    path: str = Field(description="Absolute path to the source directory")
    mode: str = Field(
        default="incremental",
        description="Indexing mode: static, incremental, append_only, disabled",
    )
    doc_type: str = Field(default="note", description="Default document type")
    name: str | None = Field(default=None, description="Display name (default: directory name)")
    extensions: list[str] | None = Field(default=None, description="File extensions to index")
    chunk_size: int | None = Field(default=None, description="Per-folder chunk size override")
    chunk_overlap: int | None = Field(default=None, description="Per-folder chunk overlap override")


class SourcesResponse(BaseModel):
    """Response containing source profiles."""

    sources: list[SourceProfile] = Field(description="List of source profiles")
    count: int = Field(description="Number of profiles")


# --- Search Engine Singleton ---

_engine = None


def get_search_engine():
    """Get or create the search engine singleton."""
    global _engine
    if _engine is None:
        from colibri.query import SearchEngine

        _engine = SearchEngine()
    return _engine


# --- FastAPI Application ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize search engine on startup."""
    get_search_engine()
    yield


app = FastAPI(
    title="CoLibri API",
    description="""
**CoLibri** (COntext LIBRary) - Local RAG system for technical books and notes.

This API provides semantic search over your personal library of technical books and notes.
Use it with Microsoft Copilot, custom scripts, or any HTTP client.

## Features

- **Semantic Search** - Find relevant passages using local embeddings
- **Book Search** - Search specifically within your book collection
- **Note Retrieval** - Get full note content and metadata
- **Link Extraction** - Discover [[wiki links]] between notes

## Authentication

This API runs locally and does not require authentication.
Ensure it's only accessible on localhost in production.
    """,
    version="0.2.0",
    contact={
        "name": "CoLibri",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local use only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check if the API is running and the library is accessible.",
)
async def health_check() -> HealthResponse:
    """Check system health and configuration."""
    return HealthResponse(
        status="healthy",
        library_path=str(LIBRARY_PATH),
        library_exists=LIBRARY_PATH.exists(),
        index_exists=LANCEDB_DIR.exists() and any(LANCEDB_DIR.iterdir())
        if LANCEDB_DIR.exists()
        else False,
    )


@app.get(
    "/api/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search entire library",
    description="Perform semantic search across all indexed content (books and notes).",
    responses={500: {"model": ErrorResponse}},
)
async def search_library(
    q: Annotated[str, Query(description="Search query", min_length=1)],
    limit: Annotated[int, Query(description="Maximum results to return", ge=1, le=50)] = 5,
) -> SearchResponse:
    """Search the entire library using semantic similarity."""
    try:
        engine = get_search_engine()
        results = engine.search_library(q, limit=limit)
        return SearchResponse(
            query=q,
            results=[SearchResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/search/books",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search books only",
    description="Perform semantic search only within the books folder.",
    responses={500: {"model": ErrorResponse}},
)
async def search_books(
    q: Annotated[str, Query(description="Search query", min_length=1)],
    limit: Annotated[int, Query(description="Maximum results to return", ge=1, le=50)] = 5,
) -> SearchResponse:
    """Search only the books collection."""
    try:
        engine = get_search_engine()
        results = engine.search_books(q, limit=limit)
        return SearchResponse(
            query=q,
            results=[SearchResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/books",
    response_model=BooksResponse,
    tags=["Books"],
    summary="List indexed books",
    description="Get a list of all indexed books with their chunk counts.",
    responses={500: {"model": ErrorResponse}},
)
async def list_books() -> BooksResponse:
    """List all indexed books."""
    try:
        engine = get_search_engine()
        books = engine.list_books()
        return BooksResponse(
            books=[BookInfo(**b) for b in books],
            count=len(books),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/notes/{note_path:path}",
    response_model=NoteResponse,
    tags=["Notes"],
    summary="Get a note",
    description="Retrieve a specific note by its path relative to the library root.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_note(note_path: str) -> NoteResponse:
    """Get a specific note by path."""
    try:
        engine = get_search_engine()
        note = engine.get_note(note_path)
        if note is None:
            raise HTTPException(status_code=404, detail=f"Note not found: {note_path}")
        return NoteResponse(**note)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/notes/{note_path:path}/links",
    response_model=LinksResponse,
    tags=["Notes"],
    summary="Get linked notes",
    description="Extract all [[wiki links]] from a note.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_linked_notes(note_path: str) -> LinksResponse:
    """Get wiki links from a note."""
    try:
        engine = get_search_engine()
        # Check if note exists via the engine's source resolution
        if engine.get_note(note_path) is None:
            raise HTTPException(status_code=404, detail=f"Note not found: {note_path}")
        links = engine.get_linked_notes(note_path)
        return LinksResponse(
            path=note_path,
            links=links,
            count=len(links),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Config Management Endpoints ---


@app.get(
    "/api/config/sources",
    response_model=SourcesResponse,
    tags=["Config"],
    summary="List source profiles",
    description="Get all configured source profiles for folder indexing.",
)
async def list_sources() -> SourcesResponse:
    """List all source profiles."""
    from colibri.config import get_sources_raw

    sources = get_sources_raw()
    return SourcesResponse(
        sources=[SourceProfile(**s) for s in sources],
        count=len(sources),
    )


@app.post(
    "/api/config/sources",
    response_model=SourceProfile,
    tags=["Config"],
    summary="Add source profile",
    description="Add a new source profile for a folder.",
    responses={409: {"model": ErrorResponse}},
)
async def add_source(profile: SourceProfile) -> SourceProfile:
    """Add a new source profile."""
    from colibri.config import get_sources_raw, set_sources_raw

    sources = get_sources_raw()
    existing_paths = {s.get("path") for s in sources}

    if profile.path in existing_paths:
        raise HTTPException(
            status_code=409,
            detail=f"Profile for path '{profile.path}' already exists",
        )

    entry = profile.model_dump(exclude_none=True)
    sources.append(entry)
    set_sources_raw(sources)
    return profile


@app.put(
    "/api/config/sources/{source_path:path}",
    response_model=SourceProfile,
    tags=["Config"],
    summary="Update source profile",
    description="Update an existing source profile.",
    responses={404: {"model": ErrorResponse}},
)
async def update_source(source_path: str, profile: SourceProfile) -> SourceProfile:
    """Update an existing source profile."""
    from colibri.config import get_sources_raw, set_sources_raw

    sources = get_sources_raw()

    for i, src in enumerate(sources):
        if src.get("path") == source_path:
            sources[i] = profile.model_dump(exclude_none=True)
            set_sources_raw(sources)
            return profile

    raise HTTPException(status_code=404, detail=f"Profile not found: {source_path}")


@app.delete(
    "/api/config/sources/{source_path:path}",
    tags=["Config"],
    summary="Delete source profile",
    description="Remove a source profile.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_source(source_path: str) -> dict:
    """Delete a source profile."""
    from colibri.config import get_sources_raw, set_sources_raw

    sources = get_sources_raw()
    new_sources = [s for s in sources if s.get("path") != source_path]

    if len(new_sources) == len(sources):
        raise HTTPException(status_code=404, detail=f"Profile not found: {source_path}")

    set_sources_raw(new_sources)
    return {"detail": f"Removed profile for '{source_path}'"}


# --- Web Config UI ---


_CONFIG_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoLibri - Source Profiles</title>
<style>
  :root {
    --bg: #1a1a2e; --surface: #16213e; --accent: #0f3460;
    --primary: #e94560; --text: #eee; --muted: #999;
    --border: #2a2a4a; --success: #2ecc71; --danger: #e74c3c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    max-width: 900px; margin: 0 auto; padding: 2rem 1rem;
  }
  h1 { color: var(--primary); margin-bottom: .25rem; }
  .subtitle { color: var(--muted); margin-bottom: 2rem; }
  table {
    width: 100%; border-collapse: collapse;
    background: var(--surface); border-radius: 8px;
    overflow: hidden; margin-bottom: 1rem;
  }
  th { background: var(--accent); text-align: left; padding: .75rem 1rem; font-weight: 600; }
  td { padding: .75rem 1rem; border-top: 1px solid var(--border); }
  tr:hover td { background: rgba(255,255,255,.03); }
  .mode-static { color: #3498db; }
  .mode-incremental { color: #2ecc71; }
  .mode-append_only { color: #f39c12; }
  .mode-disabled { color: #e74c3c; }
  button {
    padding: .5rem 1rem; border: none; border-radius: 4px;
    cursor: pointer; font-size: .875rem; transition: opacity .2s;
  }
  button:hover { opacity: .85; }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-edit { background: var(--accent); color: var(--text); }
  .btn-danger { background: var(--danger); color: #fff; }
  .btn-add { margin-bottom: 1rem; font-size: 1rem; padding: .6rem 1.5rem; }
  .actions { display: flex; gap: .5rem; }
  /* Modal */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.6); z-index: 100;
    justify-content: center; align-items: center;
  }
  .modal-overlay.active { display: flex; }
  .modal {
    background: var(--surface); border-radius: 8px;
    padding: 2rem; width: 100%; max-width: 480px;
    border: 1px solid var(--border);
  }
  .modal h2 { margin-bottom: 1rem; color: var(--primary); }
  .form-group { margin-bottom: 1rem; }
  .form-group label {
    display: block; margin-bottom: .25rem;
    color: var(--muted); font-size: .875rem;
  }
  .form-group input, .form-group select {
    width: 100%; padding: .5rem; border-radius: 4px;
    border: 1px solid var(--border); background: var(--bg);
    color: var(--text); font-size: .9rem;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none; border-color: var(--primary);
  }
  .form-buttons { display: flex; justify-content: flex-end; gap: .5rem; margin-top: 1.5rem; }
  .toast {
    position: fixed; bottom: 1rem; right: 1rem; padding: .75rem 1.5rem;
    border-radius: 4px; background: var(--success); color: #fff;
    opacity: 0; transition: opacity .3s; z-index: 200;
  }
  .toast.show { opacity: 1; }
  .toast.error { background: var(--danger); }
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
</style>
</head>
<body>

<h1>CoLibri</h1>
<p class="subtitle">Source Profile Management</p>

<button class="btn-primary btn-add" onclick="openAdd()">+ Add Profile</button>

<div id="table-container"></div>

<!-- Modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2 id="modal-title">Add Source Profile</h2>
    <div class="form-group">
      <label for="f-path">Path (absolute directory path)</label>
      <input id="f-path" placeholder="e.g. /Users/you/Documents/Notes">
    </div>
    <div class="form-group">
      <label for="f-name">Display Name (blank = directory name)</label>
      <input id="f-name" placeholder="e.g. My Books">
    </div>
    <div class="form-group">
      <label for="f-mode">Mode</label>
      <select id="f-mode">
        <option value="static">static</option>
        <option value="incremental" selected>incremental</option>
        <option value="append_only">append_only</option>
        <option value="disabled">disabled</option>
      </select>
    </div>
    <div class="form-group">
      <label for="f-doctype">Doc Type</label>
      <input id="f-doctype" value="note" placeholder="e.g. note, book">
    </div>
    <div class="form-group">
      <label for="f-chunk-size">Chunk Size (blank = global default)</label>
      <input id="f-chunk-size" type="number" placeholder="e.g. 3000">
    </div>
    <div class="form-group">
      <label for="f-chunk-overlap">Chunk Overlap (blank = global default)</label>
      <input id="f-chunk-overlap" type="number" placeholder="e.g. 200">
    </div>
    <div class="form-buttons">
      <button class="btn-edit" onclick="closeModal()">Cancel</button>
      <button class="btn-primary" id="modal-save" onclick="saveProfile()">Save</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = '/api/config/sources';
let editingPath = null; // null = add mode, string = edit mode

async function loadProfiles() {
  const res = await fetch(API);
  const data = await res.json();
  renderTable(data.sources);
}

function renderTable(sources) {
  const c = document.getElementById('table-container');
  if (!sources.length) {
    c.innerHTML = '<div class="empty">No source profiles configured.'
      + '<br>Click "+ Add Profile" to get started.</div>';
    return;
  }
  let html = '<table><thead><tr><th>Name</th><th>Path</th><th>Mode</th>'
    + '<th>Doc Type</th><th>Chunk Size</th>'
    + '<th>Actions</th></tr></thead><tbody>';
  for (const s of sources) {
    const displayName = s.name || s.path.split('/').pop();
    html += `<tr>
      <td>${esc(displayName)}</td>
      <td title="${esc(s.path)}"><code>${esc(s.path)}</code></td>
      <td><span class="mode-${s.mode}">${s.mode}</span></td>
      <td>${esc(s.doc_type)}</td>
      <td>${s.chunk_size ?? 'default'}</td>
      <td class="actions">
        <button class="btn-edit" onclick="openEdit('${esc(s.path)}')">Edit</button>
        <button class="btn-danger" onclick="deleteProfile('${esc(s.path)}')">Delete</button>
      </td>
    </tr>`;
  }
  html += '</tbody></table>';
  c.innerHTML = html;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function openAdd() {
  editingPath = null;
  document.getElementById('modal-title').textContent = 'Add Source Profile';
  document.getElementById('f-path').value = '';
  document.getElementById('f-path').disabled = false;
  document.getElementById('f-name').value = '';
  document.getElementById('f-mode').value = 'incremental';
  document.getElementById('f-doctype').value = 'note';
  document.getElementById('f-chunk-size').value = '';
  document.getElementById('f-chunk-overlap').value = '';
  document.getElementById('modal').classList.add('active');
}

async function openEdit(sourcePath) {
  const res = await fetch(API);
  const data = await res.json();
  const profile = data.sources.find(s => s.path === sourcePath);
  if (!profile) return;
  editingPath = sourcePath;
  document.getElementById('modal-title').textContent = 'Edit Source Profile';
  document.getElementById('f-path').value = profile.path;
  document.getElementById('f-path').disabled = true;
  document.getElementById('f-name').value = profile.name ?? '';
  document.getElementById('f-mode').value = profile.mode;
  document.getElementById('f-doctype').value = profile.doc_type;
  document.getElementById('f-chunk-size').value = profile.chunk_size ?? '';
  document.getElementById('f-chunk-overlap').value = profile.chunk_overlap ?? '';
  document.getElementById('modal').classList.add('active');
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
}

async function saveProfile() {
  const body = {
    path: document.getElementById('f-path').value.trim(),
    mode: document.getElementById('f-mode').value,
    doc_type: document.getElementById('f-doctype').value.trim() || 'note',
  };
  const nameVal = document.getElementById('f-name').value.trim();
  if (nameVal) body.name = nameVal;
  const cs = document.getElementById('f-chunk-size').value;
  const co = document.getElementById('f-chunk-overlap').value;
  if (cs) body.chunk_size = parseInt(cs);
  if (co) body.chunk_overlap = parseInt(co);

  if (!body.path) { toast('Path is required', true); return; }

  try {
    let res;
    if (editingPath) {
      res = await fetch(`${API}/${encodeURIComponent(editingPath)}`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
      });
    } else {
      res = await fetch(API, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
      });
    }
    if (!res.ok) {
      const err = await res.json();
      toast(err.detail || 'Error saving profile', true);
      return;
    }
    closeModal();
    toast(editingPath ? 'Profile updated' : 'Profile added');
    loadProfiles();
  } catch (e) {
    toast('Network error', true);
  }
}

async function deleteProfile(sourcePath) {
  if (!confirm(`Remove source profile for "${sourcePath}"?`)) return;
  try {
    const res = await fetch(`${API}/${encodeURIComponent(sourcePath)}`, { method: 'DELETE' });
    if (!res.ok) { toast('Error deleting profile', true); return; }
    toast('Profile removed');
    loadProfiles();
  } catch (e) {
    toast('Network error', true);
  }
}

function toast(msg, isError) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { el.className = 'toast'; }, 2500);
}

// Close modal on overlay click
document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// Close modal on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

loadProfiles();
</script>
</body>
</html>
"""


@app.get(
    "/config",
    response_class=HTMLResponse,
    tags=["Config"],
    summary="Config UI",
    description="Web interface for managing source profiles.",
    include_in_schema=False,
)
async def config_ui() -> HTMLResponse:
    """Serve the source profile management web UI."""
    return HTMLResponse(content=_CONFIG_HTML)


# --- OpenAPI Schema Customization ---


def custom_openapi():
    """Customize OpenAPI schema for better Copilot compatibility."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add server information for local deployment
    openapi_schema["servers"] = [
        {"url": "http://localhost:8420", "description": "Local CoLibri server"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# --- Server Runner ---


def run_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
