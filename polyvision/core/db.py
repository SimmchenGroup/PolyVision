"""
SQLite catalogue for the annotation pipeline.

A single database records every ingested micrograph and its derived artefacts across
five tables: `images` (path, hash, dimensions, class, annotation status, whether it
received manual boxes), `detections` (the bounding boxes produced for an image by a
given method + parameter hash), `crops` (each extracted particle crop with its box and
class), `predictions` (per-box model outputs and fused result), and `splits` (the
train/val/test assignment per dataset version). This provides provenance so every
crop and label is traceable back to its source image and how it was produced.
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

_DB_PATH: Path | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id                INTEGER PRIMARY KEY,
    path              TEXT UNIQUE NOT NULL,
    stem              TEXT NOT NULL,
    class_label       TEXT,
    status            TEXT DEFAULT 'raw',
    annotated_at      DATETIME,
    file_hash         TEXT,
    width             INTEGER,
    height            INTEGER,
    had_manual_boxes  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    method      TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    bboxes_json TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_id, method, params_hash)
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    bbox_index   INTEGER NOT NULL,
    local_probs  TEXT,
    global_probs TEXT,
    yolo_probs   TEXT,
    fusion_probs TEXT,
    final_class  INTEGER,
    confidence   REAL,
    UNIQUE(detection_id, bbox_index)
);

CREATE TABLE IF NOT EXISTS crops (
    id              INTEGER PRIMARY KEY,
    image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    crop_path       TEXT UNIQUE NOT NULL,
    bbox_rc         TEXT NOT NULL,
    class_id        INTEGER NOT NULL,
    manual_override INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS splits (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    split    TEXT NOT NULL,
    version  TEXT NOT NULL,
    PRIMARY KEY (image_id, version)
);
"""


def configure(db_path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _init_db()


def _get_db_path() -> Path:
    if _DB_PATH is not None:
        return _DB_PATH
    return Path(__file__).parent.parent.parent / "data" / "polyvision.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    _get_db_path().parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript(SCHEMA)
        # Safe migration: add column if the DB predates it
        cols = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
        if "had_manual_boxes" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN had_manual_boxes INTEGER DEFAULT 0")
        if "triage_status" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN triage_status TEXT")
        if "triage_missed_json" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN triage_missed_json TEXT")


# ── hashing helpers ───────────────────────────────────────────────────────────

def _file_hash(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _params_hash(*args) -> str:
    return hashlib.md5(
        json.dumps(args, sort_keys=True).encode()
    ).hexdigest()[:16]


def _detection_params_hash(method: str, **kwargs) -> str:
    return _params_hash(method, kwargs)


# ── images ────────────────────────────────────────────────────────────────────

def get_or_create_image(path: str | Path, class_label: str | None = None,
                        width: int | None = None, height: int | None = None) -> int:
    """Return image.id, inserting a row if this path has not been seen before."""
    _init_db()
    path = str(Path(path).resolve())
    stem = Path(path).stem
    with _conn() as conn:
        row = conn.execute("SELECT id FROM images WHERE path=?", (path,)).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO images (path, stem, class_label, width, height) VALUES (?,?,?,?,?)",
            (path, stem, class_label, width, height),
        )
        return cur.lastrowid


def mark_annotated(image_id: int, class_label: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE images SET status='annotated', class_label=?, annotated_at=? WHERE id=?",
            (class_label, datetime.utcnow().isoformat(), image_id),
        )


def mark_skipped(image_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE images SET status='skipped' WHERE id=?", (image_id,))


def get_image_status(path: str | Path) -> str | None:
    """Return status string for a path, or None if not yet registered."""
    _init_db()
    path = str(Path(path).resolve())
    with _conn() as conn:
        row = conn.execute("SELECT status FROM images WHERE path=?", (path,)).fetchone()
        return row["status"] if row else None


# ── detection cache ───────────────────────────────────────────────────────────

def get_cached_detections(image_id: int, method: str, **params) -> list | None:
    """
    Return list of bbox dicts from cache, or None on miss.
    Pass the same keyword args used to produce the detections so the hash matches.
    e.g. get_cached_detections(img_id, 'yolo', conf=0.25)
         get_cached_detections(img_id, 'threshold', otsu_offset=5, min_area=50)
    """
    ph = _detection_params_hash(method, **params)
    with _conn() as conn:
        row = conn.execute(
            "SELECT bboxes_json FROM detections WHERE image_id=? AND method=? AND params_hash=?",
            (image_id, method, ph),
        ).fetchone()
    return json.loads(row["bboxes_json"]) if row else None


def cache_detections(image_id: int, method: str, bboxes: list, **params) -> int:
    """
    Store detection results and return the detections.id.
    bboxes: list of dicts with keys bbox_rc, conf, cls_id.
    """
    ph = _detection_params_hash(method, **params)
    bboxes_json = json.dumps(bboxes)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO detections (image_id, method, params_hash, bboxes_json)
               VALUES (?,?,?,?)
               ON CONFLICT(image_id, method, params_hash)
               DO UPDATE SET bboxes_json=excluded.bboxes_json,
                             created_at=CURRENT_TIMESTAMP""",
            (image_id, method, ph, bboxes_json),
        )
        row = conn.execute(
            "SELECT id FROM detections WHERE image_id=? AND method=? AND params_hash=?",
            (image_id, method, ph),
        ).fetchone()
        return row["id"]


def get_detection_id(image_id: int, method: str, **params) -> int | None:
    ph = _detection_params_hash(method, **params)
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM detections WHERE image_id=? AND method=? AND params_hash=?",
            (image_id, method, ph),
        ).fetchone()
    return row["id"] if row else None


# ── prediction cache ──────────────────────────────────────────────────────────

def get_cached_predictions(detection_id: int, bbox_index: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            """SELECT local_probs, global_probs, yolo_probs, fusion_probs,
                      final_class, confidence
               FROM predictions WHERE detection_id=? AND bbox_index=?""",
            (detection_id, bbox_index),
        ).fetchone()
    if row is None:
        return None
    return {
        "local_probs":  json.loads(row["local_probs"])  if row["local_probs"]  else None,
        "global_probs": json.loads(row["global_probs"]) if row["global_probs"] else None,
        "yolo_probs":   json.loads(row["yolo_probs"])   if row["yolo_probs"]   else None,
        "fusion_probs": json.loads(row["fusion_probs"]) if row["fusion_probs"] else None,
        "final_class":  row["final_class"],
        "confidence":   row["confidence"],
    }


def cache_predictions(detection_id: int, bbox_index: int, *,
                      local_probs=None, global_probs=None, yolo_probs=None,
                      fusion_probs=None, final_class: int | None = None,
                      confidence: float | None = None) -> None:
    def _j(x):
        return json.dumps(x.tolist() if hasattr(x, "tolist") else x) if x is not None else None

    with _conn() as conn:
        conn.execute(
            """INSERT INTO predictions
                   (detection_id, bbox_index, local_probs, global_probs,
                    yolo_probs, fusion_probs, final_class, confidence)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(detection_id, bbox_index) DO UPDATE SET
                   local_probs=excluded.local_probs,
                   global_probs=excluded.global_probs,
                   yolo_probs=excluded.yolo_probs,
                   fusion_probs=excluded.fusion_probs,
                   final_class=excluded.final_class,
                   confidence=excluded.confidence""",
            (detection_id, bbox_index,
             _j(local_probs), _j(global_probs), _j(yolo_probs),
             _j(fusion_probs), final_class, confidence),
        )


# ── crops ─────────────────────────────────────────────────────────────────────

def record_crop(image_id: int, crop_path: str | Path,
                bbox_rc: tuple, class_id: int,
                manual_override: bool = False) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO crops (image_id, crop_path, bbox_rc, class_id, manual_override)
               VALUES (?,?,?,?,?)
               ON CONFLICT(crop_path) DO UPDATE SET
                   class_id=excluded.class_id,
                   manual_override=excluded.manual_override""",
            (image_id, str(Path(crop_path).resolve()),
             json.dumps(list(bbox_rc)), class_id, int(manual_override)),
        )


# ── dataset queries ───────────────────────────────────────────────────────────

def class_counts() -> dict[str, int]:
    """Return {class_label: annotated_image_count}."""
    _init_db()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT class_label, COUNT(*) as n FROM images
               WHERE status='annotated' AND class_label IS NOT NULL
               GROUP BY class_label""",
        ).fetchall()
    return {r["class_label"]: r["n"] for r in rows}


def crop_counts() -> dict[str, int]:
    """Return {class_label: crop_count}."""
    _init_db()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT i.class_label, COUNT(c.id) as n
               FROM crops c JOIN images i ON c.image_id=i.id
               WHERE i.class_label IS NOT NULL
               GROUP BY i.class_label""",
        ).fetchall()
    return {r["class_label"]: r["n"] for r in rows}


def annotation_progress() -> dict[str, int]:
    """Return {status: count} across all registered images."""
    _init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM images GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def set_triage(image_id: int, status: str, missed_boxes: list) -> None:
    """
    Record pre-triage result for an image.
    status: 'clean' or 'flagged'.
    missed_boxes: list of (min_r, min_c, max_r, max_c) tuples that threshold
    found but YOLO missed (empty for clean images).
    """
    with _conn() as conn:
        conn.execute(
            "UPDATE images SET triage_status=?, triage_missed_json=? WHERE id=?",
            (status, json.dumps([list(b) for b in missed_boxes]), image_id),
        )


def get_triage_by_path(path: str | Path) -> tuple[str | None, list]:
    """
    Return (triage_status, missed_boxes) for an image path.
    missed_boxes is a list of (min_r, min_c, max_r, max_c) tuples.
    Returns (None, []) if the image is not registered or not triaged.
    """
    _init_db()
    path = str(Path(path).resolve())
    with _conn() as conn:
        row = conn.execute(
            "SELECT triage_status, triage_missed_json FROM images WHERE path=?",
            (path,),
        ).fetchone()
    if row is None or row["triage_status"] is None:
        return None, []
    boxes = json.loads(row["triage_missed_json"]) if row["triage_missed_json"] else []
    return row["triage_status"], [tuple(int(v) for v in b) for b in boxes]


def triage_summary() -> dict:
    """Return counts of clean / flagged / untriaged across all registered images."""
    _init_db()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT COALESCE(triage_status, 'untriaged') as s, COUNT(*) as n
               FROM images GROUP BY s"""
        ).fetchall()
    out = {"clean": 0, "flagged": 0, "untriaged": 0}
    for r in rows:
        out[r["s"]] = r["n"]
    return out


def mark_had_manual_boxes(image_id: int, had_manual: bool) -> None:
    """Record whether the user drew any manual boxes for this image at save time."""
    with _conn() as conn:
        conn.execute(
            "UPDATE images SET had_manual_boxes=? WHERE id=?",
            (int(had_manual), image_id),
        )


def intervention_stats() -> dict:
    """
    Return stats on how often YOLO needed manual assistance.
    Only counts images with status='annotated'.
    """
    _init_db()
    with _conn() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN had_manual_boxes=0 THEN 1 ELSE 0 END) as auto_only,
                SUM(CASE WHEN had_manual_boxes=1 THEN 1 ELSE 0 END) as had_manual
               FROM images WHERE status='annotated'"""
        ).fetchone()
    total = row["total"] or 0
    auto = row["auto_only"] or 0
    manual = row["had_manual"] or 0
    return {
        "total": total,
        "auto_only": auto,
        "had_manual": manual,
        "auto_rate": round(auto / total, 3) if total else 0.0,
    }
