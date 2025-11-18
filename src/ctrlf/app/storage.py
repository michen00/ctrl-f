"""TinyDB storage adapter for persisted records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinydb import Query, TinyDB  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from ctrlf.app.models import PersistedRecord

__all__ = ("export_record", "get_storage_path", "save_record")


def get_storage_path() -> Path:
    """Get the default storage path for TinyDB.

    Returns:
        Path to the storage directory
    """
    storage_dir = Path.home() / ".ctrlf" / "db"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def save_record(record: PersistedRecord, table_name: str | None = None) -> str:
    """Save a persisted record to TinyDB.

    Args:
        record: Validated record to save
        table_name: Optional table name (defaults to schema hash from record)

    Returns:
        Record ID of saved record

    Raises:
        ValueError: If record validation fails
        RuntimeError: If database write fails
    """
    # Use schema version as table name if not provided
    if table_name is None:
        # Extract schema version from audit trail if available
        schema_version = record.audit.get("schema_version", "default")
        table_name = f"schema_{schema_version}"

    storage_path = get_storage_path()
    db_file = storage_path / f"{table_name}.json"

    db: TinyDB | None = None
    try:
        db = TinyDB(str(db_file))
        table = db.table("records")

        # Convert record to dict for storage
        record_dict = record.model_dump()

        # Insert record (TinyDB will use record_id as doc_id)
        table.insert(record_dict)
    except Exception as e:
        msg = f"Failed to save record to database: {e}"
        raise RuntimeError(msg) from e
    finally:
        if db is not None:
            db.close()

    return record.record_id


def export_record(record_id: str, table_name: str | None = None) -> dict[str, Any]:
    """Export a record as JSON-serializable dictionary.

    Args:
        record_id: Record identifier
        table_name: Optional table name

    Returns:
        JSON-serializable record data

    Raises:
        KeyError: If record not found
    """
    if table_name is None:
        # Search all tables for the record
        storage_path = get_storage_path()
        for db_file in storage_path.glob("*.json"):
            db = TinyDB(str(db_file))
            table = db.table("records")
            record_query = Query()
            results = table.search(record_query.record_id == record_id)
            db.close()

            if results:
                return results[0]  # type: ignore[no-any-return]

        msg = f"Record not found: {record_id}"
        raise KeyError(msg)

    # Search specific table
    storage_path = get_storage_path()
    db_file = storage_path / f"{table_name}.json"

    if not db_file.exists():
        msg = f"Table not found: {table_name}"
        raise KeyError(msg)

    db = TinyDB(str(db_file))
    table = db.table("records")
    record_query = Query()
    results = table.search(record_query.record_id == record_id)
    db.close()

    if not results:
        msg = f"Record not found: {record_id}"
        raise KeyError(msg)

    return results[0]  # type: ignore[no-any-return]
