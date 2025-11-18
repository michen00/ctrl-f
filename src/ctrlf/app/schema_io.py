"""Schema I/O operations for JSON Schema and Pydantic models."""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any

import jsonschema
from pydantic import BaseModel

from ctrlf.app.errors import SchemaError

__all__ = (
    "convert_json_schema_to_pydantic",
    "extend_schema",
    "import_pydantic_model",
    "validate_json_schema",
)


def validate_json_schema(schema_json: str) -> dict[str, Any]:
    """Validate JSON Schema format and return parsed schema.

    Args:
        schema_json: JSON Schema as string

    Returns:
        Validated and parsed JSON Schema

    Raises:
        SchemaError: If schema is invalid or JSON is malformed
    """
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {e}"
        raise SchemaError(msg) from e

    try:
        jsonschema.Draft7Validator.check_schema(schema)
    except jsonschema.ValidationError as e:
        msg = f"Invalid JSON Schema: {e}"
        raise SchemaError(msg) from e

    return schema  # type: ignore[no-any-return]


def convert_json_schema_to_pydantic(schema: dict[str, Any]) -> type[BaseModel]:
    """Convert JSON Schema to Pydantic v2 model class.

    Args:
        schema: Validated JSON Schema

    Returns:
        Pydantic model class

    Raises:
        SchemaError: If schema contains nested objects/arrays or types cannot be mapped
    """
    # Check for nested structures (v0 limitation)
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type")
        if prop_type == "object":
            msg = f"Nested objects not supported in v0: field '{prop_name}'"
            raise SchemaError(msg)
        if prop_type == "array":
            items = prop_schema.get("items", {})
            if items.get("type") == "object":
                msg = (
                    f"Nested arrays of objects not supported in v0: field '{prop_name}'"
                )
                raise SchemaError(msg)

    # TODO: Implement full JSON Schema to Pydantic conversion  # noqa: FIX002
    # For now, return a placeholder that will be implemented in User Story 2
    msg = "JSON Schema to Pydantic conversion not yet implemented"
    raise NotImplementedError(msg)


def import_pydantic_model(code: str) -> type[BaseModel]:
    """Import Pydantic model from Python code string.

    Args:
        code: Python code containing Pydantic model class definition

    Returns:
        Pydantic model class

    Raises:
        SchemaError: If code is invalid Python, model cannot be imported,
            or contains nested structures
    """
    try:
        # Create a temporary module
        spec = importlib.util.spec_from_loader("temp_model", loader=None)
        if spec is None:
            msg = "Failed to create module spec"
            raise SchemaError(msg)  # noqa: TRY301

        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_model"] = module

        # Execute the code
        exec(code, module.__dict__)  # noqa: S102

        # Find the BaseModel subclass
        model_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                if model_class is not None:
                    msg = "Multiple BaseModel subclasses found, expected exactly one"
                    raise SchemaError(msg)  # noqa: TRY301
                model_class = obj

        if model_class is None:
            msg = "No BaseModel subclass found in code"
            raise SchemaError(msg)  # noqa: TRY301

        # Check for nested structures (v0 limitation)
        # This is a simplified check - full validation would require
        # inspecting field types
        # TODO: Implement full nested structure detection  # noqa: FIX002
    except SyntaxError as e:
        msg = f"Invalid Python syntax: {e}"
        raise SchemaError(msg) from e
    except Exception as e:
        msg = f"Failed to import model: {e}"
        raise SchemaError(msg) from e
    else:
        # Clean up and return on success
        del sys.modules["temp_model"]
        return model_class


def extend_schema(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Create Extended Schema by coercing all fields to arrays.

    Args:
        model_cls: Original Pydantic model

    Returns:
        Extended model with all fields as List[type]

    Raises:
        SchemaError: If model contains nested objects/arrays
    """
    # TODO: Implement schema extension  # noqa: FIX002
    # This will be fully implemented in User Story 2
    msg = "Schema extension not yet implemented"
    raise NotImplementedError(msg)
