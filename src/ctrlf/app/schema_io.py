"""Schema I/O operations for JSON Schema and Pydantic models."""

from __future__ import annotations

__all__ = (
    "convert_json_schema_to_pydantic",
    "extend_schema",
    "import_pydantic_model",
    "validate_json_schema",
)

import importlib.util
import json
import sys
from typing import Any, Union, get_args, get_origin

import jsonschema
from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined

from ctrlf.app.errors import SchemaError


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
    except Exception as e:
        # Catch both ValidationError and SchemaError from jsonschema
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

    # Map JSON Schema types to Python/Pydantic types
    type_mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
    }

    # Build field definitions for create_model
    field_definitions: dict[str, tuple[Any, Any]] = {}
    required_fields = set(schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type")
        default_value = prop_schema.get("default", ...)

        # Handle arrays
        if prop_type == "array":
            items_schema = prop_schema.get("items", {})
            item_type = items_schema.get("type", "string")
            if item_type not in type_mapping:
                msg = (
                    f"Unsupported array item type: {item_type} for field '{prop_name}'"
                )
                raise SchemaError(msg)
            python_type: Any = list[type_mapping[item_type]]  # type: ignore[valid-type]
        elif prop_type in type_mapping:
            python_type = type_mapping[prop_type]
        else:
            msg = f"Unsupported type: {prop_type} for field '{prop_name}'"
            raise SchemaError(msg)

        # Set default if provided, or make optional if not required
        if default_value is not ...:
            # Has explicit default value
            field_definitions[prop_name] = (python_type, default_value)
        elif prop_name not in required_fields:
            # Not required and no default - make optional
            optional_type: Any = python_type | None
            field_definitions[prop_name] = (optional_type, None)
        else:
            # Required field
            field_definitions[prop_name] = (python_type, ...)

    # Create the model class dynamically
    model_name = schema.get("title", "GeneratedModel") or "GeneratedModel"
    return create_model(model_name, **field_definitions)  # type: ignore[no-any-return, call-overload]


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
    model_class: type[BaseModel] | None = None
    module_added = False
    try:
        # Create a temporary module
        spec = importlib.util.spec_from_loader("temp_model", loader=None)
        if spec is None:
            msg = "Failed to create module spec"
            raise SchemaError(msg)  # noqa: TRY301

        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_model"] = module
        module_added = True

        # Execute the code
        exec(code, module.__dict__)  # noqa: S102  # nosec B102

        # Find the BaseModel subclass
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
    except SchemaError:
        # Re-raise SchemaError directly to preserve original error context
        raise
    except SyntaxError as e:
        msg = f"Invalid Python syntax: {e}"
        raise SchemaError(msg) from e
    except Exception as e:
        # Only catch unexpected exceptions, not intentionally raised SchemaErrors
        msg = f"Failed to import model: {e}"
        raise SchemaError(msg) from e
    finally:
        # Always clean up the temporary module to prevent namespace pollution
        if module_added and "temp_model" in sys.modules:
            del sys.modules["temp_model"]

    # Return the model class after cleanup
    # Note: model_class guaranteed to be set (would have raised earlier)
    assert model_class is not None, "Model class not found"  # noqa: S101  # nosec B101
    return model_class


def _check_nested_structure(
    field_name: str,
    field_type: Any,  # noqa: ANN401
) -> None:
    """Check if field type contains nested structures (not allowed in v0).

    Args:
        field_name: Name of the field being checked
        field_type: Type annotation to check

    Raises:
        SchemaError: If nested structures are detected
    """
    # Check if type is a BaseModel subclass (nested object)
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        msg = f"Nested objects not supported in v0: field '{field_name}'"
        raise SchemaError(msg)

    # Check if type is a List of BaseModel (nested array of objects)
    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        if args:
            item_type = args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                msg = (
                    f"Nested arrays of objects not supported in v0: "
                    f"field '{field_name}'"
                )
                raise SchemaError(msg)


def _convert_to_extended_type(
    field_name: str,
    original_type: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Convert a field type to its extended array form.

    Args:
        field_name: Name of the field
        original_type: Original type annotation

    Returns:
        Extended type (List[type] or List[type] | None)

    Raises:
        SchemaError: If nested structures are detected
    """
    origin = get_origin(original_type)

    # If already a list, keep it as is
    if origin is list:
        return original_type

    # Handle Optional/Union types
    if origin is Union or origin is type(None):
        args = get_args(original_type)
        # Filter out None
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            # Take the first non-None type
            base_type = non_none_args[0]
            # Re-validate base_type for nested structures
            # (Union/Optional can wrap BaseModel)
            _check_nested_structure(field_name, base_type)
            # Check if base_type is a List of BaseModel
            base_origin = get_origin(base_type)
            if base_origin is list:
                base_args = get_args(base_type)
                if base_args:
                    base_item_type = base_args[0]
                    _check_nested_structure(field_name, base_item_type)
            return list[base_type] | None  # type: ignore[valid-type]
        return list[str] | None

    # Simple type - convert to List[type]
    return list[original_type]


def extend_schema(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Create Extended Schema by coercing all fields to arrays.

    Args:
        model_cls: Original Pydantic model

    Returns:
        Extended model with all fields as List[type]

    Raises:
        SchemaError: If model contains nested objects/arrays
    """
    field_definitions: dict[str, tuple[Any, Any]] = {}
    model_fields = model_cls.model_fields

    for field_name, field_info in model_fields.items():
        original_type = field_info.annotation

        # Check for nested structures (v0 limitation)
        _check_nested_structure(field_name, original_type)

        # Convert type to List[type]
        extended_type = _convert_to_extended_type(field_name, original_type)

        # Preserve default value if present, wrapping in list for array types
        # Check for PydanticUndefined (no default) or Ellipsis (required field)
        default_value = field_info.default
        if default_value is PydanticUndefined or default_value is ...:
            # Required field - no default
            field_definitions[field_name] = (extended_type, ...)
        elif default_value is None:
            # None default - preserve None (for Optional types)
            field_definitions[field_name] = (extended_type, None)
        else:
            # Has explicit non-None default value - wrap in list to match array type
            # e.g., default="value" becomes default=["value"]
            field_definitions[field_name] = (extended_type, [default_value])

    # Create extended model
    extended_model_name = f"Extended{model_cls.__name__}"
    return create_model(extended_model_name, **field_definitions)  # type: ignore[no-any-return, call-overload]
