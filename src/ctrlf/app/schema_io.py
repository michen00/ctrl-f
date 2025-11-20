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
from json_schema_to_pydantic import create_model as json_schema_create_model
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

    Uses json-schema-to-pydantic library to handle complex schemas including
    nested objects, references, and combiners.

    Returns:
        Pydantic model class

    Raises:
        SchemaError: If schema conversion fails
    """
    try:
        # Use json-schema-to-pydantic library to convert schema
        # This handles nested objects, references, combiners, etc.
        return json_schema_create_model(schema)  # type: ignore[no-any-return]
    except Exception as e:
        # Wrap any exceptions from the library in SchemaError
        msg = f"Failed to convert JSON Schema to Pydantic model: {e}"
        raise SchemaError(msg) from e


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

        # Find all BaseModel subclasses
        # Iterate through __dict__ to preserve definition order (Python 3.7+)
        for name in module.__dict__:
            obj = getattr(module, name, None)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                # Keep updating to get the last one defined
                model_class = obj

        if model_class is None:
            msg = "No BaseModel subclass found in code"
            raise SchemaError(msg)  # noqa: TRY301

        # Nested structures are now supported via json-schema-to-pydantic
        # No need to check for nested structures here
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


def _extend_nested_model(nested_model_cls: type[BaseModel]) -> type[BaseModel]:
    """Recursively extend a nested BaseModel to convert all fields to arrays.

    Returns:
        Extended model with all fields as List[type]
    """
    # Recursively extend the nested model
    return extend_schema(nested_model_cls)


def _convert_to_extended_type(
    _field_name: str,
    original_type: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Convert a field type to its extended array form.

    Handles nested BaseModel types by recursively extending them.

    Returns:
        Extended type (List[type] or List[type] | None)
    """
    origin = get_origin(original_type)

    # If already a list, check if it contains nested models
    if origin is list:
        args = get_args(original_type)
        if args:
            item_type = args[0]
            # If list contains a BaseModel, recursively extend it
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                extended_nested = _extend_nested_model(item_type)
                return list[extended_nested]  # type: ignore[valid-type]
        return original_type

    # Handle Optional/Union types
    if origin is Union or origin is type(None):
        args = get_args(original_type)
        # Filter out None
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            # Take the first non-None type
            base_type = non_none_args[0]
            # Check if base_type is a BaseModel (nested object)
            if isinstance(base_type, type) and issubclass(base_type, BaseModel):
                # Recursively extend the nested model
                extended_nested = _extend_nested_model(base_type)
                return list[extended_nested] | None  # type: ignore[valid-type]
            # Check if base_type is a List of BaseModel
            base_origin = get_origin(base_type)
            if base_origin is list:
                base_args = get_args(base_type)
                if base_args:
                    base_item_type = base_args[0]
                    if isinstance(base_item_type, type) and issubclass(
                        base_item_type, BaseModel
                    ):
                        extended_nested = _extend_nested_model(base_item_type)
                        return list[extended_nested] | None  # type: ignore[valid-type]
            return list[base_type] | None  # type: ignore[valid-type]
        return list[str] | None

    # Check if type is a BaseModel (nested object)
    if isinstance(original_type, type) and issubclass(original_type, BaseModel):
        # Recursively extend the nested model
        extended_nested = _extend_nested_model(original_type)
        return list[extended_nested]  # type: ignore[valid-type]

    # Simple type - convert to List[type]
    return list[original_type]


def extend_schema(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Create Extended Schema by coercing all fields to arrays.

    Supports nested objects by recursively extending nested BaseModel types.

    Returns:
        Extended model with all fields as List[type]
    """
    field_definitions: dict[str, tuple[Any, Any]] = {}
    model_fields = model_cls.model_fields

    for field_name, field_info in model_fields.items():
        original_type = field_info.annotation

        # Convert type to List[type] (handles nested objects recursively)
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
            # For nested BaseModel defaults, wrap in list:
            # default=Address(...) -> default=[Address(...)]
            field_definitions[field_name] = (extended_type, [default_value])

    # Create extended model
    extended_model_name = f"Extended{model_cls.__name__}"
    return create_model(extended_model_name, **field_definitions)  # type: ignore[no-any-return, call-overload]
