"""Unit tests for schema I/O operations."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from ctrlf.app.errors import SchemaError
from ctrlf.app.schema_io import (
    convert_json_schema_to_pydantic,
    extend_schema,
    import_pydantic_model,
    validate_json_schema,
)


class TestValidateJsonSchema:
    """Tests for JSON Schema validation (T041)."""

    def test_valid_json_schema(self) -> None:
        """Test validating a valid JSON Schema."""
        schema_json = json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["name"],
            }
        )
        result = validate_json_schema(schema_json)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "email" in result["properties"]

    def test_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises SchemaError."""
        with pytest.raises(SchemaError, match="Invalid JSON"):
            validate_json_schema("{ invalid json }")

    def test_invalid_schema_structure_raises_error(self) -> None:
        """Test that invalid JSON Schema structure raises SchemaError."""
        schema_json = json.dumps({"type": "invalid_type"})
        # jsonschema raises its own SchemaError which gets wrapped
        with pytest.raises(SchemaError):
            validate_json_schema(schema_json)

    def test_missing_type_raises_error(self) -> None:
        """Test that schema without type raises error."""
        # A schema without type at root level is actually valid in JSON Schema
        # (it's just not a valid object schema).
        # Let's test with an actually invalid schema.
        schema_json = json.dumps({"type": "object", "properties": "invalid"})
        with pytest.raises(SchemaError):
            validate_json_schema(schema_json)


class TestConvertJsonSchemaToPydantic:
    """Tests for JSON Schema to Pydantic conversion (T042)."""

    def test_convert_simple_schema(self) -> None:
        """Test converting a simple flat JSON Schema to Pydantic model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"},
            },
        }
        model_cls = convert_json_schema_to_pydantic(schema)
        assert issubclass(model_cls, BaseModel)

        # Test instantiation
        instance = model_cls(name="John", age=30, email="john@example.com")
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert instance.age == 30  # type: ignore[attr-defined]
        assert instance.email == "john@example.com"  # type: ignore[attr-defined]

    def test_convert_with_required_fields(self) -> None:
        """Test converting schema with required fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "optional": {"type": "string"},
            },
            "required": ["name"],
        }
        model_cls = convert_json_schema_to_pydantic(schema)

        # Required field should be required
        instance = model_cls(name="John")
        assert instance.name == "John"  # type: ignore[attr-defined]

        # Missing required field should fail
        with pytest.raises(ValidationError):
            model_cls()

    def test_convert_with_default_values(self) -> None:
        """Test converting schema with default values."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 0},
            },
        }
        model_cls = convert_json_schema_to_pydantic(schema)

        instance = model_cls(name="Test")
        assert instance.count == 0  # type: ignore[attr-defined]

    def test_nested_object_supported(self) -> None:
        """Test that nested objects are now supported."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        model_cls = convert_json_schema_to_pydantic(schema)
        assert issubclass(model_cls, BaseModel)

        # Test instantiation with nested object
        instance = model_cls(
            name="John",
            address={"street": "123 Main St", "city": "New York"},
        )
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert instance.address.street == "123 Main St"  # type: ignore[attr-defined]
        assert instance.address.city == "New York"  # type: ignore[attr-defined]

    def test_nested_array_of_objects_supported(self) -> None:
        """Test that nested arrays of objects are now supported."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
        }
        model_cls = convert_json_schema_to_pydantic(schema)
        assert issubclass(model_cls, BaseModel)

        # Test instantiation with nested array of objects
        instance = model_cls(
            name="John",
            items=[{"value": "item1"}, {"value": "item2"}],
        )
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert len(instance.items) == 2  # type: ignore[attr-defined]
        assert instance.items[0].value == "item1"  # type: ignore[attr-defined]
        assert instance.items[1].value == "item2"  # type: ignore[attr-defined]

    def test_primitive_array_allowed(self) -> None:
        """Test that primitive arrays are allowed."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        model_cls = convert_json_schema_to_pydantic(schema)
        assert issubclass(model_cls, BaseModel)

        instance = model_cls(tags=["tag1", "tag2"])
        assert instance.tags == ["tag1", "tag2"]  # type: ignore[attr-defined]


class TestImportPydanticModel:
    """Tests for Pydantic model import (T043)."""

    def test_import_simple_model(self) -> None:
        """Test importing a simple Pydantic model."""
        code = """
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
"""
        model_cls = import_pydantic_model(code)
        assert issubclass(model_cls, BaseModel)

        instance = model_cls(name="John", age=30)
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert instance.age == 30  # type: ignore[attr-defined]

    def test_import_model_with_optional_fields(self) -> None:
        """Test importing model with optional fields."""
        code = """
from pydantic import BaseModel
from typing import Optional

class Person(BaseModel):
    name: str
    email: Optional[str] = None
"""
        model_cls = import_pydantic_model(code)
        instance = model_cls(name="John")
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert instance.email is None  # type: ignore[attr-defined]

    def test_invalid_python_syntax_raises_error(self) -> None:
        """Test that invalid Python syntax raises SchemaError."""
        code = "class Person(BaseModel:  # syntax error"
        with pytest.raises(SchemaError, match="Invalid Python syntax"):
            import_pydantic_model(code)

    def test_no_basemodel_raises_error(self) -> None:
        """Test that code without BaseModel raises SchemaError."""
        code = """
class NotAModel:
    pass
"""
        with pytest.raises(SchemaError, match="No BaseModel subclass found"):
            import_pydantic_model(code)

    def test_multiple_basemodels_uses_last_one(self) -> None:
        """Test that multiple BaseModel classes use the last one defined."""
        code = """
from pydantic import BaseModel

class Model1(BaseModel):
    field1: str

class Model2(BaseModel):
    field2: str
"""
        # Should return the last BaseModel defined (Model2)
        model_cls = import_pydantic_model(code)
        assert model_cls.__name__ == "Model2"

        # Verify it's the correct model
        instance = model_cls(field2="test")
        assert instance.field2 == "test"  # type: ignore[attr-defined]

        # Verify Model1 is not returned
        assert model_cls.__name__ != "Model1"


class TestExtendSchema:
    """Tests for schema extension (array coercion) (T044)."""

    def test_extend_simple_schema(self) -> None:
        """Test extending a simple schema with primitive fields."""
        code = """
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Extended model should have all fields as lists
        instance = extended_model(
            name=["John"],
            age=[30],
            email=["john@example.com"],
        )
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert instance.age == [30]  # type: ignore[attr-defined]
        assert instance.email == ["john@example.com"]  # type: ignore[attr-defined]

    def test_extend_schema_with_optional_fields(self) -> None:
        """Test extending schema with optional fields."""
        code = """
from pydantic import BaseModel
from typing import Optional

class Person(BaseModel):
    name: str
    email: Optional[str] = None
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Optional fields should become Optional[List[type]]
        instance = extended_model(name=["John"], email=None)
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert instance.email is None  # type: ignore[attr-defined]

        instance2 = extended_model(name=["John"], email=["john@example.com"])
        assert instance2.email == ["john@example.com"]  # type: ignore[attr-defined]

    def test_extend_schema_with_pipe_union_syntax(self) -> None:
        """Test extending schema with Python 3.11+ | None syntax."""
        code = """
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    email: str | None = None
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Optional fields should become Optional[List[type]]
        instance = extended_model(name=["John"], email=None)
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert instance.email is None  # type: ignore[attr-defined]

        instance2 = extended_model(name=["John"], email=["john@example.com"])
        assert instance2.email == ["john@example.com"]  # type: ignore[attr-defined]

    def test_extend_schema_with_existing_arrays(self) -> None:
        """Test extending schema where fields are already arrays."""
        code = """
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    tags: List[str]
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Array fields should remain arrays (not become List[List[str]])
        instance = extended_model(name=["John"], tags=["tag1", "tag2"])
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert instance.tags == ["tag1", "tag2"]  # type: ignore[attr-defined]

    def test_extend_schema_preserves_default_values(self) -> None:
        """Test that extend_schema preserves custom default values."""
        code = """
from pydantic import BaseModel

class Person(BaseModel):
    name: str = "Unknown"
    age: int = 0
    email: str
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Extended model should preserve defaults by wrapping in lists
        instance = extended_model(email=["test@example.com"])
        assert instance.name == ["Unknown"]  # type: ignore[attr-defined]
        assert instance.age == [0]  # type: ignore[attr-defined]
        assert instance.email == ["test@example.com"]  # type: ignore[attr-defined]

        # Should still allow overriding defaults
        instance2 = extended_model(name=["John"], age=[30], email=["john@example.com"])
        assert instance2.name == ["John"]  # type: ignore[attr-defined]
        assert instance2.age == [30]  # type: ignore[attr-defined]

    def test_extend_schema_validation(self) -> None:
        """Test that extended schema validates correctly."""
        code = """
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
"""
        original_model = import_pydantic_model(code)
        extended_model = extend_schema(original_model)

        # Should accept list values
        instance = extended_model(name=["John", "Jane"], age=[30, 25])
        assert instance.name == ["John", "Jane"]  # type: ignore[attr-defined]
        assert instance.age == [30, 25]  # type: ignore[attr-defined]

        # Should reject non-list values
        with pytest.raises(ValidationError):
            extended_model(name="John", age=30)


class TestNestedSchemaExtension:
    """Tests for nested schema extension (array coercion with nested objects)."""

    def test_extend_schema_with_nested_object(self) -> None:
        """Test extending schema with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        original_model = convert_json_schema_to_pydantic(schema)
        extended_model = extend_schema(original_model)

        # Extended model should have nested fields as arrays
        instance = extended_model(
            name=["John"],
            address=[{"street": ["123 Main St"], "city": ["New York"]}],
        )
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert len(instance.address) == 1  # type: ignore[attr-defined]
        assert instance.address[0].street == ["123 Main St"]  # type: ignore[attr-defined]
        assert instance.address[0].city == ["New York"]  # type: ignore[attr-defined]

    def test_extend_schema_with_nested_array_of_objects(self) -> None:
        """Test extending schema with nested arrays of objects."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
        }
        original_model = convert_json_schema_to_pydantic(schema)
        extended_model = extend_schema(original_model)

        # Extended model should have nested array fields as arrays
        # When extending list[ItemModel], it becomes list[ExtendedItemModel]
        # (not list[list[ExtendedItemModel]])
        instance = extended_model(
            name=["John"],
            items=[{"value": ["item1"]}, {"value": ["item2"]}],
        )
        assert instance.name == ["John"]  # type: ignore[attr-defined]
        assert len(instance.items) == 2  # type: ignore[attr-defined]
        assert instance.items[0].value == ["item1"]  # type: ignore[attr-defined]
        assert instance.items[1].value == ["item2"]  # type: ignore[attr-defined]


class TestImportPydanticModelRestrictions:
    """Tests for import_pydantic_model restrictions (single model only)."""

    def test_multiple_basemodels_uses_last_one(self) -> None:
        """Test that import_pydantic_model uses the last BaseModel defined.

        When multiple BaseModel classes are present, the last one defined
        is treated as the top-level schema.
        """
        code = """
from pydantic import BaseModel

class Address(BaseModel):
    street: str

class Person(BaseModel):
    name: str
    address: Address
"""
        # Should return the last BaseModel defined (Person)
        model_cls = import_pydantic_model(code)
        assert model_cls.__name__ == "Person"

        # Verify it works correctly with nested models
        # Use dict for nested model since Address class is not in test scope
        instance = model_cls(name="John", address={"street": "123 Main St"})
        assert instance.name == "John"  # type: ignore[attr-defined]
        assert instance.address.street == "123 Main St"  # type: ignore[attr-defined]
