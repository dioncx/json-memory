"""
Schema — Define and validate memory structure.

Ensures your memory follows a consistent shape across sessions.
"""

import json
from typing import Any, Optional


class Schema:
    """Define a memory schema and validate data against it.

    Example:
        >>> schema = Schema({
        ...     "u": {"n": "str", "tz": "str", "tech": "str"},
        ...     "bot": {"bin": {"rst": "str", "wl": "str", "bal": "str"}},
        ... })
        >>> schema.validate({"u": {"n": "Alice", "tz": "UTC"}})
        True
        >>> schema.validate({"u": {"n": 123}})
        False
    """

    def __init__(self, template: dict):
        """Create a schema from a template dict.

        Values in the template define expected types:
        - "str" → string
        - "int" → integer
        - "float" → number
        - "bool" → boolean
        - "list" → array
        - "dict" → object
        - "any" → anything
        - nested dict → nested structure
        """
        self._template = template

    def validate(self, data: dict, strict: bool = False) -> bool:
        """Validate data against the schema.

        Args:
            data: The data dict to validate.
            strict: If True, extra keys not in schema are rejected.

        Returns:
            True if valid, False otherwise.
        """
        return self._check(data, self._template, strict=strict)

    def _check(self, data: Any, template: Any, strict: bool = False) -> bool:
        if isinstance(template, dict):
            if not isinstance(data, dict):
                return False
            if strict:
                extra = set(data.keys()) - set(template.keys())
                if extra:
                    return False
            for key, expected in template.items():
                if key not in data:
                    continue  # Missing optional keys are OK
                if not self._check(data[key], expected, strict):
                    return False
            return True

        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": object,
        }
        expected_type = type_map.get(template, object)
        return isinstance(data, expected_type)

    def defaults(self) -> dict:
        """Generate a skeleton dict matching the schema with None values."""
        return self._skeleton(self._template)

    def _skeleton(self, template: Any) -> Any:
        if isinstance(template, dict):
            return {k: self._skeleton(v) for k, v in template.items()}
        return None

    def diff(self, data: dict) -> dict:
        """Find what's missing or extra compared to the schema.

        Returns:
            Dict with 'missing' (keys in schema but not data)
            and 'extra' (keys in data but not schema).
        """
        return {
            "missing": self._missing(data, self._template),
            "extra": self._extra(data, self._template),
        }

    def _missing(self, data: dict, template: dict, prefix: str = "") -> list:
        missing = []
        if not isinstance(template, dict):
            return missing
        for key, expected in template.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in data:
                missing.append(path)
            elif isinstance(expected, dict) and isinstance(data[key], dict):
                missing.extend(self._missing(data[key], expected, prefix=path))
        return missing

    def _extra(self, data: dict, template: dict, prefix: str = "") -> list:
        extra = []
        if not isinstance(data, dict):
            return extra
        for key in data:
            path = f"{prefix}.{key}" if prefix else key
            if key not in template:
                extra.append(path)
            elif isinstance(data[key], dict) and isinstance(template.get(key), dict):
                extra.extend(self._extra(data[key], template[key], prefix=path))
        return extra

    def export(self) -> str:
        """Export schema as JSON string."""
        return json.dumps(self._template, separators=(",", ":"))

    def __repr__(self) -> str:
        keys = list(self._template.keys()) if isinstance(self._template, dict) else []
        return f"Schema(keys={keys})"
