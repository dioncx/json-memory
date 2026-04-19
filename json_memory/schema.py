"""
Schema — Define and validate memory structure.

Ensures your memory follows a consistent shape across sessions.
"""

import json
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory


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

    def validate_memory(self, mem: "Memory", strict: bool = False) -> bool:
        """Validate a Memory instance directly.

        Args:
            mem: The Memory instance to validate.
            strict: If True, extra keys in memory not in schema are rejected.
        """
        return self.validate(mem.to_dict(), strict=strict)

    def _check(self, data: Any, template: Any, strict: bool = False) -> bool:
        if isinstance(template, dict):
            if not isinstance(data, dict):
                return False
            if strict:
                template_keys_clean = {k[1:] if k.startswith("!") else k for k in template.keys()}
                extra = set(data.keys()) - template_keys_clean
                if extra:
                    return False
            for key, expected in template.items():
                is_required = key.startswith("!")
                clean_key = key[1:] if is_required else key
                
                if clean_key not in data:
                    if is_required:
                        return False
                    continue
                
                if not self._check(data[clean_key], expected, strict):
                    return False
            return True

        if isinstance(template, list):
            if not isinstance(data, list):
                return False
            if not template:  # Simple "list" check
                return True
            # Template list contains type/sub-template for items
            item_template = template[0]
            for item in data:
                if not self._check(item, item_template, strict):
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
            skeleton = {}
            for k, v in template.items():
                clean_k = k[1:] if k.startswith("!") else k
                skeleton[clean_k] = self._skeleton(v)
            return skeleton
        if isinstance(template, list):
            return []
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
            is_required = key.startswith("!")
            clean_key = key[1:] if is_required else key
            path = f"{prefix}.{clean_key}" if prefix else clean_key
            
            if clean_key not in data:
                missing.append(path)
            elif isinstance(expected, dict) and isinstance(data[clean_key], dict):
                missing.extend(self._missing(data[clean_key], expected, prefix=path))
        return missing

    def _extra(self, data: dict, template: dict, prefix: str = "") -> list:
        extra = []
        if not isinstance(data, dict):
            return extra
        for key in data:
            path = f"{prefix}.{key}" if prefix else key
            
            # Find key in template (with or without '!')
            template_key = key
            if key not in template and f"!{key}" in template:
                template_key = f"!{key}"
            
            if template_key not in template:
                extra.append(path)
            elif isinstance(data[key], dict) and isinstance(template.get(template_key), dict):
                extra.extend(self._extra(data[key], template[template_key], prefix=path))
        return extra

    def export(self) -> str:
        """Export schema as JSON string."""
        return json.dumps(self._template, separators=(",", ":"))

    def __repr__(self) -> str:
        keys = list(self._template.keys()) if isinstance(self._template, dict) else []
        return f"Schema(keys={keys})"
