import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any

CONFIG_DIR = Path(__file__).resolve().parents[3] / 'config'

@lru_cache(maxsize=1)
def load_all_algorithm_schemas() -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    if not CONFIG_DIR.exists():
        return schemas
    for p in CONFIG_DIR.glob('*.json'):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            # Normalize title to upper-case algorithm key
            title = str(data.get('title') or p.stem).upper()
            schemas[title] = data
        except Exception:
            continue
    # Second pass: handle inheritance via optional 'base' key
    for name, data in list(schemas.items()):
        base = str(data.get('base') or data.get('BASE') or '').upper()
        if base and base != name and base in schemas:
            base_schema = schemas[base]
            # Merge: base props first, then override with derived
            merged_props = dict(base_schema.get('properties', {}))
            merged_props.update(data.get('properties', {}))
            data['properties'] = merged_props
            # Merge required lists
            base_req = list(base_schema.get('required', []))
            derived_req = list(data.get('required', []))
            data['required'] = list(dict.fromkeys(base_req + derived_req))  # preserve order unique
            # Inherit selected top-level fields from base if not explicitly defined in derived
            for top_key in ('type', 'multi_column', 'ui:order', 'description'):
                if top_key not in data and top_key in base_schema:
                    data[top_key] = base_schema[top_key]
            schemas[name] = data
    return schemas


def get_algorithm_schema(algo: str) -> Dict[str, Any] | None:
    return load_all_algorithm_schemas().get(algo.upper())


def coerce_value(schema_prop: Dict[str, Any], raw: str):
    t = schema_prop.get('type')
    if raw is None:
        return None
    if t == 'integer':
        return int(raw)
    if t == 'number':
        return float(raw)
    if t == 'boolean':
        return raw.lower() in ('true','1','yes','on')
    if t == 'array':
        # Accept comma separated list
        items_type = (schema_prop.get('items') or {}).get('type')
        parts = [p.strip() for p in str(raw).split(',') if p.strip()]
        if items_type == 'integer':
            return [int(p) for p in parts]
        if items_type == 'number':
            return [float(p) for p in parts]
        if items_type == 'boolean':
            return [p.lower() in ('true','1','yes','on') for p in parts]
        return parts
    return raw  # default string


def validate_parameters(algo: str, form_data: Dict[str, str]) -> Dict[str, Any]:
    schema = get_algorithm_schema(algo)
    if not schema:
        return {}
    props: Dict[str, Any] = schema.get('properties', {})
    required = set(schema.get('required', []))
    out: Dict[str, Any] = {}
    for key, prop in props.items():
        if key in form_data and form_data[key] != '':
            try:
                out[key] = coerce_value(prop, form_data[key])
            except Exception as e:
                raise ValueError(f'Invalid value for {key}: {e}')
        elif key in required:
            raise ValueError(f'Missing required field: {key}')

    # Post validation constraints
    for key, prop in props.items():
        if key not in out:
            continue
        val = out[key]
        if prop.get('type') == 'integer':
            if 'minimum' in prop and val < prop['minimum']:
                raise ValueError(f'{key} must be >= {prop["minimum"]}')
        if prop.get('type') == 'array':
            if prop.get('minItems') and len(val) < prop['minItems']:
                raise ValueError(f'{key} must have at least {prop["minItems"]} items')
            # If paired array (like indexes vs booleans) size alignment can be checked externally
    return out
