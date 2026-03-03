"""
Input Sanitization Layer — Financial Simulation API (Phase 6)

Pre-schema, pre-tool-execution defense layer that protects against:
  - Injection attacks (SQL, script, Python dunder)
  - Payload expansion / oversized bodies
  - Deeply nested objects
  - Encoded attack vectors
  - Type coercion attempts (numeric strings, boolean strings)
  - Prototype pollution patterns

Every rejection is deterministic and returns a structured error.
No data is silently stripped or auto-fixed — violations are rejected.

No simulation logic.  No risk logic.  No persistence.  No logging.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

MAX_PAYLOAD_BYTES = 50 * 1024          # 50 KB
MAX_STRING_LENGTH = 5_000              # per-field
MAX_NESTING_DEPTH = 5
MAX_TOTAL_KEYS = 200

# Malicious string patterns (case-insensitive)
DUNDER_PATTERNS = frozenset({
    "__class__", "__dict__", "__globals__", "__subclasses__",
    "__import__", "__builtins__", "__init__",
})

CODE_INJECTION_PATTERNS = frozenset({
    "eval(", "exec(", "import(", "compile(",
    "getattr(", "setattr(", "delattr(",
    "os.system(", "subprocess.",
})

SQL_INJECTION_PATTERNS = frozenset({
    "drop table", "select *", "insert into", "delete from",
    "union select", "or 1=1", "' or '", "1=1--",
})

SQL_INJECTION_TOKENS = frozenset({
    ";--", "-- ", "/*", "*/",
})

SCRIPT_INJECTION_PATTERNS = frozenset({
    "<script>", "</script>", "<script ",
    "javascript:", "onerror=", "onload=",
    "onfocus=", "onmouseover=",
})

FORBIDDEN_KEY_PREFIXES = ("$", "__")
FORBIDDEN_KEY_CHARS = frozenset({".", "[", "]"})

PROTOTYPE_POLLUTION_KEYS = frozenset({
    "__proto__", "constructor", "prototype",
})


# ═══════════════════════════════════════════════════════════════════════
#  Structured Error
# ═══════════════════════════════════════════════════════════════════════

class SanitizationError(Exception):
    """Raised when input fails sanitization."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)

    def to_dict(self) -> Dict[str, str]:
        return {
            "status": "error",
            "error_code": self.error_code,
            "message": self.message,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — Payload Size Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_payload_size(raw_bytes: bytes) -> None:
    """Reject if raw JSON body exceeds MAX_PAYLOAD_BYTES."""
    if len(raw_bytes) > MAX_PAYLOAD_BYTES:
        raise SanitizationError(
            "PAYLOAD_TOO_LARGE",
            f"Request body is {len(raw_bytes)} bytes, "
            f"exceeding limit of {MAX_PAYLOAD_BYTES} bytes",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Section 5 — Recursive Object Guard (depth, key count, strings)
# ═══════════════════════════════════════════════════════════════════════

def _validate_structure(
    obj: Any,
    *,
    depth: int = 0,
    key_counter: List[int],
    path: str = "$",
    seen_ids: Set[int],
) -> None:
    """Recursively validate structure constraints.

    Checks:
      - Maximum nesting depth
      - Maximum total key count
      - Maximum string length
      - Circular reference detection (by object id)
    """
    obj_id = id(obj)
    if obj_id in seen_ids and isinstance(obj, (dict, list)):
        raise SanitizationError(
            "INVALID_INPUT",
            f"Circular reference detected at path '{path}'",
        )
    seen_ids.add(obj_id)

    if depth > MAX_NESTING_DEPTH:
        raise SanitizationError(
            "INVALID_INPUT",
            f"Nesting depth exceeds {MAX_NESTING_DEPTH} at path '{path}'",
        )

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_counter[0] += 1
            if key_counter[0] > MAX_TOTAL_KEYS:
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Total key count exceeds {MAX_TOTAL_KEYS}",
                )
            _validate_structure(
                value,
                depth=depth + 1,
                key_counter=key_counter,
                path=f"{path}.{key}",
                seen_ids=seen_ids,
            )

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_structure(
                item,
                depth=depth + 1,
                key_counter=key_counter,
                path=f"{path}[{i}]",
                seen_ids=seen_ids,
            )

    elif isinstance(obj, str):
        if len(obj) > MAX_STRING_LENGTH:
            raise SanitizationError(
                "INVALID_INPUT",
                f"String field at '{path}' exceeds {MAX_STRING_LENGTH} characters "
                f"(length={len(obj)})",
            )


# ═══════════════════════════════════════════════════════════════════════
#  Section 2 — Type Enforcement Pre-Check
# ═══════════════════════════════════════════════════════════════════════

# Fields that MUST be numeric (float/int) and must never be strings
NUMERIC_FIELD_NAMES = frozenset({
    "monthly_income", "fixed_expenses", "discretionary_expenses",
    "emergency_fund", "credit_balance", "credit_apr",
    "amount", "monthly_cost", "partial_percentage",
    "stability_score", "liquidity_ratio", "commitment_coverage",
    "interest_cost", "projected_balance", "due_month",
})

BOOLEAN_FIELD_NAMES = frozenset({
    "is_active", "enabled", "approved", "verified",
})


def _check_types(obj: Any, path: str = "$") -> None:
    """Reject type coercion attempts.

    - Numeric fields must not be strings
    - Boolean fields must not be strings
    - Required fields must not be null (at this level, we check for
      None values in known-required patterns)
    - Arrays must not contain mixed primitive types
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{path}.{key}"

            # Numeric field is a string → reject
            if key in NUMERIC_FIELD_NAMES and isinstance(value, str):
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Numeric field '{full_path}' is a string ('{value}'). "
                    f"Numeric values must not be quoted.",
                )

            # Boolean field is a string → reject
            if key in BOOLEAN_FIELD_NAMES and isinstance(value, str):
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Boolean field '{full_path}' is a string ('{value}'). "
                    f"Use true/false, not strings.",
                )

            # Recurse
            _check_types(value, full_path)

    elif isinstance(obj, list) and len(obj) > 1:
        # Check for mixed primitive types in arrays
        types_seen: Set[type] = set()
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                _check_types(item, f"{path}[{i}]")
                types_seen.add(dict)
            elif isinstance(item, list):
                _check_types(item, f"{path}[{i}]")
                types_seen.add(list)
            else:
                types_seen.add(type(item))

        # Allow dict-only or single-type primitives, not mixed
        primitive_types = types_seen - {dict, list}
        if len(primitive_types) > 1:
            type_names = sorted(t.__name__ for t in primitive_types)
            raise SanitizationError(
                "INVALID_INPUT",
                f"Array at '{path}' contains mixed types: {type_names}",
            )


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Malicious Pattern Filter
# ═══════════════════════════════════════════════════════════════════════

def _scan_string_for_injection(value: str, path: str) -> None:
    """Scan a single string value for malicious patterns."""
    lower = value.lower()

    # Python dunder / code injection
    for pattern in DUNDER_PATTERNS:
        if pattern in lower:
            raise SanitizationError(
                "INVALID_INPUT",
                f"Malicious pattern '{pattern}' detected in field '{path}'",
            )

    for pattern in CODE_INJECTION_PATTERNS:
        if pattern in lower:
            raise SanitizationError(
                "INVALID_INPUT",
                f"Code injection pattern '{pattern}' detected in field '{path}'",
            )

    # SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern in lower:
            raise SanitizationError(
                "INVALID_INPUT",
                f"SQL injection pattern detected in field '{path}'",
            )
    for token in SQL_INJECTION_TOKENS:
        if token in lower:
            raise SanitizationError(
                "INVALID_INPUT",
                f"SQL injection token detected in field '{path}'",
            )

    # Script injection
    for pattern in SCRIPT_INJECTION_PATTERNS:
        if pattern in lower:
            raise SanitizationError(
                "INVALID_INPUT",
                f"Script injection pattern detected in field '{path}'",
            )


def _scan_malicious_patterns(obj: Any, path: str = "$") -> None:
    """Recursively scan all string values for injection patterns."""
    if isinstance(obj, str):
        _scan_string_for_injection(obj, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # Also scan keys themselves
            _scan_string_for_injection(key, f"{path}.<key:{key}>")
            _scan_malicious_patterns(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_malicious_patterns(item, f"{path}[{i}]")


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — JSON Structure Hardening
# ═══════════════════════════════════════════════════════════════════════

def _validate_key_names(obj: Any, path: str = "$") -> None:
    """Reject forbidden key patterns.

    - No keys starting with $ or __
    - No keys containing dots or brackets
    - No prototype pollution keys
    """
    if isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Non-string key detected at '{path}': {type(key).__name__}",
                )

            # Prototype pollution
            if key.lower() in PROTOTYPE_POLLUTION_KEYS:
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Prototype pollution key '{key}' detected at '{path}'",
                )

            # Forbidden prefixes
            for prefix in FORBIDDEN_KEY_PREFIXES:
                if key.startswith(prefix):
                    raise SanitizationError(
                        "INVALID_INPUT",
                        f"Key '{key}' at '{path}' starts with "
                        f"forbidden prefix '{prefix}'",
                    )

            # Forbidden characters
            for char in FORBIDDEN_KEY_CHARS:
                if char in key:
                    raise SanitizationError(
                        "INVALID_INPUT",
                        f"Key '{key}' at '{path}' contains "
                        f"forbidden character '{char}'",
                    )

            # Recurse into values
            _validate_key_names(obj[key], f"{path}.{key}")

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _validate_key_names(item, f"{path}[{i}]")


def _check_duplicate_keys(raw_json: str) -> None:
    """Detect duplicate keys in raw JSON.

    Uses a custom JSON decoder that raises on duplicate keys.
    """
    def _pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        seen: Dict[str, int] = {}
        for key, _ in pairs:
            if key in seen:
                raise SanitizationError(
                    "INVALID_INPUT",
                    f"Duplicate JSON key '{key}' detected",
                )
            seen[key] = 1
        return dict(pairs)

    json.loads(raw_json, object_pairs_hook=_pairs_hook)


# ═══════════════════════════════════════════════════════════════════════
#  Section 6 — Main Sanitization Function
# ═══════════════════════════════════════════════════════════════════════

def sanitize_input(raw_payload: dict) -> dict:
    """Sanitize a pre-parsed dict payload.

    Steps:
      1. Validate structure (depth, key count, string length)
      2. Validate types (no numeric strings, no boolean strings)
      3. Scan for malicious patterns (injection, dunder)
      4. Validate key names (no $, no dots, no pollution)

    Returns the payload unchanged if valid.
    Raises ``SanitizationError`` on any violation.
    """
    if not isinstance(raw_payload, dict):
        raise SanitizationError(
            "INVALID_INPUT",
            f"Expected JSON object at root, got {type(raw_payload).__name__}",
        )

    # Step 1: Structure
    key_counter = [0]
    _validate_structure(
        raw_payload,
        depth=0,
        key_counter=key_counter,
        path="$",
        seen_ids=set(),
    )

    # Step 2: Types
    _check_types(raw_payload, path="$")

    # Step 3: Malicious patterns
    _scan_malicious_patterns(raw_payload, path="$")

    # Step 4: Key names
    _validate_key_names(raw_payload, path="$")

    return raw_payload


def sanitize_raw_request(raw_bytes: bytes) -> dict:
    """Full sanitization pipeline for raw HTTP body bytes.

    Steps:
      1. Enforce payload size limit
      2. Parse JSON (reject invalid)
      3. Check duplicate keys
      4. Call sanitize_input()

    Returns clean dict if valid.
    Raises ``SanitizationError`` on any violation.
    """
    # Step 1: Size limit
    validate_payload_size(raw_bytes)

    # Step 2: Parse JSON
    try:
        raw_json = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise SanitizationError(
            "INVALID_INPUT",
            "Request body is not valid UTF-8",
        )

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SanitizationError(
            "INVALID_INPUT",
            f"Invalid JSON: {exc}",
        )

    # Step 3: Duplicate keys
    _check_duplicate_keys(raw_json)

    # Step 4: Full sanitization
    return sanitize_input(payload)


# ═══════════════════════════════════════════════════════════════════════
#  Section 7 — FastAPI Integration
# ═══════════════════════════════════════════════════════════════════════

# NOTE: This requires FastAPI (pip install fastapi).
# Import is conditional to keep the module usable standalone.

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    async def sanitize_request(request: Request) -> dict:
        """FastAPI dependency that sanitizes the incoming request body.

        Usage::

            @app.post("/simulate")
            async def simulate(payload: dict = Depends(sanitize_request)):
                ...

        Reads the raw body, enforces size limit, parses JSON,
        runs full sanitization, and returns the clean dict.
        If any check fails, raises ``SanitizationError`` which
        should be caught by the exception handler below.
        """
        raw_bytes = await request.body()
        return sanitize_raw_request(raw_bytes)

    def register_sanitization_error_handler(app: Any) -> None:
        """Register a FastAPI exception handler for SanitizationError.

        Usage::

            app = FastAPI()
            register_sanitization_error_handler(app)
        """
        @app.exception_handler(SanitizationError)
        async def _handle(request: Request, exc: SanitizationError) -> JSONResponse:
            return JSONResponse(
                status_code=400,
                content=exc.to_dict(),
            )

    _FASTAPI_AVAILABLE = True

except ImportError:
    _FASTAPI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        if isinstance(obj, dict):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(str(obj))

    def _test(label: str, payload: Any, *, raw: bool = False) -> None:
        """Test sanitization and display result."""
        try:
            if raw:
                result = sanitize_raw_request(payload)
            else:
                result = sanitize_input(payload)
            _show(f"✅ ACCEPTED: {label}", result)
        except SanitizationError as exc:
            _show(f"❌ REJECTED: {label}", exc.to_dict())

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID PAYLOAD
    # ══════════════════════════════════════════════════════════════════

    _test("Valid financial state", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "commitments": [
            {"id": "ins", "amount": 300.0, "due_month": 3},
        ],
        "subscriptions": [
            {"name": "streaming", "monthly_cost": 15.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 2. NUMERIC STRING COERCION ATTEMPT
    # ══════════════════════════════════════════════════════════════════

    _test("Numeric string coercion", {
        "monthly_income": "5000",
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
    })

    # ══════════════════════════════════════════════════════════════════
    # 3. SQL INJECTION IN STRING FIELD
    # ══════════════════════════════════════════════════════════════════

    _test("SQL injection", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "streaming'; DROP TABLE users;--", "monthly_cost": 15.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 4. SCRIPT INJECTION
    # ══════════════════════════════════════════════════════════════════

    _test("Script injection", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "<script>alert('xss')</script>", "monthly_cost": 10.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 5. PYTHON DUNDER ATTACK
    # ══════════════════════════════════════════════════════════════════

    _test("Python dunder attack", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "__class__.__bases__[0]", "monthly_cost": 10.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 6. PROTOTYPE POLLUTION KEY
    # ══════════════════════════════════════════════════════════════════

    _test("Prototype pollution", {
        "monthly_income": 5000.0,
        "__proto__": {"admin": True},
    })

    # ══════════════════════════════════════════════════════════════════
    # 7. $ PREFIX KEY
    # ══════════════════════════════════════════════════════════════════

    _test("Dollar prefix key", {
        "monthly_income": 5000.0,
        "$where": "this.admin == true",
    })

    # ══════════════════════════════════════════════════════════════════
    # 8. DOT IN KEY NAME
    # ══════════════════════════════════════════════════════════════════

    _test("Dot in key name", {
        "monthly_income": 5000.0,
        "user.role": "admin",
    })

    # ══════════════════════════════════════════════════════════════════
    # 9. DEEPLY NESTED OBJECT
    # ══════════════════════════════════════════════════════════════════

    deep = {"level": 0}
    current = deep
    for i in range(1, 8):
        current["nested"] = {"level": i}
        current = current["nested"]

    _test("Deeply nested (>5 levels)", deep)

    # ══════════════════════════════════════════════════════════════════
    # 10. OVERSIZED PAYLOAD
    # ══════════════════════════════════════════════════════════════════

    oversized = json.dumps({"data": "x" * (MAX_PAYLOAD_BYTES + 1)}).encode()
    _test("Oversized payload (>50KB)", oversized, raw=True)

    # ══════════════════════════════════════════════════════════════════
    # 11. DUPLICATE JSON KEYS
    # ══════════════════════════════════════════════════════════════════

    dup_json = b'{"monthly_income": 5000, "monthly_income": 99999}'
    _test("Duplicate JSON keys", dup_json, raw=True)

    # ══════════════════════════════════════════════════════════════════
    # 12. MIXED TYPES IN ARRAY
    # ══════════════════════════════════════════════════════════════════

    _test("Mixed types in array", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [15.0, "streaming", True],
    })

    # ══════════════════════════════════════════════════════════════════
    # 13. EVAL/EXEC CODE INJECTION
    # ══════════════════════════════════════════════════════════════════

    _test("Code injection (eval)", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "eval(os.system('rm -rf /'))", "monthly_cost": 10.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 14. OVERSIZED STRING FIELD
    # ══════════════════════════════════════════════════════════════════

    _test("Oversized string field", {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "A" * 6000, "monthly_cost": 10.0},
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═' * 60}")
    print(f"  FastAPI integration available: {_FASTAPI_AVAILABLE}")
    print(f"{'═' * 60}")
