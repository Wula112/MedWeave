"""Business tool identity/permissions, independent of middleware execution."""
from __future__ import annotations

import json

from langchain.agents.middleware.types import ToolCallRequest
from pydantic import ValidationError

# Future tools that write external state do not opt in automatically.
READ_ONLY_TOOLS = frozenset({
    "get_insurance_balance", "get_consumption_records", "get_payment_records",
    "get_cross_region_info", "search_insurance_policy", "search_drug_info",
    "check_drug_interaction", "find_nearby_pharmacy", "get_otc_recommendation",
    "health_calculator", "risk_assessor", "symptom_scorer", "lab_interpreter",
    "emergency_triage", "medication_calculator",
})


def tool_identity(request: ToolCallRequest) -> tuple[str, dict, bool]:
    name = request.tool_call["name"]
    arguments = request.tool_call.get("args", {})
    schema = getattr(request.tool, "args_schema", None)
    if name == "load_skill":
        try:
            from skills import get_registry
            name = arguments["skill_name"]
            arguments = arguments.get("params_json", {})
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            schema = get_registry()._load_instance(name).input_schema
        except (KeyError, ValueError, TypeError, ImportError, AttributeError):
            return request.tool_call["name"], request.tool_call.get("args", {}), False
    if schema and isinstance(arguments, dict):
        try:
            arguments = schema.model_validate(arguments).model_dump(mode="json")
        except ValidationError:
            pass  # Framework validation is sanitized by the error middleware.
    metadata = getattr(request.tool, "metadata", None) or {}
    return name, arguments, name in READ_ONLY_TOOLS or metadata.get("runtime_readonly") is True
