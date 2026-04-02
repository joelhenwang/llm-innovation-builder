from __future__ import annotations

from auto_llm_innovator.constants import GPT2_TOKENIZER, PARAMETER_CAP, PHASES
from auto_llm_innovator.design_ir.models import DesignIR


class DesignIRValidationError(ValueError):
    pass


def validate_design_ir(design_ir: DesignIR) -> None:
    if not design_ir.title.strip():
        raise DesignIRValidationError("DesignIR is missing a title.")
    if design_ir.parameter_cap <= 0 or design_ir.parameter_cap > PARAMETER_CAP:
        raise DesignIRValidationError(f"DesignIR parameter cap must be between 1 and {PARAMETER_CAP:,}.")
    if design_ir.tokenizer_requirement.strip().lower() != GPT2_TOKENIZER:
        raise DesignIRValidationError(f"DesignIR tokenizer requirement must be '{GPT2_TOKENIZER}'.")
    if not design_ir.modules:
        raise DesignIRValidationError("DesignIR must declare modules.")

    module_names = set()
    has_core = False
    for module in design_ir.modules:
        name = module.name.strip()
        kind = module.kind.strip()
        if not name or "unknown" in name.lower():
            raise DesignIRValidationError("DesignIR modules must have concrete names.")
        if not kind or "unknown" in kind.lower():
            raise DesignIRValidationError("DesignIR modules must have concrete kinds.")
        module_names.add(name)
        if "core" in kind:
            has_core = True

    if not has_core:
        raise DesignIRValidationError("DesignIR must include at least one nontrivial core module.")

    for module in design_ir.modules:
        for dependency in module.depends_on:
            if dependency not in module_names:
                raise DesignIRValidationError(f"Module '{module.name}' depends on undefined module '{dependency}'.")

    for tensor in design_ir.tensor_interfaces:
        if tensor.producer not in module_names:
            raise DesignIRValidationError(
                f"Tensor interface '{tensor.name}' references undefined producer '{tensor.producer}'."
            )
        if tensor.consumer not in module_names:
            raise DesignIRValidationError(
                f"Tensor interface '{tensor.name}' references undefined consumer '{tensor.consumer}'."
            )

    stages = {stage.stage for stage in design_ir.training_plan}
    missing_stages = [phase for phase in PHASES if phase not in stages]
    if missing_stages:
        raise DesignIRValidationError(f"DesignIR training plan is missing stages: {', '.join(missing_stages)}.")

    if not design_ir.evaluation_plan:
        raise DesignIRValidationError("DesignIR must include at least one evaluation task.")

    for ablation in design_ir.ablation_plan:
        if not ablation.name.strip():
            raise DesignIRValidationError("Ablation plans must have names.")
        for module_name in ablation.target_modules:
            if module_name not in module_names:
                raise DesignIRValidationError(
                    f"Ablation '{ablation.name}' references undefined module '{module_name}'."
                )

    for criterion in design_ir.failure_criteria:
        if not criterion.name.strip():
            raise DesignIRValidationError("Failure criteria must have names.")
        if not criterion.focus_area.strip():
            raise DesignIRValidationError(f"Failure criterion '{criterion.name}' must declare a focus area.")
        for module_name in criterion.target_modules:
            if module_name not in module_names:
                raise DesignIRValidationError(
                    f"Failure criterion '{criterion.name}' references undefined module '{module_name}'."
                )
