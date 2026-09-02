from __future__ import annotations

from collections.abc import Mapping

from .constants import FACTOR_ORDER

RULE_FACTORS: dict[str, tuple[str, ...]] = {
    "X": ("X",),
    "C": ("C",),
    "S": ("S",),
    "C_AND_S": ("C", "S"),
    "C_AND_X": ("C", "X"),
    "S_AND_X": ("S", "X"),
    "C_AND_S_AND_X": ("C", "S", "X"),
}


def factors_from_cell(cell_id: str) -> dict[str, bool]:
    if len(cell_id) != len(FACTOR_ORDER) or set(cell_id) - {"0", "1"}:
        raise ValueError(f"Invalid trigger-matrix cell: {cell_id!r}")
    return {
        factor: bit == "1" for factor, bit in zip(FACTOR_ORDER, cell_id)
    }


def cell_from_factors(factors: Mapping[str, object]) -> str:
    missing = [factor for factor in FACTOR_ORDER if factor not in factors]
    if missing:
        raise ValueError(f"Missing trigger-matrix factors: {missing}")
    return "".join("1" if bool(factors[factor]) else "0" for factor in FACTOR_ORDER)


def expected_trigger(rule: str, factors: Mapping[str, object] | str) -> bool:
    if rule not in RULE_FACTORS:
        raise ValueError(f"Unsupported trigger-matrix rule: {rule}")
    values = factors_from_cell(factors) if isinstance(factors, str) else factors
    return all(bool(values[factor]) for factor in RULE_FACTORS[rule])


def all_cells() -> tuple[str, ...]:
    return tuple(f"{value:03b}" for value in range(8))


def rule_vector(rule: str) -> tuple[int, ...]:
    return tuple(int(expected_trigger(rule, cell)) for cell in all_cells())

