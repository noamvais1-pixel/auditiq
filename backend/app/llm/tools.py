"""Tools the model may call instead of doing arithmetic in its head.

Two separate failure modes are being closed here.

*Arithmetic.* Language models are unreliable calculators, and in this domain a
wrong number is worse than no answer -- it looks authoritative and it lands in a
working paper. Every computation is therefore executed in Python.

*Rates.* A model asked "what is the VAT rate" answers from training data, which
is a snapshot of some past year. Rates are read from ``data/rates/rates.json``
and are effective-dated, so a question about a 2024 invoice resolves to 17% and
one about 2025 resolves to 18% -- without the model needing to know either.

Every tool returns its inputs alongside its result. That makes the answer
auditable: the reviewer can see the rate that was applied and where it came from,
not just the bottom line.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

RATES_PATH = Path(__file__).resolve().parents[2] / "data" / "rates" / "rates.json"


def _load_rates() -> dict[str, Any]:
    return json.loads(RATES_PATH.read_text(encoding="utf-8"))


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def _rate_on(series: list[dict[str, Any]], when: date) -> float:
    for entry in series:
        start = _parse_date(entry["from"])
        end = _parse_date(entry["to"]) if entry["to"] else date.max
        if start <= when <= end:
            return float(entry["rate"])
    raise ValueError(f"no rate defined for {when.isoformat()}")


# --------------------------------------------------------------------- tools
def calc_vat(amount: float, direction: str = "add", on_date: str | None = None) -> dict:
    """direction='add' treats amount as pre-VAT; 'extract' treats it as VAT-inclusive."""
    when = _parse_date(on_date)
    rate = _rate_on(_load_rates()["vat"], when)
    if direction == "extract":
        net = amount / (1 + rate)
        vat = amount - net
    else:
        net = amount
        vat = amount * rate
    return {
        "rate": rate,
        "rate_effective_on": when.isoformat(),
        "net": round(net, 2),
        "vat": round(vat, 2),
        "gross": round(net + vat, 2),
        "direction": direction,
    }


def calc_depreciation(
    cost: float,
    method: str = "straight_line",
    useful_life_years: float | None = None,
    rate: float | None = None,
    salvage: float = 0.0,
    years: int = 5,
) -> dict:
    """Builds the full schedule, not just year one -- that is what goes in the file."""
    schedule: list[dict[str, float]] = []
    book = cost
    accumulated = 0.0

    if method == "straight_line":
        if not useful_life_years:
            raise ValueError("straight_line requires useful_life_years")
        annual = (cost - salvage) / useful_life_years
        for year in range(1, min(years, int(useful_life_years)) + 1):
            charge = min(annual, book - salvage)
            accumulated += charge
            book -= charge
            schedule.append(_row(year, charge, accumulated, book))

    elif method == "declining_balance":
        if rate is None:
            raise ValueError("declining_balance requires rate")
        for year in range(1, years + 1):
            charge = max(min(book * rate, book - salvage), 0.0)
            accumulated += charge
            book -= charge
            schedule.append(_row(year, charge, accumulated, book))

    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "method": method,
        "cost": cost,
        "salvage": salvage,
        "schedule": schedule,
        "closing_book_value": round(book, 2),
    }


def _row(year: int, charge: float, accumulated: float, book: float) -> dict[str, float]:
    return {
        "year": year,
        "charge": round(charge, 2),
        "accumulated": round(accumulated, 2),
        "book_value": round(book, 2),
    }


def calc_materiality(benchmark_value: float, benchmark: str, pct: float, performance_pct: float = 0.6) -> dict:
    """Overall + performance materiality, with the judgement left explicit."""
    overall = benchmark_value * pct
    return {
        "benchmark": benchmark,
        "benchmark_value": benchmark_value,
        "pct_applied": pct,
        "overall_materiality": round(overall, 2),
        "performance_pct": performance_pct,
        "performance_materiality": round(overall * performance_pct, 2),
        "note": "אחוז הביצוע הוא שיקול דעת: ככל שהסיכון המוערך גבוה יותר, יש לבחור בקצה הנמוך של הטווח.",
    }


def present_value(payment: float, periods: int, rate: float, timing: str = "end") -> dict:
    """PV of a level annuity -- the lease liability calculation."""
    if rate == 0:
        pv = payment * periods
    else:
        pv = payment * (1 - (1 + rate) ** -periods) / rate
        if timing == "begin":
            pv *= 1 + rate
    return {
        "payment": payment,
        "periods": periods,
        "rate": rate,
        "timing": timing,
        "present_value": round(pv, 2),
    }


REGISTRY = {
    "calc_vat": calc_vat,
    "calc_depreciation": calc_depreciation,
    "calc_materiality": calc_materiality,
    "present_value": present_value,
}

# Anthropic tool-use schemas.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "calc_vat",
        "description": (
            "מחשב מע\"מ לפי השיעור שהיה בתוקף במועד שנמסר. "
            "יש להשתמש בכלי זה בכל חישוב מע\"מ ולא לחשב ידנית, "
            "וכן כדי לברר את שיעור המע\"מ במועד מסוים."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "הסכום"},
                "direction": {
                    "type": "string",
                    "enum": ["add", "extract"],
                    "description": "add — הסכום לפני מע\"מ; extract — הסכום כולל מע\"מ",
                },
                "on_date": {
                    "type": "string",
                    "description": "מועד החיוב במס בפורמט YYYY-MM-DD. ברירת המחדל היא היום.",
                },
            },
            "required": ["amount"],
        },
    },
    {
        "name": "calc_depreciation",
        "description": "בונה לוח פחת מלא בשיטת הקו הישר או היתרה הפוחתת.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cost": {"type": "number"},
                "method": {"type": "string", "enum": ["straight_line", "declining_balance"]},
                "useful_life_years": {"type": "number"},
                "rate": {"type": "number", "description": "שיעור שנתי כשבר עשרוני, ליתרה פוחתת"},
                "salvage": {"type": "number"},
                "years": {"type": "integer", "description": "מספר שנים להצגה"},
            },
            "required": ["cost", "method"],
        },
    },
    {
        "name": "calc_materiality",
        "description": "מחשב מהותיות לדוחות בכללותם ומהותיות ביצוע לפי מדד ואחוז נבחרים.",
        "input_schema": {
            "type": "object",
            "properties": {
                "benchmark_value": {"type": "number"},
                "benchmark": {"type": "string", "description": "לדוגמה: רווח לפני מס"},
                "pct": {"type": "number", "description": "שבר עשרוני, לדוגמה 0.05"},
                "performance_pct": {"type": "number"},
            },
            "required": ["benchmark_value", "benchmark", "pct"],
        },
    },
    {
        "name": "present_value",
        "description": "ערך נוכחי של סדרת תשלומים שווים — לחישוב התחייבות בגין חכירה.",
        "input_schema": {
            "type": "object",
            "properties": {
                "payment": {"type": "number"},
                "periods": {"type": "integer"},
                "rate": {"type": "number", "description": "ריבית לתקופה כשבר עשרוני"},
                "timing": {"type": "string", "enum": ["end", "begin"]},
            },
            "required": ["payment", "periods", "rate"],
        },
    },
]


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(**arguments)
    except Exception as exc:
        # Surfaced back to the model so it can correct its arguments and retry,
        # rather than the request dying on a bad float.
        return {"error": f"{type(exc).__name__}: {exc}"}
