#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "conf" / "tw_etf_management_fees.json"
TWSE_ETF_EXCEL_URL = "https://www.twse.com.tw/zh/ETFortune/etfExcel"


def _normalize_code(value: object) -> str:
    token = str(value or "").strip().upper()
    if token.startswith('="') and token.endswith('"'):
        token = token[2:-1]
    m = re.search(r"(\d{4,6}[A-Z]?)", token)
    return m.group(1) if m else ""


def _parse_float(value: object) -> float | None:
    token = str(value or "").strip().replace(",", "")
    if token in {"", "-", "--"}:
        return None
    try:
        out = float(token)
    except Exception:
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out


def _normalize_fee_label(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = token.replace("％", "%").replace("﹪", "%")
    has_from = "起" in token
    m = re.search(r"(\d+(?:\.\d+)?)", token)
    if not m:
        return ""
    try:
        base = float(m.group(1))
    except Exception:
        return ""
    out = f"{base:.2f}%"
    if has_from:
        out += "起"
    return out


def _load_tw_etf_universe() -> dict[str, dict[str, object]]:
    req = Request(
        TWSE_ETF_EXCEL_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=30) as resp:  # nosec B310 (fixed, trusted URL)
        payload = resp.read()
    text = payload.decode("cp950", errors="ignore")
    rows = csv.reader(StringIO(text))
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        code = _normalize_code(row[0] if len(row) > 0 else "")
        if not code:
            continue
        name = str(row[1] if len(row) > 1 else "").strip()
        aum_billion = _parse_float(row[4] if len(row) > 4 else "")
        issuer = str(row[9] if len(row) > 9 else "").strip()
        out[code] = {
            "name": name,
            "aum_billion": aum_billion,
            "issuer": issuer,
        }
    return out


def _load_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"source": "TWSE ETF公開資訊 / 基金公開說明書", "fees": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source": "TWSE ETF公開資訊 / 基金公開說明書", "fees": {}}
    if not isinstance(data, dict):
        return {"source": "TWSE ETF公開資訊 / 基金公開說明書", "fees": {}}
    fees = data.get("fees")
    if not isinstance(fees, dict):
        data["fees"] = {}
    return data


def _normalize_fee_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        code = _normalize_code(key)
        label = _normalize_fee_label(value)
        if code and label:
            out[code] = label
    return out


def _parse_set_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        token = str(raw or "").strip()
        if "=" not in token:
            raise ValueError(f"Invalid --set value (need CODE=FEE): {raw}")
        code_raw, fee_raw = token.split("=", 1)
        code = _normalize_code(code_raw)
        fee = _normalize_fee_label(fee_raw)
        if not code:
            raise ValueError(f"Invalid ETF code in --set: {raw}")
        if not fee:
            raise ValueError(f"Invalid management fee in --set: {raw}")
        out[code] = fee
    return out


def _load_updates_from_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        code_col = next(
            (
                col
                for col in reader.fieldnames
                if str(col).strip() in {"code", "symbol", "代碼", "ETF代碼", "基金代號"}
            ),
            "",
        )
        fee_col = next(
            (
                col
                for col in reader.fieldnames
                if str(col).strip() in {"fee", "management_fee", "管理費", "經理費"}
            ),
            "",
        )
        if not code_col or not fee_col:
            raise ValueError("CSV requires code/symbol + fee columns (or 代碼/ETF代碼 + 管理費)")
        for row in reader:
            code = _normalize_code(row.get(code_col, ""))
            fee = _normalize_fee_label(row.get(fee_col, ""))
            if not code or not fee:
                continue
            out[code] = fee
    return out


def _build_missing_rows(
    universe: dict[str, dict[str, object]],
    fees: dict[str, str],
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for code, meta in universe.items():
        if code in fees:
            continue
        missing.append(
            {
                "code": code,
                "name": str(meta.get("name", "")),
                "aum_billion": meta.get("aum_billion"),
                "issuer": str(meta.get("issuer", "")),
            }
        )
    missing.sort(
        key=lambda row: (
            0 if isinstance(row.get("aum_billion"), (int, float)) else 1,
            -float(row.get("aum_billion") or 0.0),
            str(row.get("code", "")),
        )
    )
    return missing


def _fmt_aum(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    v = float(value)
    if not math.isfinite(v) or v < 0:
        return "—"
    text = f"{v:,.2f}"
    if text.endswith(".00"):
        return text[:-3]
    return text.rstrip("0").rstrip(".")


def _with_meta(
    payload: dict[str, object],
    *,
    fees: dict[str, str],
    total_codes: int,
) -> dict[str, object]:
    out = dict(payload)
    out["fees"] = dict(sorted(fees.items()))
    out["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out["source"] = str(out.get("source") or "TWSE ETF公開資訊 / 基金公開說明書")
    covered = len(out["fees"])
    out["covered_codes"] = covered
    out["total_codes"] = int(total_codes)
    out["coverage_pct"] = round((covered / total_codes * 100.0), 2) if total_codes > 0 else 0.0
    return out


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_missing_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "name", "aum_billion", "issuer"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "code": row.get("code", ""),
                    "name": row.get("name", ""),
                    "aum_billion": _fmt_aum(row.get("aum_billion")),
                    "issuer": row.get("issuer", ""),
                }
            )


def _print_summary(
    *,
    config_path: Path,
    covered_codes: int,
    total_codes: int,
    missing_rows: list[dict[str, object]],
    top: int,
) -> None:
    pct = (covered_codes / total_codes * 100.0) if total_codes > 0 else 0.0
    print(f"[info] config      : {config_path}")
    print(f"[info] coverage    : {covered_codes}/{total_codes} ({pct:.2f}%)")
    print(f"[info] missing     : {len(missing_rows)}")
    if not missing_rows:
        return
    show_n = max(0, int(top))
    if show_n <= 0:
        return
    print(f"[info] top missing by AUM (show {min(show_n, len(missing_rows))}):")
    for i, row in enumerate(missing_rows[:show_n], start=1):
        code = str(row.get("code", ""))
        name = str(row.get("name", ""))
        aum = _fmt_aum(row.get("aum_billion"))
        print(f"  {i:>2}. {code} {name} | AUM(億): {aum}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage TW ETF management fee whitelist (incremental updates + missing report)."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to tw_etf_management_fees.json")
    parser.add_argument("--set", action="append", default=[], metavar="CODE=FEE", help="Set one fee entry.")
    parser.add_argument("--input-csv", default="", help="Bulk update from CSV.")
    parser.add_argument("--missing-csv", default="", help="Write missing ETF fee list to CSV.")
    parser.add_argument("--top", type=int, default=30, help="How many missing rows to print.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write config, print report only.")
    parser.add_argument(
        "--refresh-meta",
        action="store_true",
        help="Rewrite metadata even if no new --set/--input-csv changes.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    payload = _load_payload(config_path)
    fee_map = _normalize_fee_map(payload.get("fees"))

    try:
        universe = _load_tw_etf_universe()
    except Exception as exc:
        print(f"[warn] unable to load TWSE ETF universe: {exc}", file=sys.stderr)
        universe = {}

    updates: dict[str, str] = {}
    try:
        updates.update(_parse_set_items(list(args.set or [])))
        if str(args.input_csv or "").strip():
            updates.update(_load_updates_from_csv(Path(str(args.input_csv)).expanduser()))
    except Exception as exc:
        print(f"[error] invalid update input: {exc}", file=sys.stderr)
        return 1

    if updates:
        fee_map.update(updates)

    total_codes = len(universe) if universe else int(payload.get("total_codes") or 0)
    next_payload = _with_meta(payload, fees=fee_map, total_codes=total_codes)
    missing_rows = _build_missing_rows(universe, fee_map) if universe else []

    should_write = (bool(updates) or bool(args.refresh_meta)) and (not bool(args.dry_run))
    if should_write:
        _write_payload(config_path, next_payload)
        print(f"[ok] updated config: {config_path}")
        if updates:
            print(f"[ok] merged entries: {len(updates)}")

    if str(args.missing_csv or "").strip():
        missing_csv = Path(str(args.missing_csv)).expanduser()
        _write_missing_csv(missing_csv, missing_rows)
        print(f"[ok] wrote missing CSV: {missing_csv}")

    _print_summary(
        config_path=config_path,
        covered_codes=int(next_payload.get("covered_codes") or 0),
        total_codes=int(next_payload.get("total_codes") or 0),
        missing_rows=missing_rows,
        top=int(args.top or 0),
    )
    if not should_write and updates and bool(args.dry_run):
        print("[info] dry-run mode: no files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
