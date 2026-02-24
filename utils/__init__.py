import pandas as pd


BASE_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv_frame(df: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns if columns is not None else BASE_OHLCV_COLUMNS

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        renamed: list[str] = []
        for col in out.columns:
            parts = [str(part).strip().lower() for part in col if str(part).strip()]
            candidate = ""
            for item in reversed(parts):
                if item in {
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "adj_close",
                    "volume",
                    "price",
                }:
                    candidate = item
                    break
            renamed.append(candidate or (parts[-1] if parts else ""))
        out.columns = renamed
    else:
        out.columns = [str(col).strip().lower() for col in out.columns]

    if "adj close" in out.columns and "adj_close" not in out.columns:
        out = out.rename(columns={"adj close": "adj_close"})

    if "price" in out.columns and "close" not in out.columns:
        out["close"] = out["price"]

    if "close" not in out.columns:
        return pd.DataFrame(columns=cols)

    close = pd.to_numeric(out["close"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(close)
        else:
            out[col] = close

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0

    out["close"] = close
    out = out[cols].dropna(subset=["open", "high", "low", "close"], how="any")
    return out.sort_index()
