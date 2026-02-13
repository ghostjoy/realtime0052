from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from market_data_types import OhlcvSnapshot, QuoteSnapshot
from providers.base import MarketDataProvider, ProviderError, ProviderErrorKind, ProviderRequest


class TwFugleWebSocketProvider(MarketDataProvider):
    name = "fugle_ws"
    default_ws_url = "wss://api.fugle.tw/marketdata/v1.0/stock/streaming"
    default_key_file = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "codexapp" / "fuglekey"

    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str] = None) -> Optional[str]:
        direct = str(api_key or "").strip()
        if direct:
            return direct

        env_key = str(os.getenv("FUGLE_MARKETDATA_API_KEY") or os.getenv("FUGLE_API_KEY") or "").strip()
        if env_key:
            return env_key

        key_file = str(
            os.getenv("FUGLE_MARKETDATA_API_KEY_FILE") or os.getenv("FUGLE_API_KEY_FILE") or cls.default_key_file
        ).strip()
        if not key_file:
            return None
        try:
            text = Path(key_file).expanduser().read_text(encoding="utf-8")
        except Exception:
            return None
        normalized = text.strip()
        return normalized or None

    def __init__(self, api_key: Optional[str] = None, timeout_sec: int = 8, ws_url: Optional[str] = None):
        self.api_key = self._resolve_api_key(api_key)
        self.timeout_sec = timeout_sec
        # Supports MCP relay/proxy endpoint override when needed.
        self.ws_url = ws_url or os.getenv("FUGLE_WS_URL") or self.default_ws_url

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_ts(value: Any) -> datetime:
        if isinstance(value, (int, float)):
            raw = float(value)
            # Fugle timestamps may appear in seconds/ms/us/ns across channels.
            for divisor in (1.0, 1e3, 1e6, 1e9):
                try:
                    dt = datetime.fromtimestamp(raw / divisor, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    continue
                if 2000 <= dt.year <= 2100:
                    return dt
            return datetime.now(tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if text:
                if text.isdigit():
                    return TwFugleWebSocketProvider._parse_ts(int(text))
                try:
                    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
                except ValueError:
                    pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _unwrap_payload(msg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        channel = str(msg.get("channel") or "").strip().lower()
        payload = msg

        data = msg.get("data")
        if isinstance(data, dict):
            payload = data
            if not channel:
                channel = str(data.get("channel") or data.get("type") or "").strip().lower()

        return channel, payload

    @staticmethod
    def _extract_books(payload: dict[str, Any]) -> tuple[list[float], list[int], list[float], list[int]]:
        def _norm_levels(raw_levels: Any) -> tuple[list[float], list[int]]:
            prices: list[float] = []
            sizes: list[int] = []
            if not isinstance(raw_levels, list):
                return prices, sizes
            for level in raw_levels:
                if isinstance(level, dict):
                    price = TwFugleWebSocketProvider._to_float(
                        level.get("price") or level.get("p") or level.get("bid") or level.get("ask")
                    )
                    size = TwFugleWebSocketProvider._to_int(
                        level.get("size")
                        or level.get("volume")
                        or level.get("qty")
                        or level.get("q")
                        or level.get("order")
                    )
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    price = TwFugleWebSocketProvider._to_float(level[0])
                    size = TwFugleWebSocketProvider._to_int(level[1])
                else:
                    continue
                if price is None:
                    continue
                prices.append(price)
                sizes.append(size or 0)
            return prices, sizes

        bids, bid_sizes = _norm_levels(payload.get("bids") or payload.get("bid"))
        asks, ask_sizes = _norm_levels(payload.get("asks") or payload.get("ask"))
        return bids, bid_sizes, asks, ask_sizes

    @staticmethod
    def _extract_quote_fields(payload: dict[str, Any]) -> dict[str, Any]:
        # Accept several common key variants from trade/quote channels.
        price = TwFugleWebSocketProvider._to_float(
            payload.get("price")
            or payload.get("lastPrice")
            or payload.get("tradePrice")
            or payload.get("closePrice")
            or payload.get("close")
            or payload.get("c")
        )
        prev_close = TwFugleWebSocketProvider._to_float(
            payload.get("previousClose") or payload.get("prevClose") or payload.get("referencePrice") or payload.get("y")
        )
        open_ = TwFugleWebSocketProvider._to_float(payload.get("openPrice") or payload.get("open") or payload.get("o"))
        high = TwFugleWebSocketProvider._to_float(payload.get("highPrice") or payload.get("high") or payload.get("h"))
        low = TwFugleWebSocketProvider._to_float(payload.get("lowPrice") or payload.get("low") or payload.get("l"))
        total = payload.get("total")
        total_volume = None
        if isinstance(total, dict):
            total_volume = total.get("tradeVolume")
        volume = TwFugleWebSocketProvider._to_int(
            payload.get("cumulativeVolume")
            or payload.get("totalVolume")
            or payload.get("accVolume")
            or total_volume
            or payload.get("volume")
            or payload.get("size")
            or payload.get("v")
        )
        ts = TwFugleWebSocketProvider._parse_ts(
            payload.get("timestamp")
            or payload.get("time")
            or payload.get("datetime")
            or payload.get("lastUpdatedAt")
            or payload.get("t")
        )
        name = payload.get("name") or payload.get("symbolName") or payload.get("stockName")
        return {
            "price": price,
            "prev_close": prev_close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "ts": ts,
            "name": str(name).strip() if name is not None else None,
        }

    @staticmethod
    def _extract_error_message(msg: dict[str, Any]) -> str:
        direct = msg.get("message") or msg.get("error")
        if direct:
            return str(direct)
        data = msg.get("data")
        if isinstance(data, dict):
            nested = data.get("message") or data.get("error")
            if nested:
                return str(nested)
        return "fugle websocket error"

    def quote(self, request: ProviderRequest) -> QuoteSnapshot:
        if request.market != "TW":
            raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Fugle WS provider only supports TW market")
        if not self.api_key:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "FUGLE_MARKETDATA_API_KEY is missing")

        try:
            import websocket
        except Exception as exc:  # pragma: no cover
            raise ProviderError(
                self.name,
                ProviderErrorKind.UNSUPPORTED,
                "websocket-client is required for Fugle WS provider",
                exc,
            ) from exc

        symbol = str(request.symbol or "").strip().upper()
        if not symbol:
            raise ProviderError(self.name, ProviderErrorKind.PARSE, "missing TW symbol")

        ws = None
        deadline = datetime.now(tz=timezone.utc).timestamp() + float(self.timeout_sec)
        state: dict[str, Any] = {
            "price": None,
            "prev_close": None,
            "open": None,
            "high": None,
            "low": None,
            "volume": None,
            "ts": datetime.now(tz=timezone.utc),
            "name": None,
            "bid_prices": [],
            "bid_sizes": [],
            "ask_prices": [],
            "ask_sizes": [],
        }

        def _update_state(payload: dict[str, Any], channel: str):
            fields = self._extract_quote_fields(payload)
            for key in ["price", "prev_close", "open", "high", "low", "volume", "name"]:
                if fields.get(key) is not None:
                    state[key] = fields[key]
            state["ts"] = fields.get("ts") or state["ts"]
            if channel in {"books", "book"} or "bids" in payload or "asks" in payload:
                bid_prices, bid_sizes, ask_prices, ask_sizes = self._extract_books(payload)
                if bid_prices:
                    state["bid_prices"] = bid_prices
                    state["bid_sizes"] = bid_sizes
                if ask_prices:
                    state["ask_prices"] = ask_prices
                    state["ask_sizes"] = ask_sizes

        try:
            ws = websocket.create_connection(self.ws_url, timeout=self.timeout_sec)
            ws.send(json.dumps({"event": "auth", "data": {"apikey": self.api_key}}))

            authenticated = False
            while datetime.now(tz=timezone.utc).timestamp() < deadline:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                event = str(msg.get("event") or "").strip().lower()
                if event == "authenticated":
                    authenticated = True
                    break
                if event in {"error", "auth_error", "unauthorized"}:
                    err_msg = self._extract_error_message(msg)
                    raise ProviderError(self.name, ProviderErrorKind.AUTH, err_msg)
            if not authenticated:
                raise ProviderError(self.name, ProviderErrorKind.AUTH, "fugle websocket auth timeout")

            ws.send(json.dumps({"event": "subscribe", "data": {"channel": "trades", "symbol": symbol}}))
            ws.send(json.dumps({"event": "subscribe", "data": {"channel": "books", "symbol": symbol}}))

            while datetime.now(tz=timezone.utc).timestamp() < deadline:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                if isinstance(msg, list):
                    for item in msg:
                        if isinstance(item, dict):
                            channel, payload = self._unwrap_payload(item)
                            if isinstance(payload, dict):
                                _update_state(payload, channel)
                    if state["price"] is not None:
                        break
                    continue

                if not isinstance(msg, dict):
                    continue

                event = str(msg.get("event") or "").strip().lower()
                if event in {"error", "auth_error", "unauthorized"}:
                    err_msg = self._extract_error_message(msg)
                    raise ProviderError(self.name, ProviderErrorKind.NETWORK, err_msg)

                data = msg.get("data")
                if event == "data" and isinstance(data, dict):
                    channel, payload = self._unwrap_payload(msg)
                    if isinstance(payload, dict):
                        _update_state(payload, channel)
                else:
                    channel, payload = self._unwrap_payload(msg)
                    if isinstance(payload, dict):
                        _update_state(payload, channel)

                if state["price"] is not None:
                    break

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, ProviderErrorKind.NETWORK, "Fugle websocket request failed", exc) from exc
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        if state["price"] is None:
            raise ProviderError(self.name, ProviderErrorKind.EMPTY, "Fugle websocket returned empty quote")

        return QuoteSnapshot(
            symbol=symbol,
            market=request.market,
            ts=state["ts"],
            price=state["price"],
            prev_close=state["prev_close"],
            open=state["open"],
            high=state["high"],
            low=state["low"],
            volume=state["volume"],
            source=self.name,
            is_delayed=False,
            interval="tick",
            currency="TWD",
            exchange=(request.exchange or "TW").upper(),
            extra={
                "name": state["name"],
                "bid_prices": state["bid_prices"],
                "bid_sizes": state["bid_sizes"],
                "ask_prices": state["ask_prices"],
                "ask_sizes": state["ask_sizes"],
            },
        )

    def ohlcv(self, request: ProviderRequest) -> OhlcvSnapshot:
        raise ProviderError(self.name, ProviderErrorKind.UNSUPPORTED, "Fugle WS provider does not provide historical OHLCV")
