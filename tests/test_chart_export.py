from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services.chart_export import _build_multi_line_styles, export_backtest_chart_artifact


def _bars_frame(
    *,
    start: str = "2025-01-01",
    periods: int = 140,
    base: float = 100.0,
    slope: float = 0.6,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="B", tz="UTC")
    step = pd.Series(range(periods), index=index, dtype=float)
    close = base + step * float(slope)
    frame = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 0.9,
            "low": close - 0.9,
            "close": close,
            "adj_close": close,
            "volume": 1000.0 + step * 10.0,
            "source": "test",
        },
        index=index,
    )
    return frame


class FakeStore:
    def __init__(self, bars_map: dict[tuple[str, str], pd.DataFrame]):
        self._bars_map = bars_map
        self._metadata = {
            ("TW", "0050"): {"name": "元大台灣50"},
            ("TW", "0052"): {"name": "富邦科技"},
            ("TW", "006208"): {"name": "富邦台50"},
            ("US", "SPY"): {"name": "SPDR S&P 500 ETF Trust"},
        }

    def load_daily_bars(
        self, symbol: str, market: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        frame = self._bars_map.get((market, symbol), pd.DataFrame()).copy()
        if frame.empty:
            return frame
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        return frame.loc[mask].copy()

    def sync_symbol_history(self, symbol: str, market: str, start: datetime, end: datetime):
        return None

    def load_symbol_metadata(self, symbols: list[str], market: str) -> dict[str, dict[str, object]]:
        return {symbol: self._metadata.get((market, symbol), {"name": symbol}) for symbol in symbols}


class ChartExportTests(unittest.TestCase):
    def setUp(self):
        tw_symbol = _bars_frame(base=120.0, slope=1.2)
        tw_bench = _bars_frame(base=100.0)
        us_symbol = _bars_frame(base=300.0)
        us_bench = _bars_frame(base=4500.0)
        self.store = FakeStore(
            {
                ("TW", "0050"): tw_symbol,
                ("TW", "0052"): _bars_frame(base=88.0),
                ("TW", "^TWII"): tw_bench,
                ("TW", "006208"): _bars_frame(base=95.0),
                ("US", "SPY"): us_symbol,
                ("US", "^GSPC"): us_bench,
            }
        )
        self.start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.end = datetime(2025, 7, 31, tzinfo=timezone.utc)

    @patch("services.chart_export._write_figure_image")
    def test_single_layout_exports_png_with_benchmark_layers(self, write_mock):
        captured: list[tuple[object, Path]] = []

        def _capture(fig, *, output_path, width, height, scale):
            captured.append((fig, output_path))

        write_mock.side_effect = _capture

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_backtest_chart_artifact(
                store=self.store,
                symbols=["0050"],
                layout="single",
                market="TW",
                start=self.start,
                end=self.end,
                strategy="buy_hold",
                benchmark_choice="auto",
                initial_capital=1_000_000.0,
                fee_rate=None,
                sell_tax=None,
                slippage=None,
                sync_before_run=False,
                use_split_adjustment=True,
                use_total_return_adjustment=True,
                theme="soft-gray",
                width=1600,
                height=900,
                scale=2,
                out_dir=tmpdir,
            )

        self.assertEqual(result["format"], "png")
        self.assertEqual(result["exported_count"], 1)
        fig, output_path = captured[0]
        self.assertEqual(output_path.suffix, ".png")
        trace_types = [trace.type for trace in fig.data]
        self.assertIn("candlestick", trace_types)
        trace_names = [trace.name for trace in fig.data]
        self.assertIn("TWII（同基準價）", trace_names)
        self.assertIn("Benchmark Equity", trace_names)
        self.assertEqual(fig.layout.yaxis2.tickformat, "~s")
        self.assertEqual(fig.layout.yaxis.tickformat, ",.0f")
        self.assertEqual(fig.layout.legend.orientation, "v")
        annotation_texts = [str(annotation.text) for annotation in fig.layout.annotations]
        self.assertTrue(any("<b>Equity</b>" in text for text in annotation_texts))
        self.assertTrue(any("Benchmark Eq." in text for text in annotation_texts))
        summary_box = next(text for text in annotation_texts if "Benchmark Eq." in text)
        self.assertIn("color:", summary_box)

    @patch("services.chart_export._write_figure_image")
    def test_combined_layout_uses_normalized_lines(self, write_mock):
        captured = []

        def _capture(fig, *, output_path, width, height, scale):
            captured.append(fig)

        write_mock.side_effect = _capture

        export_backtest_chart_artifact(
            store=self.store,
            symbols=["0050", "0052"],
            layout="combined",
            market="TW",
            start=self.start,
            end=self.end,
            strategy="buy_hold",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=False,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme="soft-gray",
            width=1600,
            height=900,
            scale=2,
        )

        fig = captured[0]
        trace_types = [trace.type for trace in fig.data]
        self.assertNotIn("candlestick", trace_types)
        trace_names = [trace.name for trace in fig.data]
        self.assertIn("Benchmark (^TWII)", trace_names)
        self.assertNotIn("Buy and Hold (EW Portfolio)", trace_names)
        self.assertIn("Buy and Hold (0050 元大台灣50) *", trace_names)
        self.assertIn("Buy and Hold (0052 富邦科技) *", trace_names)
        self.assertEqual(fig.data[0].type, "scatter")
        self.assertEqual(fig.layout.yaxis.title.text, None)
        self.assertGreaterEqual(min(fig.data[0].y), 1.0)
        self.assertEqual(fig.layout.legend.orientation, "v")
        self.assertIn("* = final value above benchmark", trace_names)
        self.assertEqual(trace_names[0], "Benchmark (^TWII)")

    @patch("services.chart_export._write_figure_image")
    def test_combined_layout_can_include_ew_portfolio_line(self, write_mock):
        captured = []

        def _capture(fig, *, output_path, width, height, scale):
            captured.append(fig)

        write_mock.side_effect = _capture

        export_backtest_chart_artifact(
            store=self.store,
            symbols=["0050", "0052"],
            layout="combined",
            market="TW",
            start=self.start,
            end=self.end,
            strategy="buy_hold",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=False,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme="soft-gray",
            width=1600,
            height=900,
            scale=2,
            include_ew_portfolio=True,
        )

        trace_names = [trace.name for trace in captured[0].data]
        self.assertIn("Buy and Hold (EW Portfolio) *", trace_names)

    @patch("services.chart_export._write_figure_image")
    def test_single_layout_reference_annotations_add_markers(self, write_mock):
        captured: list[object] = []

        def _capture(fig, *, output_path, width, height, scale):
            captured.append(fig)

        write_mock.side_effect = _capture

        export_backtest_chart_artifact(
            store=self.store,
            symbols=["0050"],
            layout="single",
            market="TW",
            start=self.start,
            end=self.end,
            strategy="sma_cross",
            benchmark_choice="auto",
            initial_capital=1_000_000.0,
            fee_rate=None,
            sell_tax=None,
            slippage=None,
            sync_before_run=False,
            use_split_adjustment=True,
            use_total_return_adjustment=True,
            theme="soft-gray",
            width=1600,
            height=900,
            scale=2,
            reference_annotations=True,
        )

        fig = captured[0]
        trace_names = [trace.name for trace in fig.data]
        self.assertIn("Buy Signal", trace_names)
        self.assertIn("Buy Fill", trace_names)
        self.assertIn("Sell Fill", trace_names)
        self.assertIn("Trade Path", trace_names)
        self.assertTrue(any(name in {"Buy Signal", "Sell Signal"} for name in trace_names))
        annotation_texts = [str(annotation.text) for annotation in fig.layout.annotations]
        self.assertTrue(any(text.startswith("最高價") for text in annotation_texts))
        self.assertTrue(any(text.startswith("最低價") for text in annotation_texts))
        self.assertTrue(fig.layout.shapes)

    @patch("services.chart_export._write_figure_image")
    def test_split_layout_exports_multiple_pngs_for_mixed_markets(self, write_mock):
        captured: list[tuple[str, object]] = []

        def _capture(fig, *, output_path, width, height, scale):
            captured.append((str(output_path), fig))

        write_mock.side_effect = _capture

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_backtest_chart_artifact(
                store=self.store,
                symbols=["0050", "SPY"],
                layout="split",
                market="auto",
                start=self.start,
                end=self.end,
                strategy="buy_hold",
                benchmark_choice="auto",
                initial_capital=1_000_000.0,
                fee_rate=None,
                sell_tax=None,
                slippage=None,
                sync_before_run=False,
                use_split_adjustment=True,
                use_total_return_adjustment=True,
                theme="data-dark",
                width=1600,
                height=900,
                scale=2,
                out_dir=tmpdir,
            )

        self.assertEqual(result["exported_count"], 2)
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(path.endswith(".png") for path, _ in captured))
        all_names = [trace.name for _, fig in captured for trace in fig.data]
        self.assertEqual(all_names.count("Benchmark Equity"), 2)

    def test_single_layout_rejects_multiple_symbols(self):
        with self.assertRaisesRegex(ValueError, "single layout requires exactly one symbol"):
            export_backtest_chart_artifact(
                store=self.store,
                symbols=["0050", "0052"],
                layout="single",
                market="TW",
                start=self.start,
                end=self.end,
                strategy="buy_hold",
                benchmark_choice="auto",
                initial_capital=1_000_000.0,
                fee_rate=None,
                sell_tax=None,
                slippage=None,
                sync_before_run=False,
                use_split_adjustment=True,
                use_total_return_adjustment=True,
                theme="soft-gray",
                width=1600,
                height=900,
                scale=2,
            )

    def test_build_multi_line_styles_keeps_all_asset_lines_solid(self):
        styles = _build_multi_line_styles(
            ["00988A", "00992A", "00994A", "00991A", "00987A", "00981A", "00985A", "00990A", "00995A"],
            palette={"asset_palette": ["#1D4ED8", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#BE123C", "#65A30D"]},
        )
        self.assertEqual(styles["00981A"]["dash"], "solid")
        self.assertEqual(styles["00995A"]["dash"], "solid")


if __name__ == "__main__":
    unittest.main()
