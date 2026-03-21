from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services.etf_briefing import export_etf_briefing_artifact


class EtfBriefingTests(unittest.TestCase):
    def test_export_etf_briefing_writes_expected_bundle(self):
        fake_super_export = pd.DataFrame(
            [
                {
                    "代碼": "^TWII",
                    "ETF": "加權指數",
                    "類型": "",
                    "YTD績效(%)": 8.2,
                    "ETF規模(億)": "",
                },
                {
                    "代碼": "0052",
                    "ETF": "富邦科技",
                    "類型": "科技型",
                    "YTD績效(%)": 12.5,
                    "ETF規模(億)": 90.5,
                },
                {
                    "代碼": "00935",
                    "ETF": "野村臺灣新科技50",
                    "類型": "科技型",
                    "YTD績效(%)": 9.3,
                    "ETF規模(億)": 44.2,
                },
                {
                    "代碼": "00980A",
                    "ETF": "主動台灣",
                    "類型": "主動式",
                    "YTD績效(%)": 6.8,
                    "ETF規模(億)": 21.7,
                },
            ]
        )

        def _fake_super_export(**kwargs):
            out_dir = Path(str(kwargs["out"]))
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "tw_etf_super_export_20260322.csv"
            fake_super_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return {
                "output_path": str(csv_path),
                "trade_date_anchor": "20260322",
            }

        def _fake_prepare(**kwargs):
            clean_symbols = [str(item).strip().upper() for item in list(kwargs.get("symbols") or []) if str(item).strip()]
            return {"included": clean_symbols, "excluded": []}

        def _fake_chart(**kwargs):
            out_path = Path(str(kwargs["out_path"]))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"png")
            return {"path": str(out_path), "symbols": list(kwargs.get("symbols") or []), "layout": kwargs.get("layout")}

        def _fake_split(**kwargs):
            out_dir = Path(str(kwargs["out_dir"]))
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for symbol in list(kwargs.get("symbols") or []):
                path = out_dir / f"{symbol}.png"
                path.write_bytes(b"png")
                paths.append(str(path))
            return {"paths": paths, "symbols": list(kwargs.get("symbols") or [])}

        fake_news = {
            "generated_at": "2026-03-22T00:00:00+00:00",
            "window_days": 14,
            "groups": [
                {
                    "key": "taiwan_us_china",
                    "label": "台海與美中",
                    "items": [
                        {
                            "title": "晶片與關稅談判再起波折",
                            "published_date": "2026-03-20",
                            "url": "https://example.com/a",
                        }
                    ],
                },
                {
                    "key": "ukraine_russia",
                    "label": "俄烏戰局",
                    "items": [
                        {
                            "title": "前線補給再度吃緊",
                            "published_date": "2026-03-18",
                            "url": "https://example.com/b",
                        }
                    ],
                },
                {
                    "key": "middle_east_energy",
                    "label": "中東能源與航運",
                    "items": [
                        {
                            "title": "紅海航線保費再上修",
                            "published_date": "2026-03-19",
                            "url": "https://example.com/c",
                        }
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("services.etf_briefing.export_tw_etf_super_table_artifact", side_effect=_fake_super_export),
                patch("services.etf_briefing._prepare_chartable_symbols", side_effect=_fake_prepare),
                patch("services.etf_briefing._export_chart_safe", side_effect=_fake_chart),
                patch("services.etf_briefing._export_split_chart_group", side_effect=_fake_split),
                patch("services.etf_briefing.collect_recent_geopolitics_news", return_value=fake_news),
            ):
                result = export_etf_briefing_artifact(
                    store=object(),  # type: ignore[arg-type]
                    out_root=tmpdir,
                    start="2023-01-01",
                    end="2026-03-22",
                    single_symbol="00935",
                    include_extra_splits=True,
                )

            briefing_dir = Path(str(result["briefing_dir"]))
            self.assertTrue((briefing_dir / "tw_etf_super_export_20260322.csv").exists())
            self.assertTrue((briefing_dir / "tech_etf_from_super_export_20260322.csv").exists())
            self.assertTrue((briefing_dir / "active_etf_from_super_export_20260322.csv").exists())
            self.assertTrue((briefing_dir / "report_sima_yi.html").exists())
            self.assertTrue((briefing_dir / "fb_post_sima_yi.txt").exists())
            self.assertTrue((briefing_dir / "fb_post_sima_yi.html").exists())
            self.assertTrue((briefing_dir / "manifest.json").exists())
            self.assertTrue((briefing_dir / "news_sources.json").exists())
            self.assertTrue((briefing_dir / "charts" / "tech_etf_combined_20230101_20260322.png").exists())
            self.assertTrue((briefing_dir / "charts" / "active_etf_combined_20230101_20260322.png").exists())
            self.assertTrue((briefing_dir / "charts" / "00935_single_20230101_20260322.png").exists())
            self.assertTrue((briefing_dir / "charts" / "tech_etf_split" / "0052.png").exists())
            self.assertTrue((briefing_dir / "charts" / "active_etf_split" / "00980A.png").exists())

            tech_df = pd.read_csv(briefing_dir / "tech_etf_from_super_export_20260322.csv", dtype=str)
            active_df = pd.read_csv(briefing_dir / "active_etf_from_super_export_20260322.csv", dtype=str)
            self.assertEqual(tech_df["代碼"].astype(str).tolist(), ["0052", "00935"])
            self.assertEqual(active_df["代碼"].astype(str).tolist(), ["00980A"])

            html_text = (briefing_dir / "report_sima_yi.html").read_text(encoding="utf-8")
            self.assertIn("司馬懿為主公進呈 ETF 戰報", html_text)
            self.assertIn("科技型 ETF 兵勢", html_text)
            self.assertIn("00935 單檔觀察", html_text)
            self.assertIn("charts/tech_etf_combined_20230101_20260322.png", html_text)
            self.assertNotIn("result&#x27;", html_text)
            self.assertIn("<td>0052</td>", html_text)

            fb_text = (briefing_dir / "fb_post_sima_yi.txt").read_text(encoding="utf-8")
            self.assertIn("主公。", fb_text)
            self.assertIn("#ETF", fb_text)

            manifest = json.loads((briefing_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tech_symbols"], ["0052", "00935"])
            self.assertEqual(manifest["active_symbols"], ["00980A"])


if __name__ == "__main__":
    unittest.main()
