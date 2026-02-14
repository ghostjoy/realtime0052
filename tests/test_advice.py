from __future__ import annotations

import unittest

import pandas as pd

from advice import Profile, _position_range, render_advice_scai_style


class AdviceTests(unittest.TestCase):
    def test_position_range_respects_risk_adjustment(self):
        lo_c, hi_c = _position_range(points=0, risk="保守")
        lo_n, hi_n = _position_range(points=0, risk="一般")
        lo_a, hi_a = _position_range(points=0, risk="積極")
        self.assertLessEqual(lo_c, lo_n)
        self.assertLessEqual(hi_c, hi_n)
        self.assertGreaterEqual(lo_a, lo_n)
        self.assertGreaterEqual(hi_a, hi_n)

    def test_scai_style_includes_position_line(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
        text = render_advice_scai_style(df, Profile(horizon="中期", risk="一般", style="趨勢"), symbol="2330")
        self.assertIn("倉位建議：", text)

    def test_scai_style_neutral_mentions_small_position(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)
        text = render_advice_scai_style(df, Profile(horizon="中期", risk="一般", style="趨勢"), symbol="2330")
        self.assertIn("可小倉位試單", text)


if __name__ == "__main__":
    unittest.main()
