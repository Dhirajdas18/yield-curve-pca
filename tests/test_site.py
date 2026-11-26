import numpy as np
import pandas as pd

from yc import site


def test_page_builds_with_data_and_format_parts():
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    w = pd.DataFrame(6 + np.random.default_rng(0).normal(0, 0.1, (30, 3)), index=idx,
                     columns=[1, 2, 10])
    w.iloc[0, 2] = np.nan
    html = site.build_html(w)
    assert "$" + "data" not in html and "$" + "js" not in html
    assert "Updated " in html and "Data: Reserve Bank of India" in html
    assert "Research only" in html and "How to read it:" in html
    assert '"months":["2020-01"' in html and '"10":[null,' in html
