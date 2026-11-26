import pandas as pd

from yc import data

PAGE = """<html><body><h3>Table 183 : Yield of SGL Transactions in Government Dated Securities for
Various Maturities (Per cent per annum)</h3><table><tr><td>Term to maturity (in years)</td>
<td>Apr.</td><td>May</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td>
<td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td></tr>
<tr><td>2011-12</td></tr>
<tr><td>1</td><td>8.1</td><td>8.2</td><td>-</td><td>8.4</td><td>8.5</td><td>8.6</td><td>8.7</td>
<td>8.8</td><td>8.9</td><td>9.0</td><td>9.1</td><td>9.2</td></tr>
<tr><td>2</td><td>8.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
<td>-</td><td>-</td><td>-</td><td>9.9</td></tr></table>
Source : Reserve Bank of India.</body></html>"""


def test_parse_maps_fiscal_months_and_skips_blanks():
    d = data.parse(PAGE)
    one = d[d["maturity"] == 1].set_index("date")["yield"]
    assert one[pd.Timestamp("2011-04-30")] == 8.1 and one[pd.Timestamp("2012-03-31")] == 9.2
    assert pd.Timestamp("2011-06-30") not in one.index  # '-' is a blank
    two = d[d["maturity"] == 2]
    assert len(two) == 2 and two["date"].max() == pd.Timestamp("2012-03-31")
    assert len(d) == 13


def test_latest_edition_from_index():
    html = ("<a class='link2' href=PublicationsView.aspx?id=24003>Table 178 : Month-end Yield of "
            "SGL Transactions in Government Dated Securities for Various Maturities</a>")
    assert data.latest_edition(html) == 24003
