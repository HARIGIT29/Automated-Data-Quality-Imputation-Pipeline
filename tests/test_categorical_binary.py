"""Binary bit-string categorical encoding."""

import pandas as pd

from src.categorical import encode_categoricals


def test_binary_bits_france_germany_spain_order():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "Country": ["Spain", "France", "Germany"],
        }
    )
    out = encode_categoricals(df, ["Country"], method="binary_bits")
    assert "Country_binary" in out.columns
    assert list(out["Country_binary"]) == ["001", "100", "010"]


def test_binary_bits_purchased_two_levels():
    df = pd.DataFrame({"Purchased": ["Yes", "No", "Yes"]})
    out = encode_categoricals(df, ["Purchased"], method="binary_bits")
    # Alphabetically No, Yes -> No=10, Yes=01
    assert list(out["Purchased_binary"]) == ["01", "10", "01"]
