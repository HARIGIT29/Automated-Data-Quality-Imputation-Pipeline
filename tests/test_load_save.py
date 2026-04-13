"""Encoding fallbacks for CSV bytes."""

from src.load_save import read_csv_bytes


def test_read_csv_bytes_utf8():
    raw = "a,b\n1,2\n".encode("utf-8")
    df, enc = read_csv_bytes(raw)
    assert enc == "utf-8-sig" or enc == "utf-8"
    assert list(df.columns) == ["a", "b"]


def test_read_csv_bytes_cp1252_nbsp():
    # Byte 0xA0 is NBSP in cp1252; invalid as UTF-8 multi-byte start
    raw = "name,value\nfoo,\xa0bar\n".encode("latin-1")
    df, enc = read_csv_bytes(raw)
    assert enc in ("cp1252", "latin-1")
    assert "bar" in str(df.iloc[0, 1]) or "\xa0" in str(df.iloc[0, 1])


def test_read_csv_bytes_bom_utf8():
    raw = "\ufeffx,y\n1,2\n".encode("utf-8-sig")
    df, enc = read_csv_bytes(raw)
    assert enc == "utf-8-sig"
    assert "x" in df.columns
