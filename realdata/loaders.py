"""Loaders for public trade data.

`load_nucs_day`  : Binance tick files from github.com/Nucs/cryptocurrency-ticks-data
                   (Id,time,Price,Quantity,IsBuyerMaker,BuyerOrderId,SellerOrderId,IsBestPriceMatch;
                   time in epoch seconds). Each file ends with one truncated row, which is dropped. The source README warns that timestamps may carry a constant
                   offset of a few hours; nothing in this test depends on calendar time.
`load_binance_vision_trades` : official files from data.binance.vision (spot `trades`), which have
                   no order ids: every print is treated as its own parent order.
"""
import zipfile
from pathlib import Path

import pandas as pd


def load_nucs_day(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    # every file in the source ends with one truncated row; drop it and any other incomplete row
    df = df.iloc[:-1].dropna()
    out = pd.DataFrame({
        "time": pd.to_datetime((df["time"] * 1000).round().astype("int64"), unit="ms"),
        "price": df["Price"].astype(float),
        "qty": df["Quantity"].astype(float),
        "buyer_maker": df["IsBuyerMaker"].astype(str).str.lower().eq("true"),
        "buy_oid": df["BuyerOrderId"].astype("int64"),
        "sell_oid": df["SellerOrderId"].astype("int64"),
    })
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def load_binance_vision_trades(path):
    """data.binance.vision spot trades: id, price, qty, quoteQty, time(ms), isBuyerMaker, isBestMatch."""
    cols = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
    df = pd.read_csv(path, header=None, names=cols)
    unit = "us" if df["time"].iloc[0] > 1e14 else "ms"
    out = pd.DataFrame({
        "time": pd.to_datetime(df["time"], unit=unit),
        "price": df["price"].astype(float),
        "qty": df["qty"].astype(float),
        "buyer_maker": df["is_buyer_maker"].astype(str).str.lower().eq("true"),
        "buy_oid": df["id"].astype("int64"),      # no order ids: each print is its own order
        "sell_oid": df["id"].astype("int64"),
    })
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def load_folder(folder, loader=load_nucs_day, pattern="*.zip"):
    files = sorted(Path(folder).glob(pattern))
    return pd.concat([loader(f) for f in files], ignore_index=True).sort_values(
        "time", kind="stable").reset_index(drop=True), [f.name for f in files]
