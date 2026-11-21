def grid(extent, row):
    return ((extent + row["rows"] - 1) // row["rows"], 1, 1)
