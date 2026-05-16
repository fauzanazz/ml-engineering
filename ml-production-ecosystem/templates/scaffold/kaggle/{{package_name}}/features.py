def select_numeric_features(rows):
    return rows.select_dtypes(include="number")
