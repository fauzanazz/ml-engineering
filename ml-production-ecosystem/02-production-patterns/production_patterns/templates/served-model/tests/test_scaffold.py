from {{package_name}}.predict import predict


def test_predict_sums_features() -> None:
    assert predict({"a": 1.0, "b": 2.0}) == 3.0
