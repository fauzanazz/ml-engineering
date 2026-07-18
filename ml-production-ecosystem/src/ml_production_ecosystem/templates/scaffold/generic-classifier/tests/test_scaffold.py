from {{package_name}}.predict import predict

def test_predict_returns_label() -> None:
    assert predict({"x": 1.0}) == "positive"
