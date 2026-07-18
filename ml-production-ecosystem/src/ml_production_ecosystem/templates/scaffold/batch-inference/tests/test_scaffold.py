from {{package_name}}.batch import predict_batch

def test_predict_batch_preserves_rows() -> None:
    assert len(predict_batch([{"id": 1}, {"id": 2}])) == 2
