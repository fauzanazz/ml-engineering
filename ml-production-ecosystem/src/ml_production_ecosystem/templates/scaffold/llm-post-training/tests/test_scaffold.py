from {{package_name}}.evaluate import pass_rate

def test_pass_rate() -> None:
    assert pass_rate([True, False]) == 0.5
