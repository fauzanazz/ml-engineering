from {{package_name}}.rank import recommend

def test_recommend_limits_candidates() -> None:
    assert recommend("u1", ["a", "b"], limit=1) == ["a"]
