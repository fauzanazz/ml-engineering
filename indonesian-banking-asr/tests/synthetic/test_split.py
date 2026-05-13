from indonesian_banking_asr.synthetic.split import assign_split


def test_assign_split_is_deterministic():
    first = assign_split("check_balance_001", "1234567890", "Rp1.000.000")
    second = assign_split("check_balance_001", "1234567890", "Rp1.000.000")

    assert first == second
    assert first in {"train", "validation", "test"}


def test_assign_split_can_route_known_keys_to_each_split():
    # Fixtures lock hash thresholds so split behavior stays reproducible.
    assert assign_split("template_0", "1000000000", "Rp10.000") == "train"
    assert assign_split("template_2", "1000000000", "Rp10.000") == "validation"
    assert assign_split("template_30", "1000000000", "Rp10.000") == "test"
