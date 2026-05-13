from indonesian_banking_asr.synthetic.rate_limit import RateLimiter


def test_rate_limiter_sleeps_between_requests():
    sleeps = []
    limiter = RateLimiter(seconds_per_request=2.5, sleep=sleeps.append)

    limiter.wait_before_request()
    limiter.wait_before_request()

    assert sleeps == [0.0, 2.5]
