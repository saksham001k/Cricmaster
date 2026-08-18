from cricmaster.live.mock import MOCK_MATCH_ID, MockLiveCricketProvider


def test_mock_provider_returns_demo_match() -> None:
    provider = MockLiveCricketProvider()
    live = provider.get_live_matches()
    assert len(live) == 1
    assert live[0].match_id == MOCK_MATCH_ID
    match = provider.get_match(MOCK_MATCH_ID)
    assert match is not None
    assert match.source == "mock-live"
    assert match.current_innings is not None
    assert match.current_innings.runs == 45
    score = provider.get_score(MOCK_MATCH_ID)
    assert score is not None
    assert score.wickets == 1


def test_mock_provider_can_be_unavailable() -> None:
    provider = MockLiveCricketProvider(available=False)
    assert provider.get_live_matches() == []
    assert provider.get_match(MOCK_MATCH_ID) is None
    assert provider.get_score("unknown") is None
