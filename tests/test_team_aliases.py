from cricmaster.data.team_aliases import canonicalize_team


def test_known_franchise_renames() -> None:
    assert canonicalize_team("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"
    assert canonicalize_team("Kings XI Punjab") == "Punjab Kings"
    assert canonicalize_team("Delhi Daredevils") == "Delhi Capitals"
    assert canonicalize_team("Rising Pune Supergiant") == "Rising Pune Supergiants"
    assert canonicalize_team("St Lucia Zouks") == "St Lucia Kings"
    assert canonicalize_team("Barbados Tridents") == "Barbados Royals"


def test_unknown_names_are_trimmed_not_invented() -> None:
    assert canonicalize_team("  Mumbai Indians  ") == "Mumbai Indians"
    assert canonicalize_team("Deccan Chargers") == "Deccan Chargers"
    assert canonicalize_team("Gujarat Lions") == "Gujarat Lions"


def test_empty_name() -> None:
    assert canonicalize_team("") == ""
    assert canonicalize_team(None) == ""
