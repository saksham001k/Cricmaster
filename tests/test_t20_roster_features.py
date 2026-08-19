import math
from cricmaster.features.player_form import PlayerFormBook
from cricmaster.models.roster_features import _lineup_strength, overlap_ratio

def test_overlap_ratio_measures_returning_players():
    assert overlap_ratio(['A','B','C','D'],['B','C','D','E'])==0.75

def test_overlap_ratio_requires_both_lineups():
    assert math.isnan(overlap_ratio(None,['A']))
    assert math.isnan(overlap_ratio(['A'],None))

def test_empty_player_history_produces_missing_strength_not_fake_zero():
    s=_lineup_strength(PlayerFormBook(),'T20',['A','B','C'])
    assert math.isnan(s['mean_batting_average'])
    assert math.isnan(s['core_batting_recent_runs'])
    assert math.isnan(s['core_bowling_recent_economy'])
