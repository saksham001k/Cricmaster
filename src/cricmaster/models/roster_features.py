"""Leakage-safe experimental roster features for franchise/domestic T20."""
from __future__ import annotations
import json, math
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from cricmaster.data.cricsheet import discover_match_files, load_match
from cricmaster.data.formats import MatchFormat
from cricmaster.data.team_aliases import canonicalize_team
from cricmaster.features.batting import extract_batting_innings, team_xi
from cricmaster.features.bowling import extract_bowling_spells
from cricmaster.features.player_form import PlayerFormBook

MEAN_STRENGTH_FIELDS=("mean_batting_average","mean_recent_runs","mean_bowling_economy","mean_recent_wickets")
CORE_STRENGTH_FIELDS=("core_batting_recent_runs","core_batting_recent_strike_rate","core_bowling_recent_wickets","core_bowling_recent_economy")
PREVIOUS_MEAN_DIFFS=tuple(f"previous_xi_{x}_diff" for x in MEAN_STRENGTH_FIELDS)
PREVIOUS_CORE_DIFFS=tuple(f"previous_xi_{x}_diff" for x in CORE_STRENGTH_FIELDS)
CURRENT_CORE_DIFFS=tuple(f"current_xi_{x}_diff" for x in CORE_STRENGTH_FIELDS)
PRE_ROSTER_GROUPS={
    "previous_xi_mean_strength":PREVIOUS_MEAN_DIFFS,
    "previous_xi_core_strength":PREVIOUS_CORE_DIFFS,
    "previous_roster_stability":("previous_roster_stability_diff",),
    "rest_days":("log_rest_days_diff",),
}
POST_ROSTER_GROUPS={**PRE_ROSTER_GROUPS,
    "current_xi_core_strength":CURRENT_CORE_DIFFS,
    "current_xi_continuity":("current_previous_xi_overlap_diff",),
}

def overlap_ratio(first:list[str]|None, second:list[str]|None)->float:
    if not first or not second: return float('nan')
    first_set=set(first)
    return len(first_set & set(second))/len(first_set) if first_set else float('nan')

def _mean(values:list[float])->float:
    return float(sum(values)/len(values)) if values else float('nan')

def _lineup_strength(players:PlayerFormBook, match_format:MatchFormat|str, lineup:list[str]|None)->dict[str,float]:
    empty={name:float('nan') for name in (*MEAN_STRENGTH_FIELDS,*CORE_STRENGTH_FIELDS)}
    if not lineup: return empty
    batting=[players.batter_snapshot(match_format,p) for p in lineup]
    bowling=[players.bowler_snapshot(match_format,p) for p in lineup]
    bat_core=sorted((r for r in batting if int(r['innings'])>0), key=lambda r:int(r['innings']), reverse=True)[:7]
    bowl_core=sorted((r for r in bowling if int(r['balls'])>0), key=lambda r:int(r['balls']), reverse=True)[:5]
    return {
        "mean_batting_average":_mean([float(r['average']) for r in batting if r['average'] is not None]),
        "mean_recent_runs":_mean([float(r['recent_runs']) for r in batting if r['recent_runs'] is not None]),
        "mean_bowling_economy":_mean([float(r['economy']) for r in bowling if r['economy'] is not None]),
        "mean_recent_wickets":_mean([float(r['recent_wickets']) for r in bowling if r['recent_wickets'] is not None]),
        "core_batting_recent_runs":_mean([float(r['recent_runs']) for r in bat_core if r['recent_runs'] is not None]),
        "core_batting_recent_strike_rate":_mean([float(r['recent_strike_rate']) for r in bat_core if r['recent_strike_rate'] is not None]),
        "core_bowling_recent_wickets":_mean([float(r['recent_wickets']) for r in bowl_core if r['recent_wickets'] is not None]),
        "core_bowling_recent_economy":_mean([float(r['recent_economy']) for r in bowl_core if r['recent_economy'] is not None]),
    }

def _peek_date(path:Path)->date|None:
    try:
        payload=json.loads(path.read_text(encoding='utf-8')); info=payload.get('info') if isinstance(payload,dict) else None; dates=info.get('dates') if isinstance(info,dict) else None
        return date.fromisoformat(str(dates[0])[:10]) if dates else None
    except (OSError,ValueError,TypeError,json.JSONDecodeError): return None

def build_roster_side_features(input_dir:str|Path, *, cutoff:str='2024-12-31')->pd.DataFrame:
    cutoff_date=date.fromisoformat(cutoff); ordered=[]
    for path in discover_match_files(input_dir):
        d=_peek_date(path)
        if d is not None and d<=cutoff_date: ordered.append((d,path.stem,path))
    ordered.sort(key=lambda x:(x[0],x[1],x[2].as_posix()))
    player_book=PlayerFormBook(); xi_history=defaultdict(lambda:deque(maxlen=2)); last_match_date={}; seen=set(); records=[]
    for match_date,match_id,path in ordered:
        if match_id in seen: continue
        seen.add(match_id); match=load_match(path)
        if match.metadata.format is not MatchFormat.T20: continue
        meta=match.metadata; gender=meta.gender or ''
        for raw_team in (meta.team1,meta.team2):
            team=canonicalize_team(raw_team); key=(str(meta.format),gender,team); hist=xi_history[key]
            previous_xi=list(hist[-1]) if hist else None; previous_previous_xi=list(hist[-2]) if len(hist)>=2 else None; current_xi=team_xi(match,raw_team)
            prev_strength=_lineup_strength(player_book,meta.format,previous_xi); current_strength=_lineup_strength(player_book,meta.format,current_xi)
            prev_date=last_match_date.get(key); rest_days=(match_date-prev_date).days if prev_date is not None else None
            row={"match_id":match_id,"date":match_date.isoformat(),"team":team,
                 "previous_xi_known":int(previous_xi is not None),"current_xi_known":int(current_xi is not None),
                 "previous_roster_stability":overlap_ratio(previous_xi,previous_previous_xi),
                 "current_previous_xi_overlap":overlap_ratio(current_xi,previous_xi),
                 "rest_days":rest_days,
                 "log_rest_days":math.log1p(min(max(rest_days,0),120)) if rest_days is not None else float('nan')}
            row.update({f"previous_xi_{k}":v for k,v in prev_strength.items()}); row.update({f"current_xi_{k}":v for k,v in current_strength.items()}); records.append(row)
        player_book.update_batting(meta.format,extract_batting_innings(match)); player_book.update_bowling(meta.format,extract_bowling_spells(match))
        for raw_team in (meta.team1,meta.team2):
            team=canonicalize_team(raw_team); key=(str(meta.format),gender,team); current_xi=team_xi(match,raw_team)
            if current_xi: xi_history[key].append(list(current_xi))
            last_match_date[key]=match_date
    result=pd.DataFrame.from_records(records)
    if result.empty: return result
    result['date']=pd.to_datetime(result['date'],errors='raise')
    return result.sort_values(['date','match_id','team'],kind='stable').reset_index(drop=True)

def append_roster_differences(paired:pd.DataFrame, side_features:pd.DataFrame)->pd.DataFrame:
    numeric=[c for c in side_features.columns if c not in {'match_id','date','team'}]; lookup=side_features.set_index(['match_id','team']); result=paired.copy()
    for col in numeric:
        values=[]
        for _,row in result.iterrows():
            try:
                a=lookup.loc[(str(row['match_id']),str(row['team_a'])),col]; b=lookup.loc[(str(row['match_id']),str(row['team_b'])),col]
            except KeyError:
                values.append(float('nan')); continue
            a=pd.to_numeric(pd.Series([a]),errors='coerce').iloc[0]; b=pd.to_numeric(pd.Series([b]),errors='coerce').iloc[0]
            values.append(float(a-b) if pd.notna(a) and pd.notna(b) else float('nan'))
        result[f'{col}_diff']=values
    return result
