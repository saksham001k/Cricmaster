"""Rolling pre-2025 evaluation of roster/player-strength T20 features."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'; sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from cricmaster.models.evaluate import classification_metrics
from cricmaster.models.posttoss import POST_TOSS_FEATURES,pair_post_toss_rows
from cricmaster.models.prematch import MODEL_FEATURES,pair_prematch_rows
from cricmaster.models.roster_features import POST_ROSTER_GROUPS,PRE_ROSTER_GROUPS,append_roster_differences,build_roster_side_features
from cricmaster.models.routed import symmetric_probability
FOLDS=(("2022","2021-12-31","2022-01-01","2022-12-31"),("2023","2022-12-31","2023-01-01","2023-12-31"),("2024","2023-12-31","2024-01-01","2024-12-31"))
def _logistic(): return Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0.0,keep_empty_features=True)),('scale',StandardScaler()),('model',LogisticRegression(C=1.0,fit_intercept=False,max_iter=3000,random_state=42))])
def _tree(): return Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0.0,keep_empty_features=True)),('model',HistGradientBoostingClassifier(learning_rate=0.05,max_iter=300,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,random_state=42))])
def _candidate(name): return _logistic() if name=='logistic_regression' else _tree()
def _fit(model,frame,features):
    x=frame.loc[:,features].copy()
    for c in features: x[c]=pd.to_numeric(x[c],errors='coerce')
    y=frame['team_a_win'].astype(int); model.fit(pd.concat([x,-x],ignore_index=True),pd.concat([y,1-y],ignore_index=True)); return model
def _selected(frame,features,train_end,valid_start,valid_end):
    train=frame.loc[frame['date']<=pd.Timestamp(train_end)].copy(); valid=frame.loc[(frame['date']>=pd.Timestamp(valid_start))&(frame['date']<=pd.Timestamp(valid_end))].copy(); candidates={}
    for name in ('logistic_regression','hist_gradient_boosting'):
        model=_fit(_candidate(name),train,features); p=symmetric_probability({'model':model,'features':list(features)},valid); candidates[name]=classification_metrics(valid['team_a_win'].astype(int).to_numpy(),p)
    selected=min(candidates,key=lambda n:(candidates[n]['brier_score'],candidates[n]['log_loss']))
    return {'selected_model':selected,'metrics':candidates[selected],'candidates':candidates,'train_matches':int(len(train)),'validation_matches':int(len(valid))}
def _eval_group(frame,base_features,extra):
    folds=[]
    for label,te,vs,ve in FOLDS:
        b=_selected(frame,base_features,te,vs,ve); c=_selected(frame,(*base_features,*extra),te,vs,ve); delta=c['metrics']['brier_score']-b['metrics']['brier_score']; folds.append({'fold':label,'baseline':b,'candidate':c,'brier_delta_candidate_minus_baseline':float(delta)})
    deltas=[r['brier_delta_candidate_minus_baseline'] for r in folds]; improved=sum(d<0 for d in deltas); mean=float(np.mean(deltas)); worst=float(max(deltas))
    return {'folds':folds,'improved_folds':improved,'mean_brier_delta':mean,'worst_brier_delta':worst,'passes_gate':bool(improved>=2 and mean<=-0.001 and worst<=0.003),'extra_features':list(extra)}
def _print_group(name,r):
    f=' '.join(f"{x['fold']}={x['brier_delta_candidate_minus_baseline']:+.4f}" for x in r['folds']); print(f"{name:30} {f} | mean={r['mean_brier_delta']:+.4f} improved={r['improved_folds']}/3 gate={'PASS' if r['passes_gate'] else 'FAIL'}")
def _coverage(frame):
    w=frame.loc[(frame['date']>=pd.Timestamp('2024-01-01'))&(frame['date']<=pd.Timestamp('2024-12-31'))]
    def pct(c): return float(w[c].notna().mean()) if len(w) else float('nan')
    return {'matches':int(len(w)),'previous_xi_strength_available_pct':pct('previous_xi_mean_recent_runs_diff'),'previous_roster_stability_available_pct':pct('previous_roster_stability_diff'),'current_xi_continuity_available_pct':pct('current_previous_xi_overlap_diff'),'rest_days_available_pct':pct('log_rest_days_diff')}
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--raw',default='data/raw/cricsheet/t20_expanded'); p.add_argument('--data',default='data/processed/t20_expanded/prematch_features.parquet'); p.add_argument('--roster-cache',default='data/processed/model_comparison/step17_roster_side_pre2025.parquet'); p.add_argument('--output',default='data/processed/model_comparison/step17_roster_features.json'); a=p.parse_args(argv)
    source=pd.read_parquet(a.data); source['date']=pd.to_datetime(source['date'],errors='raise'); source=source.loc[source['date']<pd.Timestamp('2025-01-01')].copy(); cache=Path(a.roster_cache)
    if cache.exists(): print(f'Loading roster cache: {cache}'); side=pd.read_parquet(cache); side['date']=pd.to_datetime(side['date'],errors='raise')
    else:
        print('Building leakage-safe roster history from raw Cricsheet ...'); side=build_roster_side_features(a.raw,cutoff='2024-12-31'); cache.parent.mkdir(parents=True,exist_ok=True); side.to_parquet(cache,index=False); print(f'saved roster cache {cache}')
    pre=append_roster_differences(pair_prematch_rows(source).query("format == 'T20'").copy(),side); post=append_roster_differences(pair_post_toss_rows(source).query("format == 'T20'").copy(),side)
    report={'policy':{'uses_2025_plus':False,'raw_cutoff':'2024-12-31','folds':[list(x) for x in FOLDS]},'coverage_2024':_coverage(post),'prematch':{},'posttoss':{}}
    print('\n=== 2024 ROSTER FEATURE COVERAGE ==='); [print(f"{k}: {v if k=='matches' else f'{v:.3f}'}") for k,v in report['coverage_2024'].items()]
    print('\n=== PRE_TOSS ROSTER FEATURE GROUPS ===')
    for name,features in PRE_ROSTER_GROUPS.items(): r=_eval_group(pre,MODEL_FEATURES,features); report['prematch'][name]=r; _print_group(name,r)
    print('\n=== POST_TOSS ROSTER FEATURE GROUPS ===')
    for name,features in POST_ROSTER_GROUPS.items(): r=_eval_group(post,POST_TOSS_FEATURES,features); report['posttoss'][name]=r; _print_group(name,r)
    passed_pre=[n for n,r in report['prematch'].items() if r['passes_gate']]; passed_post=[n for n,r in report['posttoss'].items() if r['passes_gate']]; print(f"\nPRE_TOSS passed groups: {passed_pre or 'none'}"); print(f"POST_TOSS passed groups: {passed_post or 'none'}")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(f'saved {out}'); return 0
if __name__=='__main__': raise SystemExit(main())
