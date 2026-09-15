"""Correct the recorded origin/target-hour mismatch without new model choices.

The initial protocol specified an origin at 14:00 and the next-hour target.
The first executed prepare.py instead selected target14 (origin13). This
separate script selects target15 (origin14) by default, reuses the existing
source ZIP, and leaves the original preparation and results untouched.
The correction occurs after the first study results were viewed: it is not
an untouched confirmatory evaluation. Training, features, and models agree
with the original prepare.py; target-hour selection is the only data change.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE
SOURCE_URL='https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip'
NAMES=['persistence','ridge_1','ridge_100','boosting_7','boosting_31',
       'blend_ridge_1','blend_boosting_7','blend_boosting_31']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main(output,source_path,target_hour):
    if not 0 <= target_hour <= 23:raise ValueError("target hour must be between 0 and 23")
    output.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'PROTOCOL.md'
    original=json.loads((ROOT/'data/initial/PREPARATION.json').read_text())
    source=source_path.resolve()
    assert sha(source)==original['source_sha256'], 'Existing source hash changed'
    correction={'reason':'Protocol origin14 implies target15; initial run used target14.',
        'after_initial_results_inspected':True,'virgin_confirmation':False,
        'target_hour':target_hour,'origin_hour':(target_hour-1)%24,
        'candidate_settings_changed':False,'new_source_download':False}
    emit(output/'CORRECTION_ACCESS.json',{'protocol_sha256':sha(protocol),
        'source_url':SOURCE_URL,'source_sha256':sha(source),
        'correction_start_unix':time.time(),**correction})
    with zipfile.ZipFile(source) as z:
        names=[n for n in z.namelist() if n.endswith('.csv')]
        assert len(names)==1,names
        frame=pd.read_csv(io.BytesIO(z.read(names[0])))
    stamp=pd.to_datetime(frame[['year','month','day','hour']])
    assert stamp.is_unique and stamp.is_monotonic_increasing
    frame.index=stamp
    full=pd.date_range(stamp.min(),stamp.max(),freq='h')
    frame=frame.reindex(full)
    features=pd.DataFrame(index=full)
    for lag in [1,2,3,6,12,24]:features[f'pm_lag_{lag}']=frame['pm2.5'].shift(lag)
    weather=['TEMP','PRES','DEWP','Iws','Is','Ir']
    for name in weather:features[name+'_lag_1']=frame[name].shift(1)
    features['wind_lag_1']=frame['cbwd'].shift(1)
    features['hour']=full.hour;features['weekday']=full.dayofweek;features['month']=full.month
    target=frame['pm2.5']
    lag_names=[n for n in features if n.startswith('pm_lag_')]
    eligible=target.notna() & features[lag_names].notna().all(axis=1)
    train=eligible & (full.year<=2012)
    selection=eligible & (full.year==2013) & (full.hour==target_hour)
    confirmation=eligible & (full.year==2014) & (full.hour==target_hour)
    numeric=[n for n in features if n!='wind_lag_1']
    transformer=ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),
        ('wind',make_pipeline(SimpleImputer(strategy='constant',fill_value='missing'),
                             OneHotEncoder(handle_unknown='ignore',sparse_output=False)),['wind_lag_1'])])
    # No fit, tuning, or output selection uses either later target window.
    features['wind_lag_1']=features['wind_lag_1'].astype(object).where(features['wind_lag_1'].notna(),np.nan)
    xt=transformer.fit_transform(features.loc[train]);yt=target.loc[train].to_numpy()
    xp={k:transformer.transform(features.loc[v]) for k,v in [('selection',selection),('confirmation',confirmation)]}
    raw={k:np.zeros((len(x),8)) for k,x in xp.items()}
    masks={'selection':selection,'confirmation':confirmation}
    for k in raw:raw[k][:,0]=features.loc[masks[k],'pm_lag_1'].to_numpy()
    train_base=features.loc[train,'pm_lag_1'].to_numpy();timings={}
    for column,alpha in [(1,1),(2,100)]:
        tick=time.perf_counter();model=Ridge(alpha=alpha).fit(xt,yt)
        for k in raw:raw[k][:,column]=model.predict(xp[k])
        timings[NAMES[column]]=time.perf_counter()-tick
    for column,leaves in [(3,7),(4,31)]:
        tick=time.perf_counter()
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,
            learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=20260914)
        model.fit(xt,yt-train_base)
        for k in raw:raw[k][:,column]=raw[k][:,0]+model.predict(xp[k])
        timings[NAMES[column]]=time.perf_counter()-tick
    for k in raw:
        raw[k][:,5:]=.5*(raw[k][:,[0]]+raw[k][:,[1,3,4]])
        raw[k]=np.maximum(raw[k],0)
        np.savez_compressed(output/(k+'_forecasts.npz'),predictions=raw[k],
            target=target.loc[masks[k]].to_numpy(),baseline=raw[k][:,0],
            timestamps=full[masks[k]].astype(str).to_numpy(dtype='U19'),
            features=features.loc[masks[k],numeric].to_numpy(),names=np.array(NAMES))
    emit(output/'PREPARATION.json',{'source_sha256':sha(source),'csv_member':names[0],
        'raw_rows':len(frame),'eligible_training_rows':int(train.sum()),
        'selection_trajectories':int(selection.sum()),'confirmation_trajectories':int(confirmation.sum()),
        'excluded_hours_missing_target_or_pm_lag':int((~eligible).sum()),
        'features':list(features),'forecast_origin':f'hour preceding target; daily target{target_hour:02d}:00 for later windows',
        'protocol_correction':correction,
        'training_end':str(full[train].max()),'selection_start':str(full[selection].min()),
        'selection_end':str(full[selection].max()),'confirmation_start':str(full[confirmation].min()),
        'candidate_order':NAMES,'training_seconds':timings,
        'stage_files':{k+'_forecasts.npz':sha(output/(k+'_forecasts.npz')) for k in raw},
        'target_scope':'Raw PM2.5; no upper clipping. Forecast maps fixed before archive draws.',
        'source_attribution':'Chen, S. (2015). Beijing PM2.5. UCI Machine Learning Repository. DOI:10.24432/C5JS49. CC BY 4.0.'})
    print(json.dumps({'status':'prepared','training_rows':int(train.sum()),
        'selection_trajectories':int(selection.sum()),'confirmation_trajectories':int(confirmation.sum())}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'data/beijing_pm25.zip')
    parser.add_argument('--target-hour',type=int,default=15)
    args=parser.parse_args();main(args.output,args.source,args.target_hour)
