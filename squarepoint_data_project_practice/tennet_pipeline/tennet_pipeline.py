"""Reproducible TenneT analysis and daily, fixed-origin, 24-hour forecasting.

Run: python tennet_pipeline.py --data INPUT.csv --mapping MAPPING.csv --out RESULTS
All evaluation uses observed targets, never imputed labels. Source files are read-only.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, re, time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
import joblib

TARGET = 'total_load'
SEED = 42
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 160, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})

def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)

def dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))

def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def load_data(data_path, mapping_path, out=None):
    raw = pd.read_csv(data_path)
    mapping = pd.read_csv(mapping_path, dtype={'col_id': str})
    if mapping.col_id.duplicated().any():
        raise ValueError('Duplicate mapping IDs')
    lookup = mapping.set_index('col_id')
    columns = [c for c in raw if c not in ['datetime_utc', 'datetime_berlin']]
    records = []
    for c in columns:
        ident = c.split('_')[0]
        if ident not in lookup.index:
            raise ValueError(f'Unmapped column: {c}')
        r = lookup.loc[ident]
        records.append(dict(original_column=c, col_id=ident, model_column=slug(r.english_name),
                            english_name=r.english_name, deutscher_name=r.deutscher_name, unit=r.unit))
    dictionary = pd.DataFrame(records)
    assert dictionary.model_column.is_unique
    utc = pd.to_datetime(raw.datetime_utc, utc=True, errors='coerce')
    berlin_as_utc = pd.to_datetime(raw.datetime_berlin, utc=True, errors='coerce')
    audit = dict(rows=len(raw), numeric_columns=len(columns), timestamp_parse_failures=int(utc.isna().sum()),
                 berlin_parse_failures=int(berlin_as_utc.isna().sum()),
                 duplicate_utc=int(utc.duplicated().sum()), original_monotonic=bool(utc.is_monotonic_increasing),
                 timezone_disagreements=int((utc != berlin_as_utc).sum()),
                 original_delta_counts={str(k):int(v) for k,v in utc.diff().value_counts().items()})
    if audit['timestamp_parse_failures'] or audit['duplicate_utc'] or audit['timezone_disagreements']:
        raise ValueError(f'Ambiguous timestamps; resolve before forecasting: {audit}')
    df = raw[columns].rename(columns=dict(zip(dictionary.original_column, dictionary.model_column)))
    df.index = pd.DatetimeIndex(utc, name='datetime_utc')
    for c in df:
        df[c] = pd.to_numeric(df[c], errors='raise')
    audit['infinite_values'] = int(np.isinf(df.to_numpy()).sum())
    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq='h', name='datetime_utc')
    audit['missing_timestamp_rows'] = len(full.difference(df.index))
    df = df.reindex(full)
    local = df.index.tz_convert('Europe/Berlin')
    audit.update(start_utc=str(df.index.min()), end_utc=str(df.index.max()),
                 missing_values=int(df.isna().sum().sum()), missing_by_column=df.isna().sum().to_dict(),
                 duplicate_local_wall_times=int(local.tz_localize(None).duplicated().sum()),
                 local_wall_delta_counts={str(k): int(v) for k,v in pd.Series(local.tz_localize(None)).diff().value_counts().items()})
    identity = df.total_load - df.residual_load - df.onshore_wind - df.offshore_wind - df.photovoltaics
    audit['load_identity_max_abs_error_MWh'] = float(identity.abs().max())
    audit['load_identity_median_abs_error_MWh'] = float(identity.abs().median())
    audit['load_identity_exact_fraction'] = float((identity.abs()<1e-8).mean())
    audit['load_identity_within_1MWh_fraction'] = float((identity.abs()<=1).mean())
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        dictionary.to_csv(out/'column_dictionary.csv', index=False)
        dump({'id_to_english':dict(zip(dictionary.col_id,dictionary.english_name)),
              'original_to_model':dict(zip(dictionary.original_column,dictionary.model_column)),
              'model_to_original':dict(zip(dictionary.model_column,dictionary.original_column)),
              'metadata':records}, out/'column_dictionary.json')
        export = df.copy(); export.insert(0,'datetime_berlin',local)
        export.to_parquet(out/'cleaned_hourly.parquet')
        export.to_csv(out/'cleaned_hourly.csv.gz', compression='gzip')
        dump(audit, out/'data_quality.json')
    return df, dictionary, audit

def calendar_design(index):
    local = index.tz_convert('Europe/Berlin')
    elapsed = (index - pd.Timestamp('2015-01-01',tz='UTC')).total_seconds().to_numpy()/86400
    x = pd.DataFrame({'intercept':np.ones(len(index)), 'trend':elapsed/365.2425},index=index)
    groups = {'daily':[], 'weekly':[], 'annual':[]}
    for h in range(1,24):
        name=f'hour_{h}'; x[name]=(local.hour==h).astype(float); groups['daily'].append(name)
    for d in range(1,7):
        name=f'weekday_{d}'; x[name]=(local.dayofweek==d).astype(float); groups['weekly'].append(name)
    for k in range(1,5):
        for trig,fn in [('sin',np.sin),('cos',np.cos)]:
            name=f'annual_{trig}_{k}'; x[name]=fn(2*np.pi*k*elapsed/365.2425); groups['annual'].append(name)
    return x,groups

def vif_table(frame):
    a=frame.dropna(); names=list(a.columns); z=(a-a.mean())/a.std(ddof=0)
    rows=[]
    for i,c in enumerate(names):
        y=z.iloc[:,i].to_numpy(); x=np.delete(z.to_numpy(),i,axis=1)
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            v=np.inf
        else:
            r=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
            ratio=np.dot(r,r)/max(np.dot(y,y),1e-20)
            v=np.inf if ratio<1e-10 else 1/ratio
        rows.append({'variable':c,'VIF':v})
    return pd.DataFrame(rows)

def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def eda(df, out):
    out=Path(out); figs=out/'figures'; figs.mkdir(exist_ok=True)
    train=df.loc[df.index<pd.Timestamp('2020-01-01',tz='UTC')]
    summary=df.describe().T
    summary['missing']=df.isna().sum(); summary['zero_fraction']=(df==0).mean()
    summary['negative_count']=(df<0).sum(); summary.to_csv(out/'descriptive_statistics.csv')
    # Full-period plots are retrospective descriptive graphics, never model selection inputs.
    fig,axes=plt.subplots(5,3,figsize=(16,15),sharex=True)
    for c,ax in zip(df,axes.flat):
        df[c].tz_convert('Europe/Berlin').resample('MS').mean().plot(ax=ax,color='#177e89',lw=1.5)
        ax.set_title(c.replace('_',' ')); ax.set_ylabel('MWh (hourly)'); ax.set_xlabel('')
        ax.axvline(pd.Timestamp('2022-01-01',tz='UTC'),color='#b65a38',ls='--',alpha=.7)
    fig.suptitle('Monthly mean hourly energy, 2015–2022 | dashed line: final test starts',y=1.01)
    savefig(fig,figs/'01_series_overview.png')
    local=train.index.tz_convert('Europe/Berlin')
    fig,axes=plt.subplots(3,1,figsize=(14,14))
    for ax,key,title in zip(axes,[local.hour,local.dayofweek,local.month],['Hour of day','Day of week (Monday=0)','Month']):
        prof=train.groupby(key).mean().T
        prof=prof.sub(train.mean(),axis=0).div(train.std(),axis=0)
        im=ax.imshow(prof,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
        ax.set_yticks(range(len(train.columns)),[c.replace('_',' ') for c in train])
        ax.set_xticks(range(prof.shape[1]),prof.columns); ax.set_title(title+' | training period 2015–2019')
        fig.colorbar(im,ax=ax,label='Deviation from mean / standard deviation')
    savefig(fig,figs/'02_seasonal_profiles.png')
    x,groups=calendar_design(train.index); observed=train.notna().all(axis=1)
    x=x.loc[observed]; y=train.loc[observed]
    coef=np.linalg.lstsq(x,y,rcond=None)[0]; fitted=x.to_numpy()@coef
    residual=pd.DataFrame(y.to_numpy()-fitted,index=y.index,columns=y.columns)
    sse=np.square(residual).sum().to_numpy()
    fit=x.index<pd.Timestamp('2019-01-01',tz='UTC'); val=~fit
    full_hold=x.to_numpy()[val]@np.linalg.lstsq(x.to_numpy()[fit],y.to_numpy()[fit],rcond=None)[0]
    hold_sse=np.square(y.to_numpy()[val]-full_hold).sum(axis=0)
    records=[]
    for g,cols in groups.items():
        reduced=x.drop(columns=cols).to_numpy()
        reduced_sse=np.square(y.to_numpy()-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0]).sum(axis=0)
        hold_reduced=reduced[val]@np.linalg.lstsq(reduced[fit],y.to_numpy()[fit],rcond=None)[0]
        hold_reduced_sse=np.square(y.to_numpy()[val]-hold_reduced).sum(axis=0)
        for j,c in enumerate(y):
            strength=max(0,1-sse[j]/reduced_sse[j])
            benefit=1-hold_sse[j]/hold_reduced_sse[j]
            records.append(dict(variable=c,season=g,partial_R2=strength,validation_2019_relative_SSE_gain=benefit,
                                evidence='supported' if strength>=.02 and benefit>0 else ('in_sample_only' if strength>=.02 else 'weak')))
    season=pd.DataFrame(records); season.to_csv(out/'seasonality_by_variable.csv',index=False)
    residual.to_parquet(out/'calendar_adjusted_training.parquet')
    ac=[]
    for c in train:
        for lag in [1,24,168,8760]:
            ac.append(dict(variable=c,lag_hours=lag,raw_acf=train[c].autocorr(lag),
                           calendar_adjusted_acf=residual[c].autocorr(lag),diff24_acf=train[c].diff(24).autocorr(lag)))
    pd.DataFrame(ac).to_csv(out/'autocorrelation_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    for ax,col,title,lo in zip(axes,['partial_R2','validation_2019_relative_SSE_gain'],['Partial R², 2015–2019','Held-out 2019 SSE improvement'],[0,-.2]):
        mat=season.pivot(index='variable',columns='season',values=col)
        im=ax.imshow(mat,aspect='auto',cmap='RdYlGn',vmin=lo,vmax=.7)
        ax.set_yticks(range(len(mat)),[s.replace('_',' ') for s in mat.index]); ax.set_xticks(range(3),mat.columns)
        for i in range(len(mat)):
            for j in range(3): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
        ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.6)
    savefig(fig,figs/'03_seasonality_evidence.png')
    fig,axes=plt.subplots(1,2,figsize=(17,8))
    for ax,frame,title in zip(axes,[train,residual],['Raw correlation, 2015–2019','After calendar + trend adjustment']):
        corr=frame.corr(); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        names=[c.replace('_',' ') for c in frame]
        ax.set_xticks(range(len(names)),names,rotation=90,fontsize=8); ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_title(title)
        corr.to_csv(out/('raw_correlation.csv' if title.startswith('Raw') else 'adjusted_correlation.csv'))
    fig.colorbar(im,ax=axes[-1],shrink=.65); savefig(fig,figs/'04_correlations.png')
    vifs=vif_table(train).rename(columns={'VIF':'VIF_all_15_including_response'})
    vifs=vifs.merge(vif_table(train.drop(columns=TARGET)).rename(columns={'VIF':'VIF_predictors_only'}),on='variable',how='left')
    vifs=vifs.merge(vif_table(residual.drop(columns=TARGET)).rename(columns={'VIF':'VIF_adjusted_predictors'}),on='variable',how='left')
    vifs.to_csv(out/'vif_diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    s=train[TARGET]; s.loc['2019-02-01':'2019-02-14'].plot(ax=axes[0],label='Observed',color='#177e89')
    pd.Series(fitted[:,list(train).index(TARGET)],index=y.index).loc['2019-02-01':'2019-02-14'].plot(ax=axes[0],label='Calendar + trend fit',color='#b65a38')
    axes[0].set_title('Total load: fitted recurring calendar pattern'); axes[0].legend(); axes[0].set_ylabel('MWh')
    f,p=periodogram(s.to_numpy(),fs=1,detrend='linear'); sel=(f>1/(24*400))&(f<1/2)
    axes[1].loglog(1/f[sel],p[sel],color='#177e89')
    for h in [24,168,24*365.2425]:
        axes[1].axvline(h,ls='--',color='#b65a38')
        axes[1].text(h,.95,f'{h:.0f} h',rotation=90,va='top',transform=axes[1].get_xaxis_transform())
    axes[1].set_xlabel('Period (hours)'); axes[1].set_ylabel('Spectral power'); axes[1].set_title('Periodogram: descriptive corroboration, not a significance test')
    savefig(fig,figs/'05_load_seasonality.png')
    return season,vifs

def causal_fill(df):
    """Predictor histories only. No future values; residual NAs go to fold-local median."""
    filled=df.ffill(limit=2)
    for lag in [168,336,504,672,24]:
        filled=filled.fillna(df.shift(lag))
    return filled

def build_features(df, target=TARGET, origins=None):
    """Every row at t=o+h uses measurements strictly before o, for h=0,...,23.

    Row timestamps identify interval starts. y[o-1h] must be published by o.
    A pooled direct multi-horizon model outputs all 24 rows in one predict call.
    """
    if origins is None:
        days=pd.date_range(df.index.min().ceil('D'),df.index.max().floor('D'),freq='D')
        origins=days[(days>=df.index.min()+pd.Timedelta(hours=672)) & (days+pd.Timedelta(hours=23)<=df.index.max())]
    origin=origins.repeat(24)
    horizon=np.tile(np.arange(24),len(origins))
    ts=origin+pd.to_timedelta(horizon,unit='h')
    hist=causal_fill(df)
    x=pd.DataFrame(index=ts)
    def take(col,when): return hist[col].reindex(when).to_numpy()
    for lag in [24,48,168,336]: x[f'{target}_lag{lag}']=take(target,ts-pd.Timedelta(hours=lag))
    x[f'{target}_origin_last']=take(target,origin-pd.Timedelta(hours=1))
    for window in [24,168]:
        for stat in ['mean','std']:
            roll=getattr(hist[target].rolling(window,min_periods=max(2,window//2)),stat)()
            x[f'{target}_origin_{stat}{window}']=roll.reindex(origin-pd.Timedelta(hours=1)).to_numpy()
    # Residual load is a derived quantity; its historical information is redundant here.
    for c in df.columns:
        if c in [target,'residual_load']: continue
        for lag in [24,168]: x[f'{c}_lag{lag}']=take(c,ts-pd.Timedelta(hours=lag))
        x[f'{c}_origin_mean24']=hist[c].rolling(24,min_periods=12).mean().reindex(origin-pd.Timedelta(hours=1)).to_numpy()
    local=ts.tz_convert('Europe/Berlin')
    x['cal_horizon']=horizon+1
    for h in range(1,24): x[f'cal_hour_{h}']=(local.hour==h).astype(float)
    for d in range(1,7): x[f'cal_dow_{d}']=(local.dayofweek==d).astype(float)
    x['cal_weekend']=(local.dayofweek>=5).astype(float)
    elapsed=(ts-pd.Timestamp('2015-01-01',tz='UTC')).total_seconds().to_numpy()/86400
    for k in range(1,4):
        x[f'cal_year_sin{k}']=np.sin(2*np.pi*k*elapsed/365.2425)
        x[f'cal_year_cos{k}']=np.cos(2*np.pi*k*elapsed/365.2425)
    x['cal_trend']=elapsed/365.2425
    meta=pd.DataFrame({'origin_utc':origin,'target_utc':ts,'target_berlin':local,'horizon':horizon+1,
                       'actual':df[target].reindex(ts).to_numpy()},index=ts)
    assert ((ts-pd.Timedelta(hours=24))<origin).all()
    assert ((ts-pd.Timedelta(hours=168))<origin).all()
    return x.astype(float),meta

class CorrelationPruner(TransformerMixin,BaseEstimator):
    """Train-only greedy correlation filter; stable feature order favors target lags."""
    def __init__(self,threshold=.98): self.threshold=threshold
    def fit(self,X,y=None):
        a=np.asarray(X); var=np.nanstd(a,axis=0); keep=[]
        corr=np.nan_to_num(np.corrcoef(a,rowvar=False),nan=0)
        for j in range(a.shape[1]):
            if var[j]>1e-10 and all(abs(corr[j,k])<self.threshold for k in keep): keep.append(j)
        # Remove exact multivariate dependencies (e.g. weekend = Sat + Sun).
        independent=[]
        for j in keep:
            if not independent:
                independent.append(j); continue
            sub=corr[np.ix_(independent,independent)]; cross=corr[independent,j]
            unexplained=1-cross@np.linalg.lstsq(sub,cross,rcond=None)[0]
            if unexplained>1e-9: independent.append(j)
        self.keep_=np.array(independent); self.n_features_in_=a.shape[1]; return self
    def transform(self,X): return np.asarray(X)[:,self.keep_]

def candidates():
    specs=[dict(name='Naive_previous_day',kind='naive',lag=24),
           dict(name='Naive_previous_week',kind='naive',lag=168)]
    for alpha in [1.,100.,10000.]: specs.append(dict(name=f'Ridge_{alpha:g}',kind='ridge',alpha=alpha))
    for alpha in [1.,10.]: specs.append(dict(name=f'ElasticNet_{alpha:g}',kind='elastic',alpha=alpha))
    for leaves in [15,31]:
        specs.append(dict(name=f'Boost_all_{leaves}',kind='boost',leaves=leaves,subset='all',residual=False))
    specs.append(dict(name='Boost_load_only_15',kind='boost',leaves=15,subset='load',residual=False))
    specs.append(dict(name='Boost_load_only_31',kind='boost',leaves=31,subset='load',residual=False))
    specs.append(dict(name='Boost_selected_31',kind='boost',leaves=31,subset='all',residual=False,select=True))
    specs.append(dict(name='Boost_weekly_residual_15',kind='boost',leaves=15,subset='all',residual=True))
    return specs

def chosen_columns(x,spec):
    if spec.get('subset')=='load': return [c for c in x if c.startswith(TARGET+'_') or c.startswith('cal_')]
    return list(x.columns)

def fit_model(x,y,spec):
    if spec['kind']=='naive': return {'spec':spec,'model':None,'columns':list(x.columns)}
    cols=chosen_columns(x,spec); xx=x[cols]
    steps=[('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('pruner',CorrelationPruner())]
    if spec['kind']=='ridge':
        steps += [('scale',StandardScaler()),('regressor',Ridge(alpha=spec['alpha']))]
    elif spec['kind']=='elastic':
        steps += [('scale',StandardScaler()),('regressor',ElasticNet(alpha=spec['alpha'],l1_ratio=.7,max_iter=6000,tol=.001,selection='cyclic'))]
    else:
        if spec.get('select'):
            steps += [('scale',StandardScaler()),('selector',SelectFromModel(
                ElasticNet(alpha=1.,l1_ratio=.7,max_iter=6000,tol=.001),threshold='median'))]
        steps += [('regressor',HistGradientBoostingRegressor(max_iter=220,learning_rate=.06,max_leaf_nodes=spec['leaves'],
                       min_samples_leaf=40,l2_regularization=10.,early_stopping=False,random_state=SEED))]
    model=Pipeline(steps)
    yy=np.asarray(y)
    if spec.get('residual'): yy=yy-x[f'{TARGET}_lag168'].to_numpy()
    valid=np.isfinite(yy)
    model.fit(xx.loc[valid],yy[valid])
    return dict(spec=spec,model=model,columns=cols)

def predict(bundle,x):
    s=bundle['spec']
    if s['kind']=='naive': pred=x[f'{TARGET}_lag{s["lag"]}'].to_numpy()
    else:
        pred=bundle['model'].predict(x[bundle['columns']])
        if s.get('residual'): pred=pred+x[f'{TARGET}_lag168'].to_numpy()
    return np.maximum(0,pred)

def metrics(y,p,scale):
    y=np.asarray(y);p=np.asarray(p);valid=np.isfinite(y)&np.isfinite(p);y=y[valid];p=p[valid]
    return dict(n=len(y),MAE=mean_absolute_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),
                WAPE_pct=100*np.abs(y-p).sum()/np.abs(y).sum(),R2=r2_score(y,p),
                MASE_weekly=mean_absolute_error(y,p)/scale,bias=float((p-y).mean()))

def scale_before(df,cutoff):
    s=df.loc[df.index<pd.Timestamp(cutoff,tz='UTC'),TARGET]
    return float(s.diff(168).abs().mean())

def run_models(df,x,meta,out):
    out=Path(out); specs=candidates(); cv_rows=[]; oof={s['name']:[] for s in specs}
    folds=[('2020','2020-01-01','2021-01-01'),('2021_H1','2021-01-01','2021-07-01'),('2021_H2','2021-07-01','2022-01-01')]
    for fold,start,end in folds:
        tr=(meta.origin_utc<pd.Timestamp(start,tz='UTC')) & meta.actual.notna()
        va=(meta.origin_utc>=pd.Timestamp(start,tz='UTC'))&(meta.origin_utc<pd.Timestamp(end,tz='UTC'))
        scale=scale_before(df,start)
        for spec in specs:
            now=time.time(); b=fit_model(x.loc[tr],meta.loc[tr,'actual'],spec); p=predict(b,x.loc[va])
            row=dict(fold=fold,model=spec['name'],**metrics(meta.loc[va,'actual'],p,scale),seconds=time.time()-now)
            cv_rows.append(row)
            f=meta.loc[va].copy();f['prediction']=p; f['fold']=fold;oof[spec['name']].append(f)
            log(f"{fold} {spec['name']}: MAE {row['MAE']:.1f}, {row['seconds']:.1f}s")
    cv=pd.DataFrame(cv_rows);cv.to_csv(out/'cv_scores.csv',index=False)
    ranking=[]
    for spec in specs:
        subset=cv[cv.model==spec['name']]
        ranking.append(dict(model=spec['name'],weighted_CV_MAE=np.average(subset.MAE,weights=subset.n),
                            weighted_CV_RMSE=np.sqrt(np.average(subset.RMSE**2,weights=subset.n))))
    ranking=pd.DataFrame(ranking).sort_values('weighted_CV_MAE');ranking.to_csv(out/'cv_ranking.csv',index=False)
    winner=ranking.iloc[0].model; spec=next(s for s in specs if s['name']==winner)
    log(f'Winner locked before test: {winner}')
    dump({'winner':winner,'selection':'minimum observation-weighted 2020–2021 rolling-origin CV MAE','spec':spec},out/'selection_lock.json')
    train=(meta.origin_utc<pd.Timestamp('2022-01-01',tz='UTC'))&meta.actual.notna()
    test=meta.origin_utc>=pd.Timestamp('2022-01-01',tz='UTC')
    scale=scale_before(df,'2022-01-01'); results=[]; winner_bundle=None; pred_all={}
    # Evaluate each family representative chosen using CV, without changing the locked winner.
    finalists=[winner,'Naive_previous_day','Naive_previous_week']
    for family in ['Ridge_','ElasticNet_','Boost_all_','Boost_load_only_','Boost_selected_','Boost_weekly_residual_']:
        names=ranking.loc[ranking.model.str.startswith(family),'model']
        if len(names): finalists.append(names.iloc[0])
    for name in dict.fromkeys(finalists):
        s=next(s for s in specs if s['name']==name);b=fit_model(x.loc[train],meta.loc[train,'actual'],s)
        p=predict(b,x.loc[test]); pred_all[name]=p
        results.append(dict(model=name,selected_before_test=name==winner,**metrics(meta.loc[test,'actual'],p,scale)))
        if name==winner: winner_bundle=b
    scores=pd.DataFrame(results).sort_values('MAE');scores.to_csv(out/'test_scores.csv',index=False)
    all_oof=pd.concat(oof[winner]); all_oof.to_csv(out/'winner_cv_predictions.csv.gz',index=False,compression='gzip')
    # Empirical horizon-wise absolute-error bands: calibrated before 2022, not guaranteed under drift.
    cal=all_oof.loc[all_oof.origin_utc>=pd.Timestamp('2021-01-01',tz='UTC')].copy()
    cal['abs_error']=(cal.actual-cal.prediction).abs()
    radii=cal.groupby('horizon').abs_error.quantile(.9,interpolation='higher')
    pred=meta.loc[test].copy();pred['prediction']=pred_all[winner]
    pred['lower90']=np.maximum(0,pred.prediction-pred.horizon.map(radii));pred['upper90']=pred.prediction+pred.horizon.map(radii)
    pred['error']=pred.prediction-pred.actual
    pred.to_csv(out/'test_predictions_24h.csv',index=False)
    byh=[]
    for h,g in pred.groupby('horizon'):
        byh.append(dict(horizon=h,**metrics(g.actual,g.prediction,scale),coverage90=((g.actual>=g.lower90)&(g.actual<=g.upper90)).mean()))
    pd.DataFrame(byh).to_csv(out/'test_metrics_by_horizon.csv',index=False)
    monthly=[]
    for month,g in pred.groupby(pred.target_utc.dt.strftime('%Y-%m')): monthly.append(dict(month=month,**metrics(g.actual,g.prediction,scale)))
    pd.DataFrame(monthly).to_csv(out/'test_metrics_by_month.csv',index=False)
    winner_bundle.update(target=TARGET,radii90=radii.to_dict(),training_end='2021-12-31',forecast_contract='UTC midnight, next 24 hourly intervals; observations available strictly before origin')
    # Save via this importable module; see runner and notebook for load instructions.
    joblib.dump(winner_bundle,out/'evaluation_model.joblib')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    rankplot=ranking.sort_values('weighted_CV_MAE',ascending=False)
    axes[0].barh(rankplot.model,rankplot.weighted_CV_MAE,color='#177e89');axes[0].set_xlabel('CV MAE (MWh)');axes[0].set_title('Model selected using 2020–2021 only')
    axes[1].barh(scores.model[::-1],scores.MAE[::-1],color=['#b65a38' if n==winner else '#177e89' for n in scores.model[::-1]])
    axes[1].set_xlabel('2022 test MAE (MWh)');axes[1].set_title('Final test; orange = preselected model')
    savefig(fig,out/'figures/06_model_comparison.png')
    fig,axes=plt.subplots(3,1,figsize=(14,11))
    sample=pred.loc['2022-02-07':'2022-02-13']
    axes[0].plot(sample.index,sample.actual,label='Observed',color='#16324f');axes[0].plot(sample.index,sample.prediction,label='Forecast',color='#dc7b38')
    axes[0].fill_between(sample.index,sample.lower90,sample.upper90,alpha=.15,color='#dc7b38',label='Empirical 90% band')
    axes[0].legend(ncol=3);axes[0].set_ylabel('MWh');axes[0].set_title('One fixed illustrative week; all days use a single midnight forecast origin')
    axes[1].plot([r['horizon'] for r in byh],[r['MAE'] for r in byh],marker='o',color='#177e89');axes[1].set_xlabel('Forecast horizon (hours)');axes[1].set_ylabel('MAE (MWh)')
    axes[2].bar([r['month'] for r in monthly],[r['MAE'] for r in monthly],color='#177e89');axes[2].tick_params(axis='x',rotation=45);axes[2].set_ylabel('MAE (MWh)')
    savefig(fig,out/'figures/07_forecast_diagnostics.png')
    # Explanatory block permutation importance on last validation fold, never selection on test.
    foldtr=(meta.origin_utc<pd.Timestamp('2021-07-01',tz='UTC'))&meta.actual.notna()
    foldva=(meta.origin_utc>=pd.Timestamp('2021-07-01',tz='UTC'))&(meta.origin_utc<pd.Timestamp('2022-01-01',tz='UTC'))
    vb=fit_model(x.loc[foldtr],meta.loc[foldtr,'actual'],spec); vx=x.loc[foldva]; vy=meta.loc[foldva,'actual']
    base=mean_absolute_error(vy,predict(vb,vx)); rng=np.random.default_rng(SEED);imp=[]
    groups={c:[k for k in x if k.startswith(c+'_')] for c in df if c!='residual_load'}
    groups['calendar']=[c for c in x if c.startswith('cal_')]
    for group,cols in groups.items():
        cols=[c for c in cols if c in vb['columns']]
        if not cols:continue
        deltas=[]
        for rep in range(3):
            perm=rng.permutation(len(vx)//24); xp=vx.copy()
            xp[cols]=vx[cols].to_numpy().reshape(-1,24,len(cols))[perm].reshape(-1,len(cols))
            deltas.append(mean_absolute_error(vy,predict(vb,xp))-base)
        imp.append(dict(group=group,MAE_increase=np.mean(deltas),repeat_sd=np.std(deltas)))
    importance=pd.DataFrame(imp).sort_values('MAE_increase',ascending=False);importance.to_csv(out/'validation_group_importance.csv',index=False)
    if winner_bundle['model'] is not None:
        pipe=winner_bundle['model']; names=pipe['imputer'].get_feature_names_out(winner_bundle['columns'])
        kept=names[pipe['pruner'].keep_]
        if 'selector' in pipe.named_steps: kept=kept[pipe['selector'].get_support()]
        featuretable=pd.DataFrame({'feature':kept})
        if hasattr(pipe['regressor'],'coef_'):featuretable['standardized_coefficient']=pipe['regressor'].coef_
        featuretable.to_csv(out/'selected_features.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,6));vis=importance.sort_values('MAE_increase')
    ax.barh(vis.group,vis.MAE_increase,color='#177e89',xerr=vis.repeat_sd)
    ax.set_xlabel('Increase in validation MAE after shuffling complete 24-hour blocks (MWh)');ax.set_title('Grouped importance, 2021 H2 | predictive association, not causality')
    savefig(fig,out/'figures/08_predictor_importance.png')
    # Residual correlation reports remaining structure instead of claiming it was removed.
    residual_acf={str(lag):pred.error.autocorr(lag) for lag in [1,24,168]}
    overview=dict(winner=winner,spec=spec,cv_MAE=float(ranking.iloc[0].weighted_CV_MAE),
                  test_metrics=next(r for r in results if r['model']==winner),
                  interval_coverage90=float(((pred.actual>=pred.lower90)&(pred.actual<=pred.upper90)).mean()),
                  interval_mean_width=float((pred.upper90-pred.lower90).mean()),residual_acf=residual_acf,
                  test_start=str(pred.index.min()),test_end=str(pred.index.max()),test_days=pred.origin_utc.nunique(),
                  feature_count=len(x.columns),training_rows=int(train.sum()),final_test_rows=int(test.sum()))
    # Production artifact is refit only after the locked final test has been scored.
    full=meta.actual.notna();prod=fit_model(x.loc[full],meta.loc[full,'actual'],spec)
    prod.update(target=TARGET,radii90=radii.to_dict(),training_end=str(meta.loc[full].index.max()),forecast_contract=winner_bundle['forecast_contract'])
    joblib.dump(prod,out/'production_model.joblib')
    # Latest complete forecast origin, without inventing the missing final UTC hour.
    origin=df.index.max().floor('D')
    history=df.loc[df.index<origin]
    fx,fmeta=build_features(history,origins=pd.DatetimeIndex([origin]))
    fmeta['prediction']=predict(prod,fx)
    fmeta['lower90']=np.maximum(0,fmeta.prediction-fmeta.horizon.map(radii));fmeta['upper90']=fmeta.prediction+fmeta.horizon.map(radii)
    fmeta['note']='Historical endpoint demo; labels shown only where present in supplied file'
    fmeta['actual']=df[TARGET].reindex(fmeta.index).to_numpy()
    fmeta.to_csv(out/'endpoint_24h_forecast_demo.csv',index=False)
    dump(overview,out/'model_summary.json')
    return overview,ranking,scores,importance

def missingness_experiment(df,out):
    s=df.loc[(df.index>=pd.Timestamp('2017-01-01',tz='UTC'))&(df.index<pd.Timestamp('2020-01-01',tz='UTC')),TARGET]
    rng=np.random.default_rng(SEED);rows=[]
    for length in [1,3,24,168]:
        mask=np.zeros(len(s),dtype=bool)
        starts=rng.choice(np.arange(700,len(s)-length-1),size=30,replace=False)
        for start in starts:mask[start:start+length]=True
        corrupted=s.mask(mask)
        methods={'forward_fill':corrupted.ffill(),'previous_day':corrupted.shift(24),'previous_week':corrupted.shift(168),
                 'bounded_forward_then_seasonal':causal_fill(corrupted.to_frame())[TARGET]}
        for name,filled in methods.items():
            valid=mask&filled.notna().to_numpy()&s.notna().to_numpy()
            rows.append(dict(gap_hours=length,method=name,masked_hours=int(mask.sum()),coverage=float(valid.sum()/mask.sum()),
                             MAE=float((filled[valid]-s[valid]).abs().mean())))
    pd.DataFrame(rows).to_csv(Path(out)/'simulated_missingness_load.csv',index=False)

def verify_pipeline(df,x,meta,out):
    origin=pd.Timestamp('2021-06-01',tz='UTC')
    before=build_features(df,origins=pd.DatetimeIndex([origin]))[0]
    changed=df.copy(); changed.loc[changed.index>=origin]=999999.
    after=build_features(changed,origins=pd.DatetimeIndex([origin]))[0]
    pd.testing.assert_frame_equal(before,after)
    prefix=build_features(df.loc[df.index<origin],origins=pd.DatetimeIndex([origin]))[0]
    pd.testing.assert_frame_equal(before,prefix)
    assert not meta.index.duplicated().any()
    assert (meta.groupby('origin_utc').size()==24).all()
    assert (meta.target_utc < meta.origin_utc+pd.Timedelta(hours=24)).all()
    assert meta.actual.equals(df[TARGET].reindex(meta.index).rename('actual'))
    assert len(x)==len(meta)
    dump({'future_mutation_invariance':True,'history_only_inference_equivalence':True,
          'all_forecast_origins_have_24_outputs':True,'target_alignment':True,
          'no_duplicate_target_timestamps':True},Path(out)/'verification.json')

def additional_diagnostics(df,x,meta,out):
    out=Path(out)
    identity=df.total_load-df.residual_load-df.onshore_wind-df.offshore_wind-df.photovoltaics
    errors=df.loc[identity.abs()>1e-8,['total_load','residual_load','onshore_wind','offshore_wind','photovoltaics']].copy()
    errors['identity_discrepancy_MWh']=identity.loc[errors.index]
    errors.to_csv(out/'load_identity_discrepancies.csv')
    df.groupby(df.index.tz_convert('Europe/Berlin').year).mean().to_csv(out/'annual_mean_energy.csv')
    bundle=joblib.load(out/'evaluation_model.joblib')
    if bundle['model'] is not None:
        pipe=bundle['model']; tr=(meta.origin_utc<pd.Timestamp('2022-01-01',tz='UTC'))&meta.actual.notna()
        arr=pipe['imputer'].transform(x.loc[tr,bundle['columns']])
        names=pipe['imputer'].get_feature_names_out(bundle['columns'])
        selected=pipe['pruner'].keep_
        if 'selector' in pipe.named_steps:selected=selected[pipe['selector'].get_support()]
        rows=[]; diagnostics={}
        for label,indices in [('before_pruning',np.arange(arr.shape[1])),('after_selection',selected)]:
            a=arr[:,indices];std=a.std(axis=0);nonconstant=std>1e-10
            a=a[:,nonconstant];ns=names[indices][nonconstant]
            z=(a-a.mean(axis=0))/a.std(axis=0); gram=z.T@z/len(z)
            eig,vec=np.linalg.eigh(gram); singular=eig<1e-9
            nullpart=np.square(vec[:,singular]).sum(axis=1)
            diag=np.diag(np.linalg.pinv(gram,rcond=1e-9))
            vifs=np.where(nullpart>1e-8,np.inf,diag)
            rows.extend(dict(stage=label,feature=n,VIF=float(v)) for n,v in zip(ns,vifs))
            diagnostics[label]=dict(features=len(ns),matrix_rank=int((~singular).sum()),
                                     infinite_VIF=int(np.isinf(vifs).sum()),VIF_above_10=int((vifs>10).sum()),
                                     max_finite_VIF=float(vifs[np.isfinite(vifs)].max()) if np.isfinite(vifs).any() else None)
        pd.DataFrame(rows).to_csv(out/'model_feature_vif.csv',index=False)
        dump(diagnostics,out/'model_feature_collinearity.json')
    preds=pd.read_csv(out/'test_predictions_24h.csv')
    test=meta.origin_utc>=pd.Timestamp('2022-01-01',tz='UTC')
    a=np.abs(preds.actual-preds.prediction).to_numpy().reshape(-1,24).mean(axis=1)
    b=np.abs(meta.loc[test,'actual'].to_numpy()-x.loc[test,f'{TARGET}_lag168'].to_numpy()).reshape(-1,24).mean(axis=1)
    diff=b-a; rng=np.random.default_rng(SEED);block=7;draws=[]
    for _ in range(2000):
        starts=rng.integers(0,len(diff)-block+1,size=int(np.ceil(len(diff)/block)))
        sample=np.concatenate([diff[s:s+block] for s in starts])[:len(diff)]
        draws.append(sample.mean())
    dump({'benchmark':'previous week','mean_MAE_reduction_MWh':float(diff.mean()),
          'block_bootstrap_95pct_interval_MWh':np.quantile(draws,[.025,.975]).tolist(),
          'block_days':block,'replications':2000,
          'interpretation':'Conditional retrospective uncertainty; does not cover regime shifts or model fitting uncertainty.'},out/'benchmark_uncertainty.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--mapping',required=True)
    parser.add_argument('--out',default='results');args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        df,dictionary,audit=load_data(args.data,args.mapping,out);log(f'Loaded {df.shape}, audit={audit}')
        eda(df,out);log('EDA completed')
        x,meta=build_features(df);verify_pipeline(df,x,meta,out);log(f'Leakage checks passed, {x.shape}')
        missingness_experiment(df,out)
        run_models(df,x,meta,out)
        additional_diagnostics(df,x,meta,out)
    packages=['numpy','pandas','scipy','scikit-learn','statsmodels','matplotlib','pyarrow','joblib']
    versions={p:importlib.metadata.version(p) for p in packages}
    (out/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    dump({'seed':SEED,'versions':versions,'source_sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.data,args.mapping]},
          'calendar_timezone':'Europe/Berlin','forecast_timezone':'UTC','target':TARGET,
          'validation_folds':['2020','2021_H1','2021_H2'],'test_year':2022,
          'assumption':'Previous hourly values are available at the forecast origin; revise lags if source publication is delayed.'},out/'run_manifest.json')
    log('Completed')

if __name__=='__main__':
    # Import the module under its stable name so serialized custom transformers can be loaded.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from tennet_pipeline import main as imported_main
    imported_main()
