"""Forecast 24 hourly intervals at a single UTC-midnight origin using saved model."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tennet_pipeline import load_data, build_features, predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='results/production_model.joblib')
    p.add_argument('--data',required=True)
    p.add_argument('--mapping',required=True)
    p.add_argument('--origin',help='UTC midnight, e.g. 2022-12-31T00:00:00Z. Default: latest supported midnight.')
    p.add_argument('--out',default='forecast_24h.csv')
    a=p.parse_args()
    df,_,_=load_data(a.data,a.mapping)
    model=joblib.load(a.model)
    origin=pd.Timestamp(a.origin) if a.origin else (df.index.max()+pd.Timedelta(hours=1)).floor('D')
    origin=origin.tz_localize('UTC') if origin.tzinfo is None else origin.tz_convert('UTC')
    if origin!=origin.floor('D'): raise ValueError('This trained pipeline uses UTC-midnight forecast origins.')
    train_end=pd.Timestamp(model['training_end'])
    train_end=train_end.tz_localize('UTC') if train_end.tzinfo is None else train_end.tz_convert('UTC')
    if train_end>=origin: raise ValueError('The model was trained on or after this origin; use an earlier-trained model.')
    history=df.loc[df.index<origin]
    if len(history)<672 or history.index.max()!=origin-pd.Timedelta(hours=1):
        raise ValueError('Supply at least 28 days of history ending one hour before origin. Do not invent missing endpoint hours.')
    x,meta=build_features(history,origins=pd.DatetimeIndex([origin]))
    meta=meta.drop(columns='actual')
    meta['prediction']=predict(model,x)
    radius=meta.horizon.map(model['radii90'])
    meta['lower90']=np.maximum(0,meta.prediction-radius)
    meta['upper90']=meta.prediction+radius
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    meta.to_csv(a.out,index=False)
    print(meta.to_string(index=False))
    print(f'Saved {len(meta)} hourly forecasts to {a.out}')

if __name__=='__main__':main()
