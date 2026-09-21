"""Run as python -m intent_gate.train. Model/policy selection never uses test."""
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from intent_gate.data import ROOT, load_data, audit
from intent_gate.modeling import build_models, score, evaluate
from intent_gate.policy import tune_threshold


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def run():
    output = ROOT/'outputs'
    figures = output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True)
    data = load_data()
    report = audit(data)
    dump(output/'data_audit.json',report)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(data.keys(),[len(v) for v in data.values()],color='#377eb8')
    axes[0].tick_params(axis='x',rotation=35)
    axes[0].set(title='Official split sizes',ylabel='Utterances')
    for split in ['train','val','test','oos_test']:
        axes[1].hist([len(t.split()) for t,_ in data[split]],bins=range(0,35,2),alpha=.45,density=True,label=split)
    axes[1].set(title='Utterance lengths',xlabel='Whitespace-separated words',ylabel='Density')
    axes[1].legend(); fig.tight_layout(); fig.savefig(figures/'eda.png',dpi=150); plt.close(fig)
    train_x, train_y = map(list,zip(*data['train']))
    val_x, val_y = map(list,zip(*data['val']))
    results, fitted = {}, {}
    # Stage 1: fit and lock every policy, then select a model using validation only.
    with threadpool_limits(limits=2):
        for name,model in build_models().items():
            print(f'Training {name}',flush=True)
            start=time.perf_counter(); model.fit(train_x,train_y)
            fit_seconds=time.perf_counter()-start
            pred, ins = score(model,val_x)
            _, oos = score(model,[t for t,_ in data['oos_val']])
            policy=tune_threshold(ins,pred==np.array(val_y),oos,max_far=.05)
            results[name]={'fit_seconds':fit_seconds,'policy':policy,
                           'validation':evaluate(val_y,pred,ins,oos,policy['threshold'])}
            fitted[name]=model
        selected=max(results,key=lambda n:(results[n]['policy']['correct_route_rate'],results[n]['validation']['intent_accuracy']))
        dump(output/'selection.json',{'selected_model':selected,'selection_split':'val + oos_val',
            'objective':'Maximize validation correctly routed in-scope fraction subject to empirical OOS false accepts <= 5%.',
            'threshold_acceptance':'score >= threshold', 'models':results})
        # Stage 2: final held-out reporting. Nothing below changes model or threshold.
        all_predictions=[]
        curves=[]
        for name,model in fitted.items():
            x,y=map(list,zip(*data['test']))
            start=time.perf_counter(); pred,ins=score(model,x)
            seconds=time.perf_counter()-start
            opred,oos=score(model,[t for t,_ in data['oos_test']])
            threshold=results[name]['policy']['threshold']
            results[name]['test']=evaluate(y,pred,ins,oos,threshold)
            results[name]['test']['batch_ms_per_utterance']=seconds/len(x)*1000
            train_texts={' '.join(t.casefold().split()) for t,_ in data['train']}
            clean=np.array([' '.join(t.casefold().split()) not in train_texts for t in x])
            results[name]['test']['nonoverlap_intent_accuracy']=float(np.mean(pred[clean]==np.array(y)[clean]))
            results[name]['test']['nonoverlap_test_rows']=int(clean.sum())
            for split,texts,labels,preds,scores in [('test',x,y,pred,ins),('oos_test',[t for t,_ in data['oos_test']],['oos']*len(oos),opred,oos)]:
                for text,label,prediction,confidence in zip(texts,labels,preds,scores):
                    all_predictions.append({'model':name,'split':split,'text':text,'true_label':label,
                        'predicted_label':prediction,'confidence':float(confidence),'accepted':bool(confidence>=threshold)})
            fpr,tpr,_=roc_curve(np.r_[np.zeros(len(ins)),np.ones(len(oos))],1-np.r_[ins,oos])
            curves.append((name,fpr,tpr))
            joblib.dump({'model':model,'threshold':threshold,'name':name},ROOT/f'models/{name}.joblib')
            if name==selected:
                risk=[]
                for t in np.linspace(0,1,101):
                    a=ins>=t
                    risk.append({'threshold':float(t),'coverage':float(a.mean()),'selective_accuracy':float((pred[a]==np.array(y)[a]).mean()) if a.any() else None,'oos_far':float((oos>=t).mean())})
                pd.DataFrame(risk).to_csv(output/'diagnostic_curve.csv',index=False)
    joblib.dump({'model':fitted[selected],'threshold':results[selected]['policy']['threshold'],'name':selected},ROOT/'models/router.joblib')
    dump(output/'metrics.json',{'selected_model':selected,'seed':42,'python':platform.python_version(),
          'sklearn':sklearn.__version__,'numpy':np.__version__,'models':results})
    predictions=pd.DataFrame(all_predictions)
    # Excerpts are attributed CLINC150 derivatives; raw corpus remains ignored.
    errors=predictions[(predictions.model==selected)&predictions.accepted&(predictions.true_label!=predictions.predicted_label)]
    errors.sort_values('confidence',ascending=False).head(50).to_csv(output/'high_confidence_errors.csv',index=False)
    confusions=predictions[(predictions.model==selected)&(predictions.split=='test')&(predictions.true_label!=predictions.predicted_label)]
    confusions.groupby(['true_label','predicted_label']).size().sort_values(ascending=False).head(20).to_csv(output/'confusion_pairs.csv',header=['count'])
    fig,ax=plt.subplots(figsize=(6,5))
    for name,fpr,tpr in curves: ax.plot(fpr,tpr,label=f"{name}: {results[name]['test']['oos_auroc']:.3f}")
    ax.plot([0,1],[0,1],'k--',alpha=.3); ax.set(xlabel='In-scope rejection rate',ylabel='OOS rejection rate',title='OOS detection: held-out test'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'oos_roc.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    pd.DataFrame({n:{k:r['test'][k] for k in ['intent_accuracy','in_scope_coverage','oos_false_accept_rate']} for n,r in results.items()}).T.plot.bar(ax=axes[0],ylim=(0,1))
    axes[0].tick_params(axis='x',rotation=20); axes[0].set_title('Locked validation policies on test')
    curve=pd.read_csv(output/'diagnostic_curve.csv')
    axes[1].plot(curve.coverage,curve.selective_accuracy); axes[1].set(xlabel='In-scope coverage',ylabel='Selective accuracy',title=f'{selected}: diagnostic only')
    fig.tight_layout(); fig.savefig(figures/'tradeoffs.png',dpi=150); plt.close(fig)
    print(json.dumps({'selected_model':selected,'results':results},indent=2),flush=True)
    return results


if __name__ == '__main__':
    run()
