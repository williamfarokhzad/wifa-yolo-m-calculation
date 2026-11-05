"""Trace a simplified 101-point bounding-box AP calculation."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

THRESHOLDS=tuple(Fraction(i,100) for i in range(50,96,5))


def validate(data):
    if not isinstance(data,dict) or set(data)!={'truth','predictions'}:
        raise ValueError('expected truth and predictions lists')
    for group in ('truth','predictions'):
        rows=data[group]
        if not isinstance(rows,list) or len(rows)>1000:
            raise ValueError('each list may contain at most 1000 rows')
        fields={'image','label','box'} | ({'score'} if group=='predictions' else set())
        for r in rows:
            if not isinstance(r,dict) or set(r)!=fields:
                raise ValueError(f'{group} rows require {sorted(fields)}')
            if any(not isinstance(r[k],str) or not r[k].strip() for k in ('image','label')):
                raise ValueError('image and label must be nonempty strings')
            b=r['box']
            if (not isinstance(b,list) or len(b)!=4 or any(type(v) is not int or abs(v)>10**9 for v in b)
                or not (b[0]<b[2] and b[1]<b[3])):
                raise ValueError('use integer xyxy boxes with positive area')
            if group=='predictions':
                q=r['score']
                if type(q) not in (int,float) or not 0<=q<=1 or not math.isfinite(q):
                    raise ValueError('score must be finite and in [0,1]')
    if not data['truth']:
        raise ValueError('need at least one labeled object')


def overlap(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    common=w*h
    return Fraction(common,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-common)


def class_ap(truth,predictions,label,threshold):
    gt=[r for r in truth if r['label']==label]
    if not gt:
        raise ValueError('AP needs labeled objects for the class')
    by_image={}
    for i,r in enumerate(gt):
        by_image.setdefault(r['image'],[]).append((i,r['box']))
    p=sorted((r for r in predictions if r['label']==label),key=lambda r:-r['score'])
    matched=set()
    curve=[]
    for rank,r in enumerate(p,1):
        options=[(overlap(r['box'],b),-i) for i,b in by_image.get(r['image'],[]) if i not in matched]
        best=max(options,default=(Fraction(0),0))
        hit=bool(options) and best[0]>=threshold
        if hit:
            matched.add(-best[1])
        curve.append((Fraction(len(matched),len(gt)),Fraction(len(matched),rank),hit))
    ap=sum((max((p for r,p,_ in curve if r>=Fraction(i,100)),default=Fraction(0))
            for i in range(101)),Fraction(0))/101
    return ap,curve


def report(data):
    validate(data)
    labels=sorted({r['label'] for r in data['truth']})
    by_threshold={}
    details={}
    for t in THRESHOLDS:
        values={c:class_ap(data['truth'],data['predictions'],c,t)[0] for c in labels}
        key=f'{float(t):.2f}'
        by_threshold[key]=sum(values.values())/len(labels)
        details[key]={c:float(v) for c,v in values.items()}
    return {'map50':float(by_threshold['0.50']),'map75':float(by_threshold['0.75']),
            'map50_95':float(sum(by_threshold.values())/len(THRESHOLDS)),
            'ap_by_class_and_iou':details,
            'excluded_labels':sorted({r['label'] for r in data['predictions']}-set(labels))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('example',type=Path)
    a=p.parse_args()
    try:
        print(json.dumps(report(json.loads(a.example.read_text(encoding='utf-8'))),indent=2,allow_nan=False))
    except (OSError,ValueError) as exc:
        p.error(str(exc))


if __name__=='__main__':
    main()
