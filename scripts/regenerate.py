"""Regenerate original fixtures plus whole-report cross-runtime witnesses."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.cli import _train_fixture
from quantlab.model import featurize
from quantlab.analysis import imbalanced_matrix, compare_modes
from quantlab.inspector import inspect
from quantlab.quantize import error_stats, dequantize, size_bytes
clf, train, test = _train_fixture(0,64,5)
original={}
for mode in ('per-tensor','per-channel'):
    q=clf.quantized(mode)
    original[mode]={'accuracy':q.accuracy(test),'source_accuracy':clf.accuracy(test),
                  'size':size_bytes(clf.weights,q.packed),'error':error_stats(clf.weights,dequantize(q.packed))}
original['magnitude_study']=compare_modes(imbalanced_matrix())
fixture={'weights':clf.weights,'topics':['food','space','sport'],
         'train':[dict(s,x=featurize(s['words'],64)) for s in train],
         'test':[dict(s,x=featurize(s['words'],64)) for s in test],
         'study':imbalanced_matrix(),'original':original}
controls=[{'name':'tie-collapse','value':{'weights':[[127,.5],[-127,-.5]],'x':[0,1]}},
          {'name':'prediction-flip','value':{'weights':[[127,-.5],[-127,.5]],'x':[0,1]}},
          {'name':'zero','value':{'weights':[[0,0],[0,0]],'x':[1,-2]}},
          {'name':'scale-rounding','value':{'weights':[[.1,.4],[-.3,.2]],'x':[1,3]}},
          {'name':'cancellation','value':{'weights':[[1e6,1e-9,-1e6],[1e6,0,-1e6]],'x':[1,1,1]}},
          {'name':'max-shape','value':{'weights':[[((i*13+j*7)%21-10)/3 for j in range(128)] for i in range(8)],'x':[(-1)**j*(j%4) for j in range(128)]}}]
for i in (0,1,2,19,39,59):
    controls.append({'name':f'original-test-{i}','value':{'weights':clf.weights,'x':fixture['test'][i]['x']}})
controls.append({'name':'original-row-study','value':{'weights':fixture['study'],'x':[1 if j%3==0 else -.25 for j in range(64)]}})
witnesses=[]
for control in controls:
    for mode in ('per-tensor','per-channel'):
        for bits in (32,64):
            witnesses.append({**control,'mode':mode,'scale_bits':bits,'report':inspect(control['value'],mode,bits)})
root=Path(__file__).resolve().parents[1]
for name,value in [('fixture',fixture)]+[(f'witnesses-{mode}-{bits}',[w for w in witnesses if w['mode']==mode and w['scale_bits']==bits]) for mode in ('per-tensor','per-channel') for bits in (32,64)]:
    (root/'quantlab/data'/f'{name}.json').write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
(root/'docs/data.js').write_text('window.QUANT_DATA='+json.dumps(fixture,separators=(',',':'),allow_nan=False)+';\n')
print('witnesses',len(witnesses))
