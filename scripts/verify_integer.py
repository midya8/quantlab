"""Installed-package audit from packaged witnesses, optional actual NumPy int32."""
import importlib.resources as res,json
from fractions import Fraction as F
import quantlab.integer as mod
from quantlab.integer import integer_audit
try:import numpy as np
except ImportError:np=None
reports=products=0
for file in res.files('quantlab').joinpath('data').iterdir():
    if file.name.startswith('integer-')and file.name.endswith('.json'):
        for c in json.loads(file.read_text()):
            r=integer_audit(c['input'],c['mode'],c['scale_bits'],c['activation_bits']);assert r==c['report'];reports+=1
            for row in r['rows']:
                assert F(row['weight_error_exact'])+F(row['activation_error_exact'])==F(row['total_error_exact']);products+=len(row['products'])
            if np is not None:
                w=np.asarray([row['weight_codes']for row in r['rows']],dtype=np.int32);x=np.asarray(r['activation_codes'],dtype=np.int32)
                actual=w@x;assert actual.dtype==np.int32
                assert actual.tolist()==[row['accumulator']for row in r['rows']]
                assert np.cumsum(w*x,axis=1,dtype=np.int32).tolist()==[row['partial_accumulators']for row in r['rows']]
assert reports==72 and products==6848
print(json.dumps({'reports':reports,'products':products,'module_origin':mod.__file__,'numpy':None if np is None else np.__version__}))
