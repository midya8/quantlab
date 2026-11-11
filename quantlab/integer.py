"""Integer dot products with exact represented-real error decomposition."""
from fractions import Fraction as F
import json,math,struct
from .inspector import validate_input,inspect,ratio,winner,finite
from .packed import decode


def validate(value):
    if type(value)is not dict or set(value)!={'weights','x','activation_range'}:raise ValueError('expected weights,x,activation_range only')
    w,x=validate_input({'weights':value['weights'],'x':value['x']})
    r=finite(value['activation_range'])
    if r<=0:raise ValueError('activation_range must be positive')
    return w,x,r


def read_request(text):
    if type(text)is not str or len(text.encode())>200000:raise ValueError('input exceeds 200KB')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate key')
            d[k]=v
        return d
    try:
        v=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _:None)
        validate(v);return v
    except (json.JSONDecodeError,RecursionError,OverflowError)as e:raise ValueError('invalid JSON')from e


def integer_audit(value,mode='per-channel',scale_bits=64,activation_bits=64):
    w,x,r=validate(value)
    if type(activation_bits)is not int or activation_bits not in (32,64):raise ValueError('activation_bits must be32or64')
    source_scale=r/127
    try:s=struct.unpack('<f'if activation_bits==32 else'<d',struct.pack('<f'if activation_bits==32 else'<d',source_scale))[0]
    except (OverflowError,struct.error)as e:raise ValueError('activation scale not representable')from e
    if not math.isfinite(s)or s<=0:raise ValueError('activation scale loses positive representation')
    old=inspect({'weights':w,'x':x},mode,scale_bits);p=decode(bytes.fromhex(old['payload_hex']))['packed']
    fx=list(map(F,x));fs=F(s)
    codes=[max(-127,min(127,round(v/fs)))for v in fx]
    qx=[q*fs for q in codes]
    clipped=[abs(v)>127*fs for v in fx]
    rows=[];source=[];weight=[];full=[]
    for i,wi in enumerate(w):
        sw=F(p['scales'][0 if mode=='per-tensor'else i]);qw=p['q'][i];wr=[q*sw for q in qw]
        products=[a*b for a,b in zip(qw,codes)];partials=[];acc=0
        for v in products:acc+=v;partials.append(acc)
        a=sum(F(v)*z for v,z in zip(wi,fx));b=sum(v*z for v,z in zip(wr,fx));c=sw*fs*acc
        we=[(v-F(z))*xx for v,z,xx in zip(wr,wi,fx)]
        ae=[v*(z-xx)for v,z,xx in zip(wr,qx,fx)]
        assert sum(we)==b-a and sum(ae)==c-b and b-a+c-b==c-a
        runtime=(float(acc)*p['scales'][0 if mode=='per-tensor'else i])*s
        if not math.isfinite(runtime):raise ValueError('nonfinite rescaled score')
        rows.append({'weight_codes':qw,'products':products,'partial_accumulators':partials,'accumulator':acc,
            'weight_scale':float(sw),'runtime_rescaled_score':runtime,
            'source_score_exact':ratio(a),'weight_only_score_exact':ratio(b),'full_score_exact':ratio(c),
            'weight_error_exact':ratio(b-a),'activation_error_exact':ratio(c-b),'total_error_exact':ratio(c-a),
            'weight_error_contributions_exact':[ratio(v)for v in we],
            'activation_error_contributions_exact':[ratio(v)for v in ae]})
        source.append(a);weight.append(b);full.append(c)
    n=len(x);bound=n*127*127
    assert all(abs(v)<=bound<2**31 for row in rows for v in row['partial_accumulators'])
    return {'schema':'quantlab-integer/1','mode':mode,'scale_bits':scale_bits,'activation_bits':activation_bits,
            'activation_range':r,'source_activation_scale':source_scale,'stored_activation_scale':s,
            'activation_codes':codes,'activation_restored':[float(v)for v in qx],
            'activation_restored_exact':[ratio(v)for v in qx],'clipped_coordinates':clipped,
            'features':n,'products_per_row':n,'absolute_accumulator_bound':bound,'signed_int32_max':2**31-1,
            'all_partial_accumulators_fit_int32':True,'rows':rows,'weight_payload_hex':old['payload_hex'],
            'source_prediction_exact':winner(source),'weight_only_prediction_exact':winner(weight),'full_prediction_exact':winner(full),
            'runtime_prediction':winner([row['runtime_rescaled_score']for row in rows]),
            'scope':'exact integer products/partial sums; exact dyadic represented-real scores; final rescale uses binary64, not integer-only end-to-end inference, hardware speed or accuracy guarantee'}
