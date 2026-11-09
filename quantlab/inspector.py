"""Independent artifact-level inference audits; original training stays untouched."""
from __future__ import annotations
from fractions import Fraction
import json
import math
from .packed import encode, decode
from .quantize import quantize_tensor, quantize_per_channel


def finite(v):
    if type(v) not in (int, float) or abs(v) > 1e6 or not math.isfinite(v):
        raise ValueError('expected finite nonboolean number with magnitude <= 1e6')
    return float(v)


def validate_input(value):
    if type(value) is not dict or set(value) != {'weights', 'x'}:
        raise ValueError('expected weights and x only')
    w, x = value['weights'], value['x']
    if type(w) is not list or not 2 <= len(w) <= 8 or type(x) is not list or not 1 <= len(x) <= 128:
        raise ValueError('require 2..8 classes and 1..128 features')
    if any(type(row) is not list or len(row) != len(x) for row in w):
        raise ValueError('weights must be rectangular with one column per feature')
    return [[finite(v) for v in row] for row in w], [finite(v) for v in x]


def strict_json(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 200000:
        raise ValueError('input exceeds 200KB')
    depth = 0
    quoted = escape = False
    for c in text:
        if quoted:
            if escape: escape = False
            elif c == '\\': escape = True
            elif c == '"': quoted = False
        elif c == '"': quoted = True
        elif c in '[{':
            depth += 1
            if depth > 8: raise ValueError('input nesting exceeds 8')
        elif c in ']}': depth -= 1
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise ValueError('duplicate key')
            d[k] = v
        return d
    def bad(_): raise ValueError('nonfinite literal')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
        validate_input(value)
        return value
    except (json.JSONDecodeError, OverflowError, RecursionError) as e:
        raise ValueError('invalid bounded JSON input') from e


def winner(scores):
    return max(range(len(scores)), key=lambda i: scores[i])


def ratio(q):
    return str(q.numerator) + '/' + str(q.denominator)


def exact_audit(weights, x, packed):
    """Exact rational arithmetic over binary64 inputs and stored IEEE scales.

    This checks the represented real linear maps, NOT floating-point summation
    or uncertain inputs. Pairwise residual bounds are sufficient, not necessary.
    """
    wx = [Fraction(v) for v in x]
    w = [[Fraction(v) for v in row] for row in weights]
    restored = [[Fraction(q) * Fraction(packed['scales'][0 if packed['mode'] == 'per-tensor' else i])
                 for q in row] for i, row in enumerate(packed['q'])]
    scores = [sum(a * b for a, b in zip(row, wx)) for row in w]
    qscores = [sum(a * b for a, b in zip(row, wx)) for row in restored]
    top = winner(scores)
    pairs = []
    for j in range(len(w)):
        if j == top: continue
        margin = scores[top] - scores[j]
        bound = sum(abs((a - b) - (c - d)) * abs(v) for a,b,c,d,v in zip(w[top],w[j],restored[top],restored[j],wx))
        lower = margin - bound
        safe = lower >= 0 if top < j else lower > 0
        pairs.append({'competitor':j, 'margin':float(margin), 'residual_bound':float(bound),
                      'lower_bound':float(lower), 'margin_exact':ratio(margin),
                      'bound_exact':ratio(bound), 'lower_exact':ratio(lower), 'sufficient':safe})
    return {'source_prediction':top, 'restored_prediction':winner(qscores),
            'source_scores_exact':[ratio(v) for v in scores],
            'restored_scores_exact':[ratio(v) for v in qscores],
            'pairs':pairs, 'sufficient_same_prediction':all(p['sufficient'] for p in pairs),
            'scope':'Exact dyadic linear maps for this fixed x; does not certify floating-point accumulation, unseen inputs, accuracy or deployment.'}


def sequential_sum(values):
    # Python 3.12 changed built-in float sum to compensated accumulation.
    # Explicit left-to-right binary64 matches scalar JavaScript on all versions.
    out = 0.0
    for value in values:
        out += value
    return out


def inspect(value, mode='per-channel', scale_bits=64):
    weights, x = validate_input(value)
    if mode not in ('per-tensor','per-channel'):
        raise ValueError('unknown mode')
    quantizer = quantize_tensor if mode == 'per-tensor' else quantize_per_channel
    try:
        original = quantizer(weights)
    except (ZeroDivisionError, OverflowError) as e:
        raise ValueError("scale cannot be represented as positive finite binary64") from e
    blob = encode(original, scale_bits)
    artifact = decode(blob)
    p = artifact['packed']
    restored = [[q * p['scales'][0 if mode == 'per-tensor' else i] for q in row] for i,row in enumerate(p['q'])]
    source_scores = [sequential_sum(a*b for a,b in zip(row,x)) for row in weights]
    restored_scores = [sequential_sum(a*b for a,b in zip(row,x)) for row in restored]
    rows = []
    for i,(w,r,q) in enumerate(zip(weights,restored,p['q'])):
        si = 0 if mode == 'per-tensor' else i
        err = [b-a for a,b in zip(w,r)]
        # Display diagnostic only; exact sufficient conditions are separate.
        rows.append({'source':w,'q':q,'restored':r,'error':err,
                     'source_scale':original['scales'][si],'stored_scale':p['scales'][si],
                     'max_abs_error':max(map(abs,err)),
                     'absolute_logit_bound':sequential_sum(abs(e)*abs(v) for e,v in zip(err,x)),
                     'contributions':[e*v for e,v in zip(err,x)]})
    n = len(x)*len(weights)
    return {'mode':mode,'scale_bits':scale_bits,'x':x,'rows':rows,
            'source_scores':source_scores,'restored_scores':restored_scores,
            'score_errors':[b-a for a,b in zip(source_scores,restored_scores)],
            'source_prediction':winner(source_scores),'restored_prediction':winner(restored_scores),
            'exact':exact_audit(weights,x,p), 'layout':artifact['layout'],
            'payload_hex':blob.hex(), 'payload_fp32_ratio':4*n/(artifact['layout']['scale_bytes']+n),
            'container_fp32_ratio':4*n/len(blob), 'conceptual_fp32_weights_bytes':4*n,
            'source_binary64_weights_bytes':8*n,
            'limits':'Weights only, no bias/labels/features/activations. Binary64 source and inference. fp32 is a conceptual raw storage comparator, not the original arithmetic. No integer matmul, RAM or speed claim.'}
