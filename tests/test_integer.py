import copy,itertools,json,math,random,struct,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from quantlab.integer import integer_audit,read_request
from quantlab.packed import decode
from quantlab.inspector import inspect


def nearest(value,scale):
    # Separate floor/remainder rounding reference, not round(Fraction).
    z=F(value)/F(scale);lo=z.numerator//z.denominator;rem=z-lo
    q=lo+int(rem>F(1,2)or(rem==F(1,2)and lo%2!=0))
    return max(-127,min(127,q))


class IntegerTests(unittest.TestCase):
    def test_all_products_and_fraction_decomposition(self):
        rnd=random.Random(2041);cases=0
        for i in range(120):
            n=1+i%12;m=2+i%7
            v={'weights':[[rnd.uniform(-20,20)for _ in range(n)]for _ in range(m)],'x':[rnd.uniform(-3,3)for _ in range(n)],'activation_range':[.5,1.,2.,4.][i%4]}
            for mode in ['per-channel','per-tensor']:
                r=integer_audit(v,mode,32 if i%2 else 64,32 if i%3 else 64);p=decode(bytes.fromhex(r['weight_payload_hex']))['packed'];sx=F(r['stored_activation_scale'])
                qx=[nearest(x,sx)for x in v['x']];self.assertEqual(r['activation_codes'],qx)
                src=[];wo=[];full=[]
                for j,row in enumerate(r['rows']):
                    sw=F(p['scales'][0 if mode=='per-tensor'else j]);wq=[sw*q for q in p['q'][j]]
                    a=sum(F(w)*F(x)for w,x in zip(v['weights'][j],v['x']));b=sum(w*F(x)for w,x in zip(wq,v['x']));c=sum(w*sx*q for w,q in zip(wq,qx))
                    self.assertEqual(F(row['source_score_exact']),a);self.assertEqual(F(row['weight_only_score_exact']),b);self.assertEqual(F(row['full_score_exact']),c)
                    self.assertEqual(F(row['weight_error_exact']),b-a);self.assertEqual(F(row['activation_error_exact']),c-b);self.assertEqual(F(row['total_error_exact']),c-a)
                    self.assertEqual(sum(map(F,row['weight_error_contributions_exact'])),b-a);self.assertEqual(sum(map(F,row['activation_error_contributions_exact'])),c-b)
                    self.assertEqual(row['products'],[a*b for a,b in zip(p['q'][j],qx)]);self.assertEqual(row['partial_accumulators'],list(itertools.accumulate(row['products'])))
                    src.append(a);wo.append(b);full.append(c)
                self.assertEqual(r['source_prediction_exact'],max(range(m),key=lambda j:src[j]));self.assertEqual(r['weight_only_prediction_exact'],max(range(m),key=lambda j:wo[j]));self.assertEqual(r['full_prediction_exact'],max(range(m),key=lambda j:full[j]));cases+=1
        self.assertEqual(cases,240)

    def test_exhaustive_activation_half_ties_and_clipping(self):
        # range127 -> stored scale1 exactly, no float division ambiguity.
        for x in [F(j,2)for j in range(-259,260)]:
            v={'weights':[[1.],[0.]],'x':[float(x)],'activation_range':127.}
            r=integer_audit(v);self.assertEqual(r['activation_codes'],[nearest(x,1)])
            self.assertEqual(r['clipped_coordinates'],[abs(x)>127])
        self.assertEqual(integer_audit({'weights':[[1.],[0.]],'x':[126.5],'activation_range':127})['activation_codes'],[126])

    def test_activation_changes_decision_without_weight_error(self):
        v={'weights':[[127.,0.],[0.,127.]],'x':[.4,.49],'activation_range':127}
        r=integer_audit(v);self.assertEqual(r['source_prediction_exact'],1);self.assertEqual(r['weight_only_prediction_exact'],1);self.assertEqual(r['full_prediction_exact'],0)
        self.assertTrue(all(F(row['weight_error_exact'])==0 for row in r['rows']))

    def test_all_zero_and_scale_underflow(self):
        v={'weights':[[0.,0.],[0.,0.]],'x':[0.,0.],'activation_range':1}
        r=integer_audit(v);self.assertEqual(r['activation_codes'],[0,0]);self.assertEqual(r['full_prediction_exact'],0)
        v['activation_range']=5e-324
        for bits in [32,64]:
            with self.assertRaises(ValueError):integer_audit(v,activation_bits=bits)

    def test_accumulator_bound_both_signs_and_all_prefixes(self):
        for sign in [-1,1]:
            v={'weights':[[127.]*128,[-127.]*128],'x':[sign*127.]*128,'activation_range':127}
            r=integer_audit(v);self.assertEqual(r['absolute_accumulator_bound'],2064512);self.assertEqual(r['rows'][0]['accumulator'],sign*2064512)
            self.assertTrue(all(abs(z)<=2064512 for row in r['rows']for z in row['partial_accumulators']))

    def test_weight_payload_and_old_report_unchanged(self):
        v={'weights':[[.1,.4],[-.3,.2]],'x':[1.,3.],'activation_range':1.}
        for mode,bits in itertools.product(['per-channel','per-tensor'],[32,64]):
            self.assertEqual(integer_audit(v,mode,bits)['weight_payload_hex'],inspect({'weights':v['weights'],'x':v['x']},mode,bits)['payload_hex'])
        before=copy.deepcopy(v);integer_audit(v);self.assertEqual(v,before)

    def test_strict_request_and_parse(self):
        base={'weights':[[1.],[0.]],'x':[1.],'activation_range':1}
        for r in [True,0,-1,float('inf'),'1',1000001]:
            with self.assertRaises(ValueError):integer_audit(dict(base,activation_range=r))
        for bits in [True,16,'32']:
            with self.assertRaises(ValueError):integer_audit(base,activation_bits=bits)
        for text in ['{"weights":[[1],[0]],"x":[1],"activation_range":1,"activation_range":2}', '{"weights":[[1],[0]],"x":[1],"activation_range":NaN}', '['*2000+'0'+']'*2000]:
            with self.assertRaises(ValueError):read_request(text)
        self.assertEqual(read_request(json.dumps(base)),base)

    def test_cli_complete_report_and_rejection(self):
        v={'weights':[[1.],[0.]],'x':[1.],'activation_range':1.}
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'input.json';p.write_text(json.dumps(v));args=[sys.executable,'-m','quantlab.cli','integer','--input',str(p)]
            r=subprocess.run(args,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),integer_audit(v))
            p.write_text('{}');r=subprocess.run(args,capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertNotIn('Traceback',r.stderr)
