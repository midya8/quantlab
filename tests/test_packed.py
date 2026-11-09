import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from quantlab.packed import encode,decode,HEADER,MAX_BYTES
from quantlab.inspector import inspect,strict_json
from quantlab.quantize import quantize_per_channel,quantize_tensor

class PackedTests(unittest.TestCase):
    def test_roundtrip_all_bytes_both_modes_precisions(self):
        w=[list(range(-127,1)),list(range(0,128))]
        for fn in (quantize_tensor,quantize_per_channel):
            for bits in (32,64):
                p=fn(w); b=encode(p,bits); d=decode(b)
                self.assertEqual(d['packed']['q'],p['q'])
                self.assertEqual(d['layout']['total_bytes'],24+len(p['scales'])*(bits//8)+256+4)
                self.assertEqual(encode(d['packed'],bits),b)
                self.assertEqual(b[:4],b'QLB1')
    def test_every_truncation_and_trailing(self):
        b=encode(quantize_tensor([[1,2],[3,4]]))
        for i in range(len(b)):
            with self.assertRaises(ValueError):decode(b[:i])
        for tail in (b'\0',b,bytes(MAX_BYTES)):
            with self.assertRaises(ValueError):decode(b+tail)
    def test_every_single_byte_corruption(self):
        b=encode(quantize_per_channel([[1,2],[3,4]]))
        for i in range(len(b)):
            a=bytearray(b); a[i]^=1
            with self.assertRaises(ValueError):decode(a)
    def test_authenticated_looking_crc_not_authentication(self):
        b=bytearray(encode(quantize_tensor([[1,2],[3,4]])))
        b[-5]=128;b[-4:]=struct.pack('<I',zlib.crc32(b[:-4]))
        with self.assertRaises(ValueError):decode(b)
        b[-5]=42;b[-4:]=struct.pack('<I',zlib.crc32(b[:-4]))
        self.assertEqual(decode(b)['packed']['q'][-1][-1],42)
    def test_malformed_headers_with_recomputed_crc(self):
        b=encode(quantize_tensor([[1,2],[3,4]]))
        fields=list(HEADER.unpack_from(b))
        for i,values in ((1,[0,2]),(2,[2,255]),(3,[0,1,16]),(4,[1]),(5,[0,1,9,2**32-1]),(6,[0,129]),(7,[0,5]),(8,[0,2])):
            for value in values:
                f=fields[:];f[i]=value;body=HEADER.pack(*f)+b[24:-4]
                with self.assertRaises(ValueError):decode(body+struct.pack('<I',zlib.crc32(body)))
    def test_scale_validation_underflow_and_float32_rounding(self):
        p={'mode':'per-tensor','q':[[1],[2]],'scales':[.1]}
        self.assertNotEqual(decode(encode(p,32))['packed']['scales'][0],.1)
        self.assertEqual(decode(encode(p,64))['packed']['scales'][0],.1)
        for s in (0,-1,float('nan'),float('inf'),True,1e7):
            p['scales']=[s]
            with self.assertRaises(ValueError):encode(p)
        p['scales']=[1e-100]
        with self.assertRaises(ValueError):encode(p,32)
    def test_strict_input_shapes_numbers_and_json(self):
        good={'weights':[[1,2],[3,4]],'x':[1,1]}
        for bad in ({**good,'extra':1},{**good,'x':[True,1]}, {**good,'weights':[[1],[2]]}, {**good,'x':[1e7,1]}, {**good,'x':[float('nan'),1]}, {**good,'x':[10**500,1]}):
            with self.assertRaises(ValueError):inspect(bad)
        for text in ('{"weights":[],"weights":[],"x":[]}', '{"weights":[[1],[2]],"x":[1e999]}','{"weights":[[1],[2]],"x":[NaN]}','['*10+'0'+']'*10,'{} trailing',' '*200001):
            with self.assertRaises(ValueError):strict_json(text)
    def test_exact_tie_direction_and_counterexample(self):
        retain=inspect({'weights':[[127,.5],[-127,-.5]],'x':[0,1]})
        flip=inspect({'weights':[[127,-.5],[-127,.5]],'x':[0,1]})
        self.assertTrue(retain['exact']['sufficient_same_prediction'])
        self.assertFalse(flip['exact']['sufficient_same_prediction'])
        self.assertNotEqual(flip['exact']['source_prediction'],flip['exact']['restored_prediction'])
    def test_random_sufficient_conditions_independent_fraction_oracle(self):
        rng=random.Random(99)
        for _ in range(100):
            w=[[rng.uniform(-10,10) for j in range(6)] for i in range(3)]
            x=[rng.uniform(-3,3) for j in range(6)]
            for mode in ('per-channel','per-tensor'):
                r=inspect({'weights':w,'x':x},mode,32)
                p=decode(bytes.fromhex(r['payload_hex']))['packed']
                original=[sum(Fraction(a)*Fraction(b) for a,b in zip(row,x)) for row in w]
                restored=[sum(Fraction(q)*Fraction(p['scales'][0 if mode=='per-tensor' else i])*Fraction(v) for q,v in zip(row,x)) for i,row in enumerate(p['q'])]
                a=max(range(3),key=lambda i:original[i]);b=max(range(3),key=lambda i:restored[i])
                self.assertEqual((a,b),(r['exact']['source_prediction'],r['exact']['restored_prediction']))
                if r['exact']['sufficient_same_prediction']:self.assertEqual(a,b)
    def test_full_frozen_reports(self):
        for w in [w for p in sorted((Path(__file__).parents[1]/'quantlab/data').glob('witnesses-*.json')) for w in json.loads(p.read_text())]:
            self.assertEqual(inspect(w['value'],w['mode'],w['scale_bits']),w['report'])
    def test_original_sources_and_complete_fixture_unchanged(self):
        import hashlib
        from quantlab.cli import _train_fixture
        from quantlab.model import featurize
        from quantlab.analysis import imbalanced_matrix
        root=Path(__file__).parents[1]
        hashes=json.loads((root/'quantlab/data/original-hashes.json').read_text())
        for p,digest in hashes.items():
            self.assertEqual(hashlib.sha256((root/p).read_bytes()).hexdigest(),digest)
        fixture=json.loads((root/'quantlab/data/fixture.json').read_text())
        clf,train,test=_train_fixture(0,64,5)
        self.assertEqual(clf.weights,fixture['weights'])
        self.assertEqual([dict(s,x=featurize(s['words'],64)) for s in train],fixture['train'])
        self.assertEqual([dict(s,x=featurize(s['words'],64)) for s in test],fixture['test'])
        self.assertEqual(imbalanced_matrix(),fixture['study'])

    def test_cli_export_unpack_no_overwrite_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'input.json').write_text('{"weights":[[1,2],[3,4]],"x":[1,1]}')
            def run(*args):return subprocess.run([sys.executable,'-m','quantlab.cli',*args],capture_output=True,text=True)
            self.assertEqual(run('pack','--input',str(p/'input.json'),'--output',str(p/'out.qlb')).returncode,0)
            self.assertEqual(run('pack','--input',str(p/'input.json'),'--output',str(p/'out.qlb')).returncode,2)
            self.assertEqual(run('unpack','--input',str(p/'out.qlb')).returncode,0)
            self.assertEqual(run('inspect','--input',str(p/'input.json')).returncode,0)
            (p/'out.qlb').write_bytes(b'bad')
            self.assertEqual(run('unpack','--input',str(p/'out.qlb')).returncode,2)
