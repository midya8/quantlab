"""Separate new activation controls; original fixture and witnesses stay exact."""
import json,itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.integer import integer_audit

def cases():
    root=Path(__file__).resolve().parents[1];f=json.loads((root/'quantlab/data/fixture.json').read_text())
    return [
        {'name':'Activation tie collapses the winner','input':{'weights':[[127.,0.],[0.,127.]],'x':[.4,.49],'activation_range':127.}},
        {'name':'Fixed range clips the large coordinate','input':{'weights':[[1.,-1.],[-1.,1.]],'x':[8.,2.],'activation_range':1.}},
        {'name':'Exact half-way activation codes','input':{'weights':[[127.,1.,-1.,2.],[-127.,-1.,1.,-2.]],'x':[.5,1.5,-.5,-1.5],'activation_range':127.}},
        {'name':'All zeros','input':{'weights':[[0.,0.],[0.,0.]],'x':[0.,0.],'activation_range':1.}},
        {'name':'Maximum accumulator bound','input':{'weights':[[127.]*128,[-127.]*128],'x':[127.]*128,'activation_range':127.}},
        {'name':'Stored scale precision','input':{'weights':[[.1,.4],[-.3,.2]],'x':[1.,3.],'activation_range':.7}},
    ]+[{'name':'Original test row '+str(i),'input':{'weights':f['weights'],'x':f['test'][i]['x'],'activation_range':1.}}for i in [0,19,59]]

def export(check=False):
    root=Path(__file__).resolve().parents[1];cs=cases();refs=[]
    for c,mode,bits,abits in itertools.product(cs,['per-channel','per-tensor'],[32,64],[32,64]):refs.append({'name':c['name'],'input':c['input'],'mode':mode,'scale_bits':bits,'activation_bits':abits,'report':integer_audit(c['input'],mode,bits,abits)})
    outputs={'docs/integer-data.js':'globalThis.IntegerCases='+json.dumps(cs,separators=(',',':'))+';\n'}
    for i in range(0,len(refs),12):outputs['quantlab/data/integer-'+str(i//12)+'.json']=json.dumps(refs[i:i+12],separators=(',',':'))+'\n'
    for f,s in outputs.items():
        if check:assert(root/f).read_text()==s,'stale '+f
        else:(root/f).write_text(s)
    print(json.dumps({'reports':len(refs),'products':sum(len(r['products'])for c in refs for r in c['report']['rows'])}))
if __name__=='__main__':export('--check'in sys.argv)
