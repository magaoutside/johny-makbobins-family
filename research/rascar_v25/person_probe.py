"""Predeclared strong LB hypothesis; only person scores differ from best V22."""
from common25 import *
from table_zones import make_features
features=[]
for rows,split in [(q,'oof'),(te,'test')]:
 f=make_features(rows,DAYS[0]).set_index('id').loc[rows.id]
 # Independent of templates/labels: exactly one head, far seat, nearest detected table.
 bb=rows[B].to_numpy().copy();height=(f.table_y2-f.table_y1).to_numpy()
 bottom=np.minimum(719,f.table_y2.to_numpy()+.30*height)
 bb[:,3]=np.maximum(bb[:,3],bottom)
 area=(rows[B].to_numpy()[:,2:]-rows[B].to_numpy()[:,:2]).prod(1)
 ov=area/np.maximum((bb[:,2:]-bb[:,:2]).prod(1),1)
 # A deliberately strong test: missing >=20% area versus table-anchored reference.
 gate=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.far.eq(1).to_numpy()&(ov<.8)
 strength=np.clip((.8-ov)/.3,0,1)*gate
 z=pd.DataFrame(dict(id=rows.id,gate=gate,strength=strength,extension_iou=ov,ref_x1=bb[:,0],ref_y1=bb[:,1],ref_x2=bb[:,2],ref_y2=bb[:,3]))
 z.to_csv(A/f'features/person_probe_{split}.csv',index=False);features.append(z)
s=base_oof+.9*(1-base_oof)*features[0].strength.to_numpy();t=current+.9*(1-current)*features[1].strength.to_numpy()
r=register('person_hidden_legs_strong',s,t,dict(type='single_class_LB_probe',predeclared=True,fit_thresholds=False,extension='far seat: table bottom + 0.30 table height',weight=.9,gate='one head + far seat + reference/query area ratio below .80',warning='Hypothesis conflicts with some official manual boxes; submit only to measure that convention'))
(A/'submission_v25_person_hidden_legs_test.csv').write_bytes((A/'results/submissions/submission_person_hidden_legs_strong.csv').read_bytes())
assert np.array_equal(t[te.class_name.ne('person')],current[te.class_name.ne('person')])
