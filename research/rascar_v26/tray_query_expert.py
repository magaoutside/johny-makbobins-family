"""Image/physical-size confirmation when the old tray detector misses an object.

The query seeds the search region, but the proposed rotated rectangle is fixed
by fit-day physical dimensions, camera metric and image edges. This is distinct
from the independent-reference completion experiment and is reported separately.
"""
from perspective_tray import *

def extract(rows,fit,policy='checked',mode='physical'):
    result=[]
    for r in rows.itertuples():
        f=dict(id=r.id,geometry_iou=0.,support=0.,gain=0.,area_ratio=1.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
        if r.class_name=='tray':
            b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2
            if 140<cc[0]<1260 and 190<cc[1]<400:
                d=complete(r.frame_id,b,fit,mode,policy)
                if d['valid']:
                    f.update(geometry_iou=float(iou(b,d['box'])[0,0]),support=d['edge'],gain=d['gain'],area_ratio=d['area_ratio'],**dict(zip(['ref_'+c for c in B],d['box'])))
        result.append(f)
    return pd.DataFrame(result).set_index('id')

if __name__=='__main__':
    policy='oriented_grouped' if 'oriented_grouped' in sys.argv else 'oriented' if 'oriented' in sys.argv else 'checked'
    mode='monotone' if 'monotone' in sys.argv else 'physical'
    key='tray_query' if policy=='checked' else 'tray_query_oriented'
    if mode=='monotone':key+='_'+mode
    if policy=='oriented_grouped':key+='_grouped'
    of=[];tf=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        o=extract(q[q.day.eq(held)],fit,policy,mode);t=extract(te,fit,policy,mode);of.append(o);tf.append(t)
        o.to_csv(A/f'features/{key}_held_{fit}.csv');t.to_csv(A/f'features/{key}_test_{fit}.csv')
    fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
    for cut in ([.35,.5,.65] if policy.startswith('oriented') else [.65,.8]):
        for ov in [.7,.8]:
            for weight in [.2,.4]:
                scores=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    gate=guard(split,rows,'tray')&rows.class_name.eq('tray').to_numpy()&f.support.ge(cut).to_numpy()&f.geometry_iou.ge(ov).to_numpy()
                    if split=='test':
                        # Both independently calibrated days must confirm.
                        gate &= np.logical_and.reduce([(z.support.ge(cut)&z.geometry_iou.ge(ov)).to_numpy() for z in tf])
                    geometry_error=np.clip((.9-f.geometry_iou.to_numpy())/.5,0,1)
                    scores.append(np.clip(base+weight*gate*(np.minimum(base,geometry_error)-base),0,1))
                register(f'{key}_confirm_e{cut}_iou{ov}_w{weight}',*scores,dict(type='query_seeded_physical_rectangle_image_confirmation',policy=policy,camera_mode=mode,edge=cut,iou=ov,weight=weight,fit='opposite-day calibration and physical dimensions',direction='rescue only',test_fold_agreement=True))
