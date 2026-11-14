from pathlib import Path
import json,csv,base64,hashlib,collections,datetime,html
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
A=Path(__file__).resolve().parent;ROOT=A.parents[1]
notes={int(a[0]):dict(flag=a[1],certainty=a[2],note=a[3]) for l in (A/'review.txt').read_text().splitlines() if (a:=l.split('|',3))}
assert set(notes)==set(range(1,363))
t=json.loads((A/'manifest.json').read_text());q=json.loads((A/'train_manifest.json').read_text());tn=json.loads((A/'train_notes.json').read_text())
status={'down':'Возможно завышен','up':'Возможно занижен','review':'Спорный случай','ok':'Явной проблемы не видно','rank_up':'Занижен относительно правильных','rank_down':'Завышен относительно ошибочных','rank_ok':'Нет перепутанных пар'}
for r in t:
 r.update(notes[r['review_no']],split='test',key=f"test-{r['review_no']}",display=f"№{r['review_no']:03d}",image=f"crops/{r['review_no']:03d}.jpg",label_source='visual_diagnostic_not_ground_truth',error=None,wrong_pairs=None)
 r['status']=status[r['flag']];r['score']=r['score_v52'];r['rank_v52']=1+sum(x['score_v52']>r['score_v52'] for x in t);r['rank_a40']=1+sum(x['score_a40']>r['score_a40'] for x in t)
for r in q:
 for k in ['score','max_gt_iou','ap_loss_contribution']:r[k]=float(r[k])
 for k in ['error','wrong_pairs','train_no']:r[k]=int(r[k])
 r['flag']=('rank_up' if r['error'] else 'rank_down') if r['wrong_pairs'] else 'rank_ok';r['certainty']='official';r['status']=status[r['flag']]
 r['note']=tn.get(str(r['train_no']),f"Просмотрено с официальным GT. Метка error={r['error']}, максимальный IoU={r['max_gt_iou']:.3f}; перепутанных пар V52: {r['wrong_pairs']}. Дополнительного визуального основания пересматривать официальную метку не найдено.")
 r.update(split='train',key=f"train-{r['train_no']}",display=f"T{r['train_no']:03d}",image=f"crops/T{r['train_no']:03d}.jpg",label_source='official_train_error_and_day_out_oof',score_v52=r['score'],score_a40=None,delta=None,rank_v52=None,rank_a40=None)
assert len(q)==198 and len(t)==362 and len({r['id'] for r in t})==362
pos=[r for r in q if r['error']];neg=[r for r in q if not r['error']]
pairs=[dict(error_id=p['id'],correct_id=n['id'],error_train_no=p['train_no'],correct_train_no=n['train_no'],error_class=p['class_name'],correct_class=n['class_name'],error_score=p['score'],correct_score=n['score']) for p in pos for n in neg if n['score']>p['score']]
assert len(pairs)==153 and sum(r['wrong_pairs'] for r in q)==2*len(pairs)
def savecsv(name,rows):
 cols=list(dict.fromkeys(k for r in rows for k in r if k!='image_data'))
 with (A/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
savecsv('test_all_362.csv',t);savecsv('test_suspected_43.csv',[r for r in t if r['flag'] in ['up','down']]);savecsv('test_uncertain_127.csv',[r for r in t if r['flag']=='review']);savecsv('train_all_198.csv',q);savecsv('train_rank_errors_49.csv',[r for r in q if r['wrong_pairs']]);savecsv('train_wrong_pairs_153.csv',pairs)
changed=[r for r in t if abs(r['delta'])>1e-9];savecsv('a40_changed_test_131.csv',sorted(changed,key=lambda r:-abs(r['delta'])))
summary=dict(test_count=len(t),test_frames=len({r['frame_id'] for r in t}),train_count=len(q),train_frames=len({r['frame_id'] for r in q}),visually_reviewed_test=362,visually_reviewed_train=198,closeup_second_pass=24,test_flags=dict(collections.Counter(r['flag'] for r in t)),test_by_class={c:dict(collections.Counter(r['flag'] for r in t if r['class_name']==c)) for c in sorted({r['class_name'] for r in t})},train_oof_ap=average_precision_score([r['error'] for r in q],[r['score'] for r in q]),train_wrong_pairs=len(pairs),train_boxes_in_wrong_pairs=sum(r['wrong_pairs']>0 for r in q),train_underprioritized_errors=sum(r['wrong_pairs']>0 and r['error']==1 for r in q),train_overprioritized_correct=sum(r['wrong_pairs']>0 and r['error']==0 for r in q),a40_changed_test=len(changed),a40_increased=sum(r['delta']>0 for r in changed),a40_decreased=sum(r['delta']<0 for r in changed),a40_visual_down_and_increased=[r['review_no'] for r in t if r['flag']=='down' and r['delta']>0.04],public_v52=.96300,public_a40=.96285,public_delta=-.00015,remaining_submissions=1,public_source='user message 2026-11-14',test_gt_available=False,submissions_modified=False,models_trained_locally=False)
(A/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(ROOT/'rascar_v56/audit/public_result_v56_a40.json').write_text(json.dumps({k:summary[k] for k in ['public_v52','public_a40','public_delta','remaining_submissions','public_source']},ensure_ascii=False,indent=2))
for r in t+q:r['image_data']='data:image/jpeg;base64,'+base64.b64encode((A/r['image']).read_bytes()).decode()
data=json.dumps(t+q,ensure_ascii=False).replace('</','<\\/')
template=(A/'template.html').read_text();report=template.replace('__DATA__',data).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=False))
(A/'report.html').write_text(report)
sources=[ROOT/'rascar_data_full/test.csv',ROOT/'rascar_data_full/train.csv',ROOT/'rascar_data_full/human_boxes_train.csv',ROOT/'rascar_v52/release/submission_v52_candidate_purged_a20.csv',ROOT/'rascar_v56/release/submission_v56_tabicl_hands_a40.csv',A/'review.txt',A/'train_notes.json',A/'report.html']
integrity={'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'coverage':{'test_ids':len({r['id'] for r in t}),'train_ids':len({r['id'] for r in q}),'test_pages':31,'train_pages':17,'focus_pages':4},'test_labels_created':False,'test_notes_used_for_training':False,'test_notes_used_to_edit_submission':False}
(A/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2));print('HTML bytes',(A/'report.html').stat().st_size)
