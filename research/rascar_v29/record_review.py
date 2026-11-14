"""Append explicit visual annotations, kept separate from official labels."""
from pathlib import Path
import json,sys
A=Path(__file__).resolve().parent
codes={
'G':('plausible_correct','visible_extent_plausible'),
'L':('uncertain','loose_box_needs_iou'),
'U':('clear_error','severe_visible_undercoverage'),
'O':('uncertain','occluded_person_extent'),
'D':('uncertain','duplicate_assignment'),
'M':('clear_error','merged_instances'),
'F':('clear_error','background_or_clothing'),
'W':('clear_error','wrong_class'),
'A':('uncertain','insufficient_visual_evidence'),
'P':('likely_error','localization_needs_quantification')}
def record(lines):
    p=A/'annotations/fresh_review.json'
    out=json.loads(p.read_text()) if p.exists() else {}
    for line in lines.strip().splitlines():
        no,code,note=line.split('|',2)
        assert code in codes
        status,issue=codes[code]
        out[no]=dict(review_no=int(no),visual_status_v29=status,issue_v29=issue,note_v29=note,
                     review_source='assistant_visual_reaudit',official_test_label=False,used_as_training_target=False,
                     scoring_hidden_on_contact_sheet=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('Reviewed',len(out),'of 362')
if __name__=='__main__':record(sys.stdin.read())
