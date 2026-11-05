from fractions import Fraction
import json
from pathlib import Path
import unittest
from detection_ap import report,class_ap,overlap

D=Path(__file__).resolve().parents[1]/'examples'

def case(n):
    return json.loads((D/(n+'.json')).read_text())

class APTests(unittest.TestCase):
    def test_duplicate_exact_envelope(self):
        a=case('duplicate')
        ap,curve=class_ap(a['truth'],a['predictions'],'gear',Fraction(1,2))
        self.assertEqual(ap,Fraction(253,303))
        self.assertEqual([hit for _,_,hit in curve],[True,False,True])

    def test_localization_sweep_exact_boundary(self):
        a=case('localization')
        self.assertEqual(overlap(a['truth'][0]['box'],a['predictions'][0]['box']),Fraction(7,10))
        r=report(a)
        self.assertEqual((r['map50'],r['map75'],r['map50_95']),(1,0,.5))
        self.assertEqual(r['ap_by_class_and_iou']['0.70']['gear'],1)

    def test_macro_average_includes_unpredicted_truth(self):
        r=report(case('classes'))
        self.assertEqual(r['map50_95'],.5)
        self.assertEqual(r['excluded_labels'],['washer'])
        self.assertEqual(r['ap_by_class_and_iou']['0.50'],{'gear':1,'bolt':0})

    def test_wrong_image_and_no_predictions(self):
        a=case('localization')
        a['predictions'][0]['image']='wrong'
        self.assertEqual(report(a)['map50'],0)
        a['predictions']=[]
        self.assertEqual(report(a)['map50_95'],0)

    def test_invalid_inputs(self):
        for score in [float('nan'),True,-1,1.01]:
            a=case('localization'); a['predictions'][0]['score']=score
            with self.assertRaises(ValueError):
                report(a)
        a=case('localization');a['truth'][0]['box']=[0,0,0,4]
        with self.assertRaises(ValueError):
            report(a)
