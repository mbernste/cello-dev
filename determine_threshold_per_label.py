from optparse import OptionParser
import pandas as pd
import json
import sys

from common import the_ontology

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    pr_curves_f = args[0]
    out_f = options.out_file

    og = the_ontology.the_ontology()
    
    with open(pr_curves_f, 'r') as f:
        method_to_label_to_pr_curves = json.load(f)
        
    assert len(method_to_label_to_pr_curves) == 1
    method = sorted(method_to_label_to_pr_curves.keys())[0]
    label_to_pr_curves = method_to_label_to_pr_curves[method]

    da = []
    for label, pr in label_to_pr_curves.items():
        precs = pr['precisions']
        recs = pr['recalls']
        threshs = pr['thresholds']
        f1s = map(_compute_f1, zip(precs, recs))
        max_f1_thresh = max(zip(f1s, threshs), key=lambda x: x[0])
        da.append((label, og.id_to_term[label].name, max_f1_thresh[1], max_f1_thresh[0]))
    
    df = pd.DataFrame(
        data=da, 
        columns=['label', 'label_name', 'threshold', 'F1-score']
    )
    df.to_csv(out_f, sep='\t', index=False)
    print(df) 
                
        

def _compute_f1(r):
    prec = r[0]
    rec = r[1]
    try:
        f1 = 2 * ((prec * rec)/(prec + rec))    
    except ZeroDivisionError:
        f1 = 0.0
    return f1
    

if __name__ == "__main__":
    main()
