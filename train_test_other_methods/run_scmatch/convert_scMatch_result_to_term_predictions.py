from optparse import OptionParser
from collections import defaultdict
import sys
import os
from os.path import join
import json
import pandas as pd

from common import the_ontology

SCMATCH_OUTPUT_TO_TERMS = {
    'CD4+ T Cells': [
        'CL:0000624'    # CD4-positive, alpha-beta T cell 
    ],
    'CD4+CD25+CD45RA- memory regulatory T cells expanded': [
        'CL:0002678'    # memory regulatory T cell 
    ],
    'CD133+ stem cells - adult bone marrow derived': [
        'CL:0000037'    # hematopoietic stem cell
    ],
    'CD133+ stem cells - adult bone marrow derived, pool1.CNhs12552.12224-129F1': [
        'CL:0000037'    # hematopoietic stem cell
    ], 
    'CD4+CD25+CD45RA- memory regulatory T cells': [
        'CL:0002678'    # memory regulatory T cell 
    ], 
    'CD4+CD25-CD45RA+ naive conventional T cells expanded': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ], 
    'CD4+CD25-CD45RA+ naive conventional T cells': [
        'CL:0000895'    # naive thymus-derived CD4-positive, alpha-beta T cell
    ], 
    'CD4+CD25-CD45RA- memory conventional T cells': [
        'CL:0000897'    # CD4-positive, alpha-beta memory T cell 
    ], 
    'CD4+CD25-CD45RA- memory conventional T cells expanded': [
        'CL:0000897'    # CD4-positive, alpha-beta memory T cell 
    ], 
    'CD34+ stem cells - adult bone marrow derived': [
        'CL:0000037'    # hematopoietic stem cell
    ], 
    'CD14+CD16+ Monocytes': [
        'CL:0002397'    # CD14-positive, CD16-positive monocyte
    ], 
    'CD4+CD25+CD45RA+ naive regulatory T cells': [
        'CL:0000895',   # naive thymus-derived CD4-positive, alpha-beta T cell
        'CL:0002677'    # naive regulatory T cell
    ], 
    'CD8+ T Cells (pluriselect)': [
        'CL:0000625'    # CD8-positive, alpha-beta T cell
    ],
    'Monocyte-derived macrophages response to mock influenza infection': [
        'CL:0000235'    # macrophage
    ], 
    'CD4+CD25+CD45RA+ naive regulatory T cells expanded': [
        'CL:0000895',   # naive thymus-derived CD4-positive, alpha-beta T cell
        'CL:0002677'    # naive regulatory T cell
    ],
    'gamma delta positive T cells': [
        'CL:0000798'    # gamma-delta T cell
    ], 
    'CD8+ T Cells': [
        'CL:0000625'    # CD8-positive, alpha-beta T cell
    ], 
    'Natural Killer Cells': [
        'CL:0000623'    # natural killer cell
    ], 
    'Dendritic Cells - plasmacytoid': [
        'CL:0000784'    # plasmacytoid dendritic cell
    ], 
    'CD14-CD16+ Monocytes': [
        'CL:0002396'    # CD14-low, CD16-positive monocyte
    ],
    'CD19+ B Cells (pluriselect)': [
        'CL:0000236'    # B cell
    ],
    'CD34+ Progenitors': [
        'CL:0008001'    # hematopoietic precursor cell
    ],
    'Basophils': [
        'CL:0000767'    # basophil
    ], 
    'Monocyte-derived macrophages response to udorn influenza infection': [
        'CL:0000235'    # macrophage
    ],
    'CD14+CD16- Monocytes': [
        'CL:0002057'    # CD14-positive, CD16-negative classical monocyte
    ],
    'CD19+ B Cells': [
        'CL:0000236'    # B cell
    ]
}


def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write output")
    (options, args) = parser.parse_args()

    result_f = args[0]
    out_dir = options.out_dir

    og = the_ontology.the_ontology()

    scmatch_output_to_all_terms = defaultdict(lambda: set())
    all_terms = set()
    for scmatch_out, terms in SCMATCH_OUTPUT_TO_TERMS.items():
        for term in terms:
            scmatch_output_to_all_terms[scmatch_out].update(
                og.recursive_superterms(term)
            )
            all_terms.update(
                og.recursive_superterms(term)
            )
    scmatch_output_to_all_terms = dict(scmatch_output_to_all_terms)
    all_terms = sorted(all_terms)

    results_df = pd.read_csv(result_f, index_col=0)
    print(results_df)
    conf_da = []
    bin_da = []
    nonmapped_samples = set()
    for cell in results_df.index:
        scmatch_out = results_df.loc[cell]['top sample'].split(',')[0]
        score = results_df.loc[cell]['top correlation score']
        try:
            terms = scmatch_output_to_all_terms[scmatch_out]
        except KeyError:
            nonmapped_samples.add(scmatch_out)
            terms = []
        term_scores = []
        term_assigns = []
        for term in all_terms:
            if term in terms:
                term_scores.append(score)
                term_assigns.append(1)
            else:
                term_scores.append(float('-inf'))
                term_assigns.append(0)
        conf_da.append(term_scores)
        bin_da.append(term_assigns)
    print('Could not the following samples to ontology terms:')
    print('\n'.join(nonmapped_samples)) 
    conf_df = pd.DataFrame(
        data=conf_da,
        columns=all_terms,
        index=results_df.index
    )
    bin_df = pd.DataFrame(
        data=bin_da,
        columns=all_terms,
        index=results_df.index
    )
    conf_df.to_csv(join(out_dir, 'classification_results.tsv'), sep='\t')
    bin_df.to_csv(join(out_dir, 'binary_classification_results.tsv'), sep='\t')
    



            

if __name__ == "__main__":
    main()
