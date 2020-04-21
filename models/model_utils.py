import pandas as pd

def main():
    label_to_score_list = [
        {'A': 1, 'B': 1, 'C': 1},
        {'A': 2, 'B': 2, 'C': 2},
        {'A': 3, 'B': 3, 'C': 3}
    ]
    label_to_name = {
        'A': 'nA', 
        'B': 'nB', 
        'C': 'nC'
    }
    exps = ['e1', 'e2', 'e3']
    df = convert_to_matrix(
        label_to_score_list, 
        exps
    )
    print(df)

def convert_to_matrix(label_to_score_list, exps):
    all_labels = set()
    for label_to_score in label_to_score_list:
        all_labels.update(label_to_score.keys())
    all_labels = sorted(all_labels)
    
    mat = [
        [
            label_to_score[label]
            for label in all_labels
        ]
        for label_to_score in label_to_score_list
    ]
    df = pd.DataFrame(
        data=mat,
        index=exps,
        columns=all_labels
    )
    df = df.transpose()
    return df

if __name__ == '__main__':
    main()
