import json
from optparse import OptionParser
import h5py

def main():
    usage = ""
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    dataset_f = args[0]
    out_f = options.out_file

    with h5py.File(dataset_f, 'r') as f:
        cell_ids = [
            str(x)[2:-1]
            for x in f['experiment'][:]
        ]

    cell_id_to_study = {
        cell_id: 'laughney'
        for cell_id in cell_ids
    }
    
    with open(out_f, 'w') as f:
        json.dump(cell_id_to_study, f, indent=4)           

if __name__ == '__main__':
    main()
