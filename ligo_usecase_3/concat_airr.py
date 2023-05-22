import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', help='metadata file containing a filename column with path to files')
parser.add_argument('--repertoire_concat_file', help='Output file with concatenated repertoires')
args = parser.parse_args()


def concatenate_airr_files(metadata_file, repertoire_concat_file):
    meta = pd.read_csv(metadata_file, header=0)
    outfile_df = _read_and_concatenate_repertoires(meta)
    outfile_df.to_csv(repertoire_concat_file, index=None, sep="\t")


def _read_and_concatenate_repertoires(meta):
    li = []
    for i, filename in enumerate(meta['filename']):
        print("processing file number:", i)
        fn_path = os.path.join("repertoires", filename)
        fn = pd.read_csv(fn_path, header=0, sep='\t')
        fn["epitope"] = meta["sim_item"][i]
        fn["repertoire_id"] = meta["repertoire_id"][i]
        li.append(fn)
    outfile_df = pd.concat(li, axis=0, ignore_index=True)
    return outfile_df


def execute():
    concatenate_airr_files(args.metadata_file, args.repertoire_concat_file)