import argparse
import pandas as pd
from multiprocessing import Pool
import numpy as np
from collections import Counter, defaultdict
import itertools
import os
import yaml
from fisher import pvalue_npy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', help='config file in YAML format containing config for running the classifier')
args = parser.parse_args()


def parse_data(concatenated_receptors_file, label_field):
    fields = ["junction_aa", label_field, "repertoire_id"]
    df = pd.read_csv(concatenated_receptors_file, header=0, sep="\t", usecols=fields)
    return df


def pairwise_kmer_count(sequences, repertoire_ids, k):
    counts = {}
    for seq, repertoire_id in zip(sequences, repertoire_ids):
        seq_kmers = set([seq[i:i + k] for i in range(len(seq) - k + 1)])
        seq_pairs = [(x, y) if x < y else (y, x) for x, y in itertools.combinations(seq_kmers, 2) if x != y]
        if repertoire_id not in counts:
            counts[repertoire_id] = Counter()
        for pair in seq_pairs:
            counts[repertoire_id][pair] = 1
    return counts


def pairwise_kmer_count_chunked(sequences, repertoire_ids, k, num_processes):
    counts = {}
    chunk_size = len(sequences) // num_processes
    pool = Pool(num_processes)
    results = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(sequences)
        chunk_sequences = sequences[start_idx:end_idx]
        chunk_repertoire_ids = repertoire_ids[start_idx:end_idx]
        results.append(pool.apply_async(pairwise_kmer_count, (chunk_sequences, chunk_repertoire_ids, k)))
    for r in results:
        chunk_counts = r.get()
        for repertoire_id, repertoire_counts in chunk_counts.items():
            if repertoire_id not in counts:
                counts[repertoire_id] = Counter()
            counts[repertoire_id].update(repertoire_counts)
    pool.close()
    pool.join()
    return counts


def get_unique_pairs_counts(counts):
    unique_pairs_counts = defaultdict(int)
    for repertoire_counts in counts.values():
        for pair in repertoire_counts.keys():
            unique_pairs_counts[pair] += 1
    return unique_pairs_counts


def get_pairwise_kmer_counts(receptor_file, label_field, target_label, k, num_processes):
    unique_df = receptor_file.drop_duplicates(subset='repertoire_id')
    label_counts = unique_df[label_field].value_counts().to_dict()
    airr1 = receptor_file.loc[receptor_file[label_field] == target_label, ['junction_aa', 'repertoire_id']]
    airr1_seq = airr1['junction_aa'].values
    airr1_reps = airr1['repertoire_id'].values
    airr1_pairwise_counts = pairwise_kmer_count_chunked(airr1_seq, airr1_reps, k, num_processes)
    airr1_pairwise_counts = get_unique_pairs_counts(airr1_pairwise_counts)
    airr2 = receptor_file.loc[receptor_file[label_field] != target_label, ['junction_aa', 'repertoire_id']]
    airr2_seq = airr2['junction_aa'].values
    airr2_reps = airr2['repertoire_id'].values
    airr2_pairwise_counts = pairwise_kmer_count_chunked(airr2_seq, airr2_reps, k, num_processes)
    airr2_pairwise_counts = get_unique_pairs_counts(airr2_pairwise_counts)
    print("combining counters: ---------")
    pair_counts = combine_defaultdicts(airr1_pairwise_counts, airr2_pairwise_counts)
    n1 = label_counts[target_label]
    n2 = label_counts[list(set(label_counts.keys()).difference(set([config['target_label']])))[0]]
    print("performing fishers exact test: ---------")
    pair_counts = fisher_exact_test(pair_counts, n1=n1, n2=n2)
    return airr1_seq, airr2_seq, pair_counts, n1, n2


def combine_defaultdicts(dict1, dict2):
    unique_pairs = set(dict1.keys()) | set(dict2.keys())
    data = []
    for pair in unique_pairs:
        count1 = dict1[pair] if pair in dict1 else 0
        count2 = dict2[pair] if pair in dict2 else 0
        data.append([pair, pair[0], pair[1], count1, count2])
    df = pd.DataFrame(data, columns=['pair', 'kmer1', 'kmer2', 'class1_pos', 'class2_pos'])
    df.set_index('pair', inplace=True)
    return df


def fisher_exact_test(df, n1, n2):
    df['class1_neg'] = n1 - df['class1_pos']
    df['class2_neg'] = n2 - df['class2_pos']
    contingency_npy = df[['class1_pos', 'class1_neg', 'class2_pos', 'class2_neg']].values
    contingency_npy = contingency_npy.astype(dtype=np.uint)
    _, _, twosided = pvalue_npy(contingency_npy[:, 0], contingency_npy[:, 1],
                                contingency_npy[:, 2], contingency_npy[:, 3])
    contin_odd_npy = contingency_npy + 1
    odds = (contin_odd_npy[:, 0] * contin_odd_npy[:, 3]) / (contin_odd_npy[:, 1] * contin_odd_npy[:, 2])
    df['odds_ratio'] = odds
    df['p_value'] = twosided
    return df.drop(['class1_neg', 'class2_neg'], axis=1)


def score_sequences(sequences, k, significant_pairs, num_processes):
    significant_pairs = set(significant_pairs)
    scores = np.zeros(len(sequences))
    chunk_size = len(sequences) // num_processes
    pool = Pool(num_processes)
    results = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(sequences)
        chunk_sequences = sequences[start_idx:end_idx]
        results.append(
            pool.apply_async(compute_scores_chunk, (chunk_sequences, k, significant_pairs, start_idx, end_idx)))
    for r in results:
        chunk_scores, start_idx, end_idx = r.get()
        scores[start_idx:end_idx] += chunk_scores
    return scores


def compute_scores_chunk(sequences, k, significant_pairs, start_idx, end_idx):
    chunk_scores = np.zeros(len(sequences))
    for i, seq in enumerate(sequences):
        seq_kmers = set([seq[j:j + k] for j in range(len(seq) - k + 1)])
        seq_pairs = set([(x, y) if x < y else (y, x) for x, y in itertools.combinations(seq_kmers, 2) if x != y])
        n_pairs_intersect = len(set.intersection(significant_pairs, seq_pairs))
        if n_pairs_intersect > 0:
            chunk_scores[i] = n_pairs_intersect
    return chunk_scores, start_idx, end_idx


def get_unique_repertoire_int_ids(receptor_file):
    print("returning int ids: ---------")
    reps = receptor_file['repertoire_id'].values
    epitopes = receptor_file['epitope'].values
    unique_reps = list(set(reps))
    id_map = {id_: i for i, id_ in enumerate(unique_reps)}
    int_ids = [id_map[id_] for id_ in reps]
    print("computing nd returning unique epitopes: ---------")
    epitope_dict = {}
    for id_, epitope in zip(reps, epitopes):
        if id_ not in epitope_dict:
            epitope_dict[id_] = epitope

    unique_epitopes = [epitope_dict[id_] for id_ in unique_reps]
    #     unique_epitopes = np.asarray(unique_epitopes, dtype=int)
    return int_ids, unique_epitopes


def get_significant_pairs(pair_counts, top_n, neg_class_size=None):
    print("returning significant pairs: ---------")
    pair_counts_sorted = pair_counts.sort_values("p_value", ascending=True)
    pair_counts_sorted = pair_counts_sorted[pair_counts_sorted["odds_ratio"] > 2]
    if neg_class_size is not None:
        neg_class_threshold = int(neg_class_size * 0.03)
        pair_counts_sorted = pair_counts_sorted[pair_counts_sorted["class2_pos"] < neg_class_threshold]
    significant_pairs = pair_counts_sorted.head(n=top_n).index.values
    return significant_pairs


def get_pos_instance_counts(receptor_file, k, significant_pairs, num_processes):
    seq = receptor_file['junction_aa'].values
    print("scoring all sequences: ---------")
    seq_scores = score_sequences(seq, k, significant_pairs, num_processes)
    pos_instances = seq[np.nonzero(seq_scores)]
    int_ids, unique_epitopes = get_unique_repertoire_int_ids(receptor_file)
    pos_inst_counts = np.bincount(int_ids, weights=seq_scores)
    return pos_inst_counts, pos_instances, unique_epitopes


def fit_model(pos_inst_counts, class_labels):
    clf = LogisticRegression().fit(pos_inst_counts, class_labels)
    return clf


def classifier(receptor_file, label_field, target_label, k, num_processes, top_n, group_field, n_splits,
               filter_by_neg_class_size=None):
    y = receptor_file[label_field]
    groups = receptor_file[group_field]
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    balanced_accuracy_scores = []
    detailed_results = {}
    for i, (train_idx, test_idx) in enumerate(sgkf.split(receptor_file, y, groups)):
        X_train, X_test = receptor_file.iloc[train_idx], receptor_file.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        airr1_seq, airr2_seq, pair_counts, n1, n2 = get_pairwise_kmer_counts(X_train, label_field=label_field,
                                                                             target_label=target_label, k=k,
                                                                             num_processes=num_processes)
        if filter_by_neg_class_size is not None:
            significant_pairs = get_significant_pairs(pair_counts, top_n=top_n, neg_class_size=n2)
        else:
            significant_pairs = get_significant_pairs(pair_counts, top_n=top_n, neg_class_size=None)
        pos_inst_counts, pos_instances, class_labels = get_pos_instance_counts(X_train, k, significant_pairs,
                                                                               num_processes)
        fitted_model_ = fit_model(pos_inst_counts.reshape(-1, 1), np.array(class_labels))

        test_pos_inst_counts, test_pos_instances, test_class_labels = get_pos_instance_counts(X_test, k,
                                                                                              significant_pairs,
                                                                                              num_processes)

        predicted_test_labels = fitted_model_.predict(test_pos_inst_counts.reshape(-1, 1))
        test_rep_ids, true_test_labels = get_unique_repertoire_int_ids(X_test)
        balanced_accuracy_scores.append(
            balanced_accuracy_score(np.array(true_test_labels), np.array(predicted_test_labels)))
        pos_instances = {"train_pos_instances": pos_instances, "test_pos_instances": test_pos_instances}

        detailed_results[f"split_{i}"] = {"significant_pairs": significant_pairs, "pos_inst_counts": pos_inst_counts,
                                          "class_labels": X_train[label_field].values,
                                          "test_pos_inst_counts": test_pos_inst_counts,
                                          "test_class_labels": X_test[label_field].values,
                                          "predicted_test_labels": predicted_test_labels,
                                          "pair_counts": pair_counts, "positive_instances": pos_instances}
        detailed_results["crossval_balanced_accuracy_scores"] = balanced_accuracy_scores
    return detailed_results


def write_nested_dict_to_files(data, path, delimiter=","):
    if not os.path.exists(path):
        os.makedirs(path)
    for key, value in data.items():
        subpath = os.path.join(path, str(key))
        if isinstance(value, dict):
            write_nested_dict_to_files(value, subpath, delimiter)
        elif isinstance(value, (list, np.ndarray)):
            filename = f"{subpath}.csv" if delimiter == "," else f"{subpath}.tsv"
            if isinstance(value, np.ndarray):
                value = value.tolist()
            df = pd.DataFrame(value)
            df.to_csv(filename, sep=delimiter, index=False)
        elif isinstance(value, pd.DataFrame):
            filename = f"{subpath}.csv" if delimiter == "," else f"{subpath}.tsv"
            value.to_csv(filename, sep=delimiter, index=False)


def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def execute():
    config = read_yaml_config(args.config_file)
    receptor_file = parse_data(concatenated_receptors_file=config['concatenated_receptors_file'],
                               label_field=config['label_field'])
    detailed_results = classifier(receptor_file,
                                  label_field=config['label_field'],
                                  target_label=config['target_label'],
                                  k=config['k'],
                                  num_processes=config['num_processes'],
                                  top_n=config['top_n'],
                                  group_field=config['group_field'],
                                  n_splits=config['n_splits'],
                                  filter_by_neg_class_size=None)
    write_nested_dict_to_files(data=detailed_results, path=config['output_path'], delimiter="\t")