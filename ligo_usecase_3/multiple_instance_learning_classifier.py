import pandas as pd
from multiprocessing import Pool
import numpy as np
from collections import Counter
import itertools
import os
import yaml
import argparse
from fisher import pvalue_npy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', help='config file in YAML format containing config for running the classifier')
args = parser.parse_args()


def parse_data(concatenated_receptors_file, label_field):
    fields = ["junction_aa", label_field, "repertoire_id"]
    df = pd.read_csv(concatenated_receptors_file, header=0, sep="\t", usecols=fields)
    return df


def pairwise_kmer_count(sequences, k):
    counts = Counter()
    for seq in sequences:
        seq_kmers = set([seq[i:i + k] for i in range(len(seq) - k + 1)])
        seq_pairs = [(x, y) if x < y else (y, x) for x, y in itertools.combinations(seq_kmers, 2) if x != y]
        for pair in seq_pairs:
            counts[pair] += 1
    return counts


def pairwise_kmer_count_chunked(sequences, k, num_processes):
    counts = Counter()
    chunk_size = len(sequences) // num_processes
    pool = Pool(num_processes)
    results = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(sequences)
        chunk_sequences = sequences[start_idx:end_idx]
        results.append(pool.apply_async(pairwise_kmer_count, (chunk_sequences, k)))
    for r in results:
        chunk_counts = r.get()
        for pair, count in chunk_counts.items():
            counts[pair] += count
    return counts


def combine_counters(counter1, counter2):
    pairs = list(set(counter1.keys()) | set(counter2.keys()))
    pairs.sort()
    data = []
    for pair in pairs:
        count1 = counter1[pair] if pair in counter1 else 0
        count2 = counter2[pair] if pair in counter2 else 0
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


def score_sequences(sequences, k, significant_pairs):
    significant_pairs = set(significant_pairs)
    scores = np.zeros(len(sequences))
    for i, seq in enumerate(sequences):
        seq_kmers = set([seq[i:i + k] for i in range(len(seq) - k + 1)])
        seq_pairs = set([(x, y) if x < y else (y, x) for x, y in itertools.combinations(seq_kmers, 2) if x != y])
        if len(set.intersection(significant_pairs, seq_pairs)) > 0:
            scores[i] = 1
    return scores


def get_unique_repertoire_int_ids(receptor_file, label_field, label):
    reps = receptor_file.loc[receptor_file[label_field] == label, 'repertoire_id'].values
    id_map = {id_: i for i, id_ in enumerate(set(reps))}
    int_ids = [id_map[id_] for id_ in reps]
    return int_ids


def get_pairwise_kmer_counts(receptor_file, label_field, k, num_processes):
    receptor_file = receptor_file.sort_values(label_field)
    label_counts = receptor_file[label_field].value_counts().to_dict()
    labels = list(label_counts.keys())
    labels.sort()
    if len(labels) != 2:
        raise ValueError("Two unique labels are required")
    airr1_seq = receptor_file.loc[receptor_file[label_field] == labels[0], 'junction_aa'].values
    airr1_pairwise_counts = pairwise_kmer_count_chunked(airr1_seq, k, num_processes)
    airr2_seq = receptor_file.loc[receptor_file[label_field] == labels[1], 'junction_aa'].values
    airr2_pairwise_counts = pairwise_kmer_count_chunked(airr2_seq, k, num_processes)
    pair_counts = combine_counters(airr1_pairwise_counts, airr2_pairwise_counts)
    pair_counts = fisher_exact_test(pair_counts, n1=label_counts[labels[0]], n2=label_counts[labels[1]])
    return airr1_seq, airr2_seq, pair_counts


def get_significant_pairs(pair_counts, pval_threshold):
    pval_filtered = pair_counts[pair_counts["p_value"] < pval_threshold]
    pval_filtered = pval_filtered[pval_filtered["odds_ratio"] > 2]
    return pval_filtered.index.values


def get_pos_instance_counts(class1_seq, class2_seq, k, significant_pairs, receptor_file, label_field):
    airr1_seq_scores = score_sequences(class1_seq, k, significant_pairs)
    airr2_seq_scores = score_sequences(class2_seq, k, significant_pairs)
    class1_pos_instances = class1_seq[np.nonzero(airr1_seq_scores)]
    class2_pos_instances = class2_seq[np.nonzero(airr2_seq_scores)]
    labels = list(receptor_file[label_field].value_counts().to_dict().keys())
    labels.sort()
    class1_int_ids = get_unique_repertoire_int_ids(receptor_file, label_field=label_field, label=labels[0])
    class2_int_ids = get_unique_repertoire_int_ids(receptor_file, label_field=label_field, label=labels[1])
    class1_pos_inst_counts = np.bincount(class1_int_ids, weights=airr1_seq_scores)
    class2_pos_inst_counts = np.bincount(class2_int_ids, weights=airr2_seq_scores)
    class_labels = np.concatenate((np.repeat(labels[0], len(class1_pos_inst_counts)),
                                   np.repeat(labels[1], len(class2_pos_inst_counts))))
    pos_inst_counts = np.concatenate((class1_pos_inst_counts, class2_pos_inst_counts))
    return pos_inst_counts, class_labels, class1_pos_instances, class2_pos_instances


def fit_model(pos_inst_counts, class_labels):
    clf = LogisticRegression().fit(pos_inst_counts, class_labels)
    return clf


def classifier(receptor_file, label_field, k, num_processes, group_field, n_splits, pval_threshold):
    y = receptor_file[label_field]
    groups = receptor_file[group_field]
    gkf = GroupKFold(n_splits=n_splits)
    balanced_accuracy_scores = []
    detailed_results = {}
    for i, (train_idx, test_idx) in enumerate(gkf.split(receptor_file, y, groups)):
        X_train, X_test = receptor_file.iloc[train_idx], receptor_file.iloc[test_idx]

        airr1_seq, airr2_seq, pair_counts = get_pairwise_kmer_counts(X_train, label_field=label_field, k=k,
                                                                     num_processes=num_processes)
        significant_pairs = get_significant_pairs(pair_counts, pval_threshold)
        pos_inst_counts, class_labels, class1_pos_instances, class2_pos_instances = get_pos_instance_counts(airr1_seq,
                                                                                                            airr2_seq,
                                                                                                            k,
                                                                                                            significant_pairs,
                                                                                                            X_train,
                                                                                                            label_field)
        fitted_model_ = fit_model(pos_inst_counts.reshape(-1, 1), class_labels)

        test_airr1_seq, test_airr2_seq, test_pair_counts = get_pairwise_kmer_counts(X_test, label_field=label_field,
                                                                                    k=k, num_processes=num_processes)
        test_pos_inst_counts, test_class_labels, test_class1_pos_instances, test_class2_pos_instances = get_pos_instance_counts(
            test_airr1_seq, test_airr2_seq, k, significant_pairs, X_test, label_field)

        predicted_test_labels = fitted_model_.predict(test_pos_inst_counts.reshape(-1, 1))
        balanced_accuracy_scores.append(balanced_accuracy_score(test_class_labels, predicted_test_labels))
        pos_instances = {"train_class1": class1_pos_instances, "train_class2": class2_pos_instances,
                         "test_class1": test_class1_pos_instances, "test_class2": test_class2_pos_instances}
        detailed_results[f"split_{i}"] = {"significant_pairs": significant_pairs, "pos_inst_counts": pos_inst_counts,
                                          "class_labels": class_labels, "test_pos_inst_counts": test_pos_inst_counts,
                                          "test_class_labels": test_class_labels,
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
                                  k=config['k'],
                                  num_processes=config['num_processes'],
                                  group_field=config['group_field'],
                                  n_splits=config['n_splits'],
                                  pval_threshold=config['pval_threshold'])
    write_nested_dict_to_files(data=detailed_results, path=config['output_path'], delimiter="\t")
