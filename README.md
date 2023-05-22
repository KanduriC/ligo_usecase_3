## Instructions for reproducing the results of usecase-3

In this case study, the following tools were used for simulations and training machine learning models: Ligo was used for simulations, immuneML (https://immuneml.uio.no/) was used for running the logistic regression model and an adhoc implementation of a multiple instance learning (MIL) -based classifier included in this repository under `ligo_usecase_3/` was used.

All the configuration scripts used for Ligo or immuneML or the adhoc MIL classifier are under `simulation_and_ML/` in this repository. Each experiment was independently replicated three times and the respective configuration files are placed under directory structure reflecting that {run1, run2, run3}. The experiments were carried out at two different witness rates that were reflected in the filenames of the configuration files. 

Assuming that Ligo, immuneML and this repository are installed using pip, below are examples of the commands of Ligo, immuneML and MIL classifier for repeating the experiments:

Ligo usage:

```
ligo interaction_witnessrate_0.01%.yaml witnessrate_0.0001
```

immuneML usage:

```
immune-ml logistic_witnessrate_0.01%.yaml logistic_witnessrate_0.0001
```

The MIL classifier is intended to first identify the positive instances (receptor sequences that distinguish labels) across all repertoires and then identify the bag labels based on pooling information from instance-level labels. Thus, this implementation of MIL classifier takes in a single file of receptor sequences that is a concatenated version of all the AIRR exported files from Ligo. The `concat_airr` command-line utility function included in this package generates such concatenated file. The usage is shown below:

```
concat_airr --metadata_file metadata.csv --repertoire_concat_file concatenated_flat_receptor_file.tsv
```

MIL classifier usage:

```
mil_classifier --config_file mil_config_witnessrate_0.01%.yaml
```