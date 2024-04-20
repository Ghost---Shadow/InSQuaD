# In Context Learning

## Installation

```sh
source ./devops/install.sh
```

## Linting

```sh
black .
```

## Tests

```sh
# Test everything (some tests may fail on windows)
python -m unittest discover -s src -p "*_test.py"

# Individual modules example
python -m unittest discover -s src.dataloaders -p "*_test.py"
python -m unittest discover -s src.dense_indexes -p "*_test.py"
python -m unittest discover -s src.shortlist_strategies -p "*_test.py"
python -m unittest discover -s src.subset_selection_strategies -p "*_test.py"
```

## Experiments

Make sure `./.env` is populated with correct keys

```sh
sh run_all_experiments.sh
```
