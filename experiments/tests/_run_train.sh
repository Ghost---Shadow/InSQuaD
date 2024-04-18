python src/train.py --config=experiments/tests/dummy_experiment.yaml
python src/train.py --config=experiments/tests/quaild_gc_mpnet_gpt2.yaml
python src/train.py --config=experiments/tests/quaild_gc_mpnet_neo175.yaml
python src/train.py --config=experiments/tests/quaild_test_experiment.yaml
python src/train.py --config=experiments/tests/random_test_experiment.yaml
python src/train.py --config=experiments/tests/zeroshot_test_experiment.yaml

source devops/stop_current_gcp_instance.sh
