python src/offline_eval.py --config=experiments/tests/fastvotek_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/ideal_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/leastconfidence_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/quaild_gc_mpnet_gpt2.yaml
python src/offline_eval.py --config=experiments/tests/quaild_gc_mpnet_neo175.yaml
python src/offline_eval.py --config=experiments/tests/quaild_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/quaild_test_experiment_t5.yaml
python src/offline_eval.py --config=experiments/tests/random_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/shortlistandtopk_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/votek_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/tests/votek_test_experiment.yaml
python src/offline_eval.py --config=experiments/tests/zeroshot_test_experiment.yaml

source devops/stop_current_gcp_instance.sh
