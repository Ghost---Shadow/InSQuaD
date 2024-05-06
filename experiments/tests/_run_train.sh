python src/train.py --config=experiments/tests/quaild_test_experiment.yaml
source devops/upload_artifacts.sh

python src/train.py --config=experiments/tests/quaild_test_experiment_t5.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
