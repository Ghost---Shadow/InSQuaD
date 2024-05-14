python src/offline_eval.py --config=experiments/tests/diversity_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/fastvotek_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/gc_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/ideal_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/leastconfidence_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/mfl_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/oracle_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/quaild_gc_mpnet_gpt2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/quaild_gc_mpnet_neo175.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/quaild_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/quaild_test_experiment_t5.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/quaild_test_similar.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/random_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/shortlistandtopk_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/votek_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/votek_test_experiment.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/tests/zeroshot_test_experiment.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
