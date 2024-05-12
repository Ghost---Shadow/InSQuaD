python src/offline_eval.py --config=experiments/main_table_stablelm/leastconfidence_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/oracle_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/quaild_gain_fl_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/quaild_gain_gc_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/quaild_nt_fl_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/quaild_nt_gc_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/random_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_stablelm/zeroshot_mpnet_stablelm.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
