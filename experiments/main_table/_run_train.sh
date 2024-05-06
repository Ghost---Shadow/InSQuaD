python src/train.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/train.py --config=experiments/main_table/quaild_gain_gc_mpnet_stablelm.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
