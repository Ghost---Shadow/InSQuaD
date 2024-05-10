python src/train.py --config=experiments/budget_ablations/quaild_gain_fl_mpnet_stablelm_100.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/budget_ablations/quaild_gain_gc_mpnet_stablelm_100.yaml
source devops/upload_all_checkpoints.sh


source devops/stop_current_gcp_instance.sh
