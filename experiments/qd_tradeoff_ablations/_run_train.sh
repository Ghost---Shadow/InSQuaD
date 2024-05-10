python src/train.py --config=experiments/qd_tradeoff_ablations/quaild_gain_fl_mpnet_stablelm_lambda_0.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/qd_tradeoff_ablations/quaild_gain_fl_mpnet_stablelm_lambda_1.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/qd_tradeoff_ablations/quaild_gain_gc_mpnet_stablelm_lambda_0.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/qd_tradeoff_ablations/quaild_gain_gc_mpnet_stablelm_lambda_1.yaml
source devops/upload_all_checkpoints.sh


source devops/stop_current_gcp_instance.sh
