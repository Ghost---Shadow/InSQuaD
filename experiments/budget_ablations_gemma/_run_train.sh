python src/train.py --config=experiments/budget_ablations_gemma/quaild_gain_fl_mpnet_gemma_100.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/budget_ablations_gemma/quaild_gain_gc_mpnet_gemma_100.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/budget_ablations_gemma/quaild_gain_ld_mpnet_gemma_100.yaml
source devops/upload_all_checkpoints.sh


source devops/stop_current_gcp_instance.sh
