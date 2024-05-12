python src/train.py --config=experiments/main_table_gemma/quaild_gain_fl_mpnet_gemma.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/main_table_gemma/quaild_gain_gc_mpnet_gemma.yaml
source devops/upload_all_checkpoints.sh


source devops/stop_current_gcp_instance.sh
