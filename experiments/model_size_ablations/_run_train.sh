python src/train.py --config=experiments/model_size_ablations/quaild_gain_fl_mpnet_davinci2.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_fl_mpnet_gemma7b.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_fl_mpnet_gptj6b.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_gc_mpnet_davinci2.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_gc_mpnet_gemma7b.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_gc_mpnet_gptj6b.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_ld_mpnet_davinci2.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_ld_mpnet_gemma7b.yaml
source devops/upload_all_checkpoints.sh

python src/train.py --config=experiments/model_size_ablations/quaild_gain_ld_mpnet_gptj6b.yaml
source devops/upload_all_checkpoints.sh


source devops/stop_current_gcp_instance.sh
