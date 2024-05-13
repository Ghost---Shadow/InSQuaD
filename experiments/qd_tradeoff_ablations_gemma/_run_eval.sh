python src/offline_eval.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_fl_mpnet_gemma_lambda_0.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_fl_mpnet_gemma_lambda_1.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_gc_mpnet_gemma_lambda_0.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_gc_mpnet_gemma_lambda_1.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
