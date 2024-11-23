python src/offline_eval.py --config=experiments/budget_ablations_gemma/quaild_comb_fl_mpnet_gemma_100.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/budget_ablations_gemma/quaild_comb_gc_mpnet_gemma_100.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/budget_ablations_gemma/quaild_comb_ld_mpnet_gemma_100.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/budget_ablations_gemma/random_mpnet_gemma_100.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
