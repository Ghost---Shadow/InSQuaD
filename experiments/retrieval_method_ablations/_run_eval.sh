python src/offline_eval.py --config=experiments/retrieval_method_ablations/quaild_random_fl_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/retrieval_method_ablations/quaild_random_gc_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/retrieval_method_ablations/quaild_similar_fl_mpnet_stablelm.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/retrieval_method_ablations/quaild_similar_gc_mpnet_stablelm.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
