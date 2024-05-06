python src/offline_eval.py --config=experiments/budget_ablations/quaild_gain_fl_mpnet_stablelm_100.yaml
python src/offline_eval.py --config=experiments/budget_ablations/quaild_gain_gc_mpnet_stablelm_100.yaml
python src/offline_eval.py --config=experiments/budget_ablations/random_mpnet_stablelm_100.yaml

source devops/stop_current_gcp_instance.sh
