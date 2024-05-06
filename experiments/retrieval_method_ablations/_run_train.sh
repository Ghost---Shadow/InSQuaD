python src/train.py --config=experiments/retrieval_method_ablations/quaild_gain_fl_mpnet_stablelm.yaml
python src/train.py --config=experiments/retrieval_method_ablations/quaild_gain_gc_mpnet_stablelm.yaml

source devops/stop_current_gcp_instance.sh
