python src/train.py --config=experiments/in_paper/quaild_gc_mpnet_stablelm.yaml
python src/train.py --config=experiments/in_paper/random_mpnet_stablelm.yaml
python src/train.py --config=experiments/in_paper/zeroshot_mpnet_stablelm.yaml

source devops/stop_current_gcp_instance.sh
