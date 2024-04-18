python src/offline_eval.py --config=in_paper/zeroshot_gc_mpnet_stablelm.yaml
python src/offline_eval.py --config=in_paper/quaild_gc_mpnet_stablelm.yaml
python src/offline_eval.py --config=in_paper/random_gc_mpnet_stablelm.yaml

source devops/stop_current_gcp_instance.sh
