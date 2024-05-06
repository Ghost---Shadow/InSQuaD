python src/offline_eval.py --config=experiments/main_table/leastconfidence_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/quaild_gain_gc_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/quaild_nt_gc_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/random_mpnet_stablelm.yaml
python src/offline_eval.py --config=experiments/main_table/zeroshot_mpnet_stablelm.yaml

source devops/stop_current_gcp_instance.sh
