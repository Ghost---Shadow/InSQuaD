python src/offline_eval.py --config=experiments/model_size_ablations/quaild_gain_fl_mpnet_davinci2.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/quaild_gain_fl_mpnet_llama7b.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/quaild_gain_gc_mpnet_davinci2.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/quaild_gain_gc_mpnet_llama7b.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/random_mpnet_davinci2.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/random_mpnet_llama7b.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/zeroshot_mpnet_davinci2.yaml
python src/offline_eval.py --config=experiments/model_size_ablations/zeroshot_mpnet_llama7b.yaml

source devops/stop_current_gcp_instance.sh
