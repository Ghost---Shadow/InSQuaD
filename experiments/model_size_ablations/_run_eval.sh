python src/offline_eval.py --config=experiments/model_size_ablations/oracle_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/oracle_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/oracle_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_fl_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_fl_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_fl_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_gc_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_gc_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_gc_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_ld_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_ld_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/quaild_comb_ld_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/random_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/random_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/random_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/zeroshot_mpnet_davinci2.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/zeroshot_mpnet_gemma7b.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/model_size_ablations/zeroshot_mpnet_gptj6b.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
