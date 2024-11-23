python src/offline_eval.py --config=experiments/main_table_gemma/diversity_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/gc_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/leastconfidence_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/mfl_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/oracle_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_comb_fl_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_comb_gc_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_comb_ld_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_combnt_fl_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_combnt_gc_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/quaild_combnt_ld_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/random_mpnet_gemma.yaml
source devops/upload_artifacts.sh

python src/offline_eval.py --config=experiments/main_table_gemma/zeroshot_mpnet_gemma.yaml
source devops/upload_artifacts.sh


source devops/stop_current_gcp_instance.sh
