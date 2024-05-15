python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table_gemma/quaild_gain_fl_mpnet_gemma.yaml  --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table_gemma/quaild_gain_gc_mpnet_gemma.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table_gemma/quaild_nt_fl_mpnet_gemma.yaml --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table_gemma/quaild_nt_gc_mpnet_gemma.yaml --limit=1000
source devops/upload_artifacts.sh

python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_fl_mpnet_gemma_lambda_025.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_gc_mpnet_gemma_lambda_025.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_fl_mpnet_gemma_lambda_0.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_gc_mpnet_gemma_lambda_0.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_fl_mpnet_gemma_lambda_1.yaml --should_load_checkpoint --limit=1000
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/qd_tradeoff_ablations_gemma/quaild_gain_gc_mpnet_gemma_lambda_1.yaml --should_load_checkpoint --limit=1000
source devops/upload_artifacts.sh

source devops/stop_current_gcp_instance.sh
