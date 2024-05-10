python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml  --should_load_checkpoint
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_gc_mpnet_stablelm.yaml --should_load_checkpoint
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_gc_mpnet_stablelm.yaml 

python src/gain_analysis_scripts/plot_maximum_achieveable_f1.py
source devops/upload_artifacts.sh
