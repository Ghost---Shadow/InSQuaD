python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   --q_d_tradeoff_lambda=0.5
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml --q_d_tradeoff_lambda=0.5
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   --q_d_tradeoff_lambda=0.0
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml --q_d_tradeoff_lambda=0.0
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   --q_d_tradeoff_lambda=1.0
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml --q_d_tradeoff_lambda=1.0
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   --q_d_tradeoff_lambda=0.25
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml --q_d_tradeoff_lambda=0.25
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_nt_fl_mpnet_stablelm.yaml   --q_d_tradeoff_lambda=0.75
python src/gain_analysis_scripts/maximum_achieveable_f1.py --config=experiments/main_table/quaild_gain_fl_mpnet_stablelm.yaml --q_d_tradeoff_lambda=0.75


python src/gain_analysis_scripts/plot_maximum_achieveable_f1.py
