######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
# python main.py  -c config/config_DCN_CosineAnnealing_LR.yaml  -m DCNV3  -w True  -r DCNV3_CosineAnnealing_LR
# python main.py  -c config/config_DCN_Exponential_LR.yaml      -m DCNV3  -w True  -r DCNV3_Exponential_LR
python main.py  -c config/config_DCN_MultiStep_LR.yaml        -m DCNV3  -w True  -r DCNV3_MultiStep_LR
python main.py  -c config/config_DCN_ReduceLROnPlateau.yaml   -m DCNV3  -w True  -r DCNV3_educeLROnPlateau
python main.py  -c config/config_DCN_step_LR.yaml             -m DCNV3  -w True  -r DCNV3_step_LR