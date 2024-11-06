######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml  -m FFM  -w True  -r FFM_seed1 --seed 1
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml  -m FFM  -w True  -r FFM_seed2 --seed 2
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml  -m FFM  -w True  -r FFM_seed3 --seed 3
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml -m DCNV3  -w True  -r DCNV3_seed1 --seed 1
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml -m DCNV3  -w True  -r DCNV3_seed2 --seed 2
python main.py  -c config/config_FFM_DCNV3_ensemble.yaml -m DCNV3  -w True  -r DCNV3_seed3 --seed 3