######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
python main.py  -c config/config_Many_ensemble.yaml  -m FM  -w True  -r FM
python main.py  -c config/config_Many_ensemble.yaml  -m FFM  -w True  -r FFM
python main.py  -c config/config_Many_ensemble.yaml -m NCF  -w True  -r NCF
python main.py  -c config/config_Many_ensemble.yaml -m WDN  -w True  -r WDN
python main.py  -c config/config_Many_ensemble.yaml -m DeepFM  -w True  -r DeepFM
python main.py  -c config/config_Many_ensemble.yaml -m DCNV2  -w True  -r DCNV2
python main.py  -c config/config_Many_ensemble.yaml -m DCNV3  -w True  -r DCNV3