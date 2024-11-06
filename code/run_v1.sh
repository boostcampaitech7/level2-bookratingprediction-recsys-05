######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
python main.py  -c config/config_v1.yaml  -m FM  -w True  -r FM_v1_step_lr
python main.py  -c config/config_v1.yaml  -m FFM  -w True  -r FFM_v1_step_lr
python main.py  -c config/config_v1.yaml  -m DeepFM  -w True  -r DeepFM_v1_step_lr
python main.py  -c config/config_v1.yaml  -m WDN  -w True  -r WDN_v1_step_lr
python main.py  -c config/config_v1.yaml  -m DCN  -w True  -r DCN_v1_step_lr
python main.py  -c config/config_v1.yaml  -m DCNV2  -w True  -r DCN_v2_step_lr
# python main.py  -c config/config_v1.yaml  -m DCNV3  -w True  -r DCN_v3_step_lr
python main.py  -c config/config_v1.yaml  -m NCF  -w True  -r NCF_v1_step_lr
# python main.py  -c config/config_v1.yaml  -m Image_FM  -w True  -r Image_FM_v1_step_lr
# python main.py  -c config/config_v1.yaml  -m Image_DeepFM  -w True  -r Image_DeepFM_v1_step_lr
# python main.py  -c config/config_v1.yaml  -m Text_FM  -w True  -r Text_FM_v1_step_lr
# python main.py  -c config/config_v1.yaml  -m Text_DeepFM  -w True  -r Text_DeepFM_v1_step_lr
python main.py  -c config/config_v1.yaml  -m ResNet_DeepFM  -w True  -r ResNet_DeepFM_v1_step_lr

# python main.py  -c config/config_fm.yaml  -m Text_FM  -w True  -r Text_FM_v1
