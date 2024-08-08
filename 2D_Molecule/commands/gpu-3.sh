SEED=3

python main.py --cfg configs/GVM/vocsuperpixels-GVM.yaml device cuda:$SEED seed $SEED

python main.py --cfg configs/GVM/peptides-func-GVM.yaml device cuda:$SEED seed $SEED

python main.py --cfg configs/GVM/peptides-struct-GVM.yaml device cuda:$SEED seed $SEED

# python main.py --cfg configs/GVM/pcqm-contact-GVM.yaml device cuda:$SEED

