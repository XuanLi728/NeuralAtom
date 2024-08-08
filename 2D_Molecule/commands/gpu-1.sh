for SEED in {0..3}; do
    python main.py --cfg configs/GVM/peptides-func-GVM.yaml device cuda:6 seed $SEED 
done

for SEED in {0..3}; do
    python main.py --cfg configs/GVM/peptides-struct-GVM.yaml device cuda:6 seed $SEED
done
