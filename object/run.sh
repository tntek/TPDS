# ====================  office  ==========================
# ---------- train source model -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 1
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 2

# ---------- train target model for adaptation -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 0 --t 2 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 1 --t 0 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 1 --t 2 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 2 --t 0 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --da uda --dset office --gpu_id 0 --s 2 --t 1 --output_src ckps/source/ --output ckps/target_tpds/


# =====================  office-home  =====================
# ---------- train source model -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 1
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 2
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 3

# ---------- train target model for adaptation -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 0 --t 2 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 0 --t 3 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 1 --t 0 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 1 --t 2 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 1 --t 3 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 2 --t 0 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 2 --t 1 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 2 --t 3 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 3 --t 0 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 3 --t 1 --output_src ckps/source/ --output ckps/target_tpds/
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py  --da uda --dset office-home --gpu_id 0 --s 3 --t 2 --output_src ckps/source/ --output ckps/target_tpds/

# =====================  VISDA-C  ========================
# ---------- train source model -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_src.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0

# ---------- train target model for adaptation -------------
~/anaconda3/envs/vit_kd/bin/python dou_train_tgt_tpds.py --batch_size 64 --da uda --dset VISDA-C --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target_tpds/ --net resnet101 --lr 1e-3 --seed 2020

