collection=tvr
visual_feature=i3d_rgb_lgi
root_path=/data/dtt/mssl/dataset
model_dir=tvr-runs_0-2022_10_28_10_48_57

# training
#DATASET=charades
#FEATURE=i3d_rgb_lgi
#ROOTPATH=/data/dtt/mssl/dataset
#MODELDIR=checkpoint_charades

python method/eval_dtt.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir