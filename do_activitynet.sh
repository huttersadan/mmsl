collection=activitynet
visual_feature=i3d
exp_id=runs_0
root_path=/data/dtt/mssl/dataset
device_ids=3
# training
python method/train_dtt.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --device_ids 3