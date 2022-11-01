collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
margin=0.1
exp_id=runs_0
root_path=/data/dtt/mssl/dataset
device_ids=4
# training
python method/train_dtt.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --q_feat_size $q_feat_size --margin $margin --device_ids 5