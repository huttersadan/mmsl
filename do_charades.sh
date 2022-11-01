collection=charades
visual_feature=i3d_rgb_lgi
clip_scale_w=0.5
frame_scale_w=0.5
exp_id=runs_0
root_path=/data/dtt/mssl/dataset
device_ids=4

# training

python method/train_dtt.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids 0

#RUN_ID=runs_0
#ROOTPATH=/data/dtt/PRVR/ms-sl-main/ms-sl-main/dataset/test_dataset
# GPU_DEVICE_ID=0
#sh do_charades.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
