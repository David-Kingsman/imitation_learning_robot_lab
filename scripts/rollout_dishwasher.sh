python ./imitation_learning_robot_lab/scripts/rollout.py \
  --policy.path=outputs/models/act_dishwasher/checkpoints/014000/pretrained_model \
  --env.type=dishwasher \
  --policy.device=cuda \
  --policy.use_amp=false