_base_ = ["./semseg-pt-v3m1-1-rpe.py"]

# low-memory test preset for 4GB GPUs
save_path = "exp/s3dis/ptv3-rpe-lowmem"
weight = "exp/s3dis/ptv3-rpe/model/model_best.pth"
resume = False
evaluate = True
test_only = False
num_worker = 0
empty_cache = True
enable_amp = True

data = dict(
    test=dict(
        split="test_one.json",
        # test_cfg=dict(
        #     voxelize=dict(grid_size=0.02),                          # 与训练一致
        #     # crop=dict(
        #     #     _delete_=True,
        #     #     type="SlidingWindowCropAll",
        #     #     window_size=(5.0, 5.0),
        #     #     stride=(2.5, 2.5),
        #     #     min_points=512,
        #     #     point_max=12000,
        #     #     use_z=False,
        #     # ),
        #     post_transform=[
        #         dict(type="CenterShift", apply_z=False),
        #         dict(type="ToTensor"),
        #         dict(
        #             type="Collect",
        #             keys=("coord", "grid_coord", "index"),
        #             feat_keys=("color", "normal"),
        #         ),
        #     ],
        #     # aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        # ),
    )
)
