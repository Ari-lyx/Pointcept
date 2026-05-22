#!/usr/bin/env python
import importlib
import os
import sys
import traceback


def section(name):
    print(f"\n== {name} ==")


def check(name, fn):
    try:
        result = fn()
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        raise SystemExit(1)
    if result is None:
        print(f"[ OK ] {name}")
    else:
        print(f"[ OK ] {name}: {result}")


def import_module(name, package=None):
    module = importlib.import_module(name, package=package)
    path = getattr(module, "__file__", "built-in")
    return path


def main():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    section("Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")

    section("Basic packages")
    modules = [
        "numpy",
        "h5py",
        "yaml",
        "torch",
        "torchvision",
        "torchaudio",
        "torch_geometric",
        "open3d",
        "spconv",
        "timm",
        "scipy",
        "plyfile",
        "wandb",
        "tensorboard",
        "einops",
        "addict",
        "colorama",
    ]
    for module_name in modules:
        check(f"import {module_name}", lambda m=module_name: import_module(m))

    section("Torch CUDA")

    def torch_cuda_check():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        return (
            f"torch={torch.__version__}, cuda={torch.version.cuda}, "
            f"gpu={torch.cuda.get_device_name(0)}, sum={y.sum().item():.3f}"
        )

    check("CUDA matmul", torch_cuda_check)

    section("Pointcept package")
    check("import pointcept", lambda: import_module("pointcept"))
    check("import pointcept.models", lambda: import_module("pointcept.models"))
    check("import pointcept.datasets", lambda: import_module("pointcept.datasets"))

    section("Pointops CUDA extension")

    def pointops_check():
        import torch
        from pointops.query import knn_query
        from pointops.sampling import farthest_point_sampling

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ).contiguous()
        offset = torch.tensor([xyz.shape[0]], device="cuda", dtype=torch.int32)
        new_offset = torch.tensor([2], device="cuda", dtype=torch.int32)

        fps_idx = farthest_point_sampling(xyz, offset, new_offset)
        idx, dist = knn_query(3, xyz, offset)
        torch.cuda.synchronize()

        if fps_idx.numel() != 2:
            raise RuntimeError(f"unexpected fps output shape: {tuple(fps_idx.shape)}")
        if idx.shape != (5, 3):
            raise RuntimeError(f"unexpected knn idx shape: {tuple(idx.shape)}")
        if dist.shape != (5, 3):
            raise RuntimeError(f"unexpected knn dist shape: {tuple(dist.shape)}")
        return f"fps_idx={fps_idx.detach().cpu().tolist()}, knn_shape={tuple(idx.shape)}"

    check("pointops fps + knn", pointops_check)

    section("PointGroup CUDA extension")

    def pointgroup_check():
        import torch
        from pointgroup_ops.functions.functions import ballquery_batch_p

        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            device="cuda",
            dtype=torch.float32,
        ).contiguous()
        batch_idxs = torch.zeros(coords.shape[0], device="cuda", dtype=torch.int32)
        batch_offsets = torch.tensor([0, coords.shape[0]], device="cuda", dtype=torch.int32)
        idx, start_len = ballquery_batch_p(coords, batch_idxs, batch_offsets, 0.05, 4)
        torch.cuda.synchronize()

        if start_len.shape != (coords.shape[0], 2):
            raise RuntimeError(f"unexpected start_len shape: {tuple(start_len.shape)}")
        return f"active={idx.numel()}, start_len_shape={tuple(start_len.shape)}"

    check("pointgroup ballquery", pointgroup_check)

    section("Done")
    print("Pointcept install check passed.")


if __name__ == "__main__":
    main()
