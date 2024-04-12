from types import SimpleNamespace
import torch
from diff_eig import GaussianRasterizer
from diff_gaussian_rasterization_w_pose import GaussianRasterizer as OriginalGaussianRasterizer
import os


def main():

    path = "./test_dict.pt"

    test_dict = torch.load(path)
    raster_settings = SimpleNamespace(**test_dict['raster_settings'])
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    call_dict = test_dict["call_dict"]
    ref_hessian_dict = test_dict['H']

    # param_names = ['means3D', 'means2D', 'shs', 'opacities', 'scales', 'rotations']
    # for name in param_names:
    #     call_dict[name].requires_grad_()
    for k, v in call_dict.items():
        if v is not None:
            v.retain_grad()

    # im, depth, radii = rasterizer(**call_dict)
    im, radii, depth, opacity, n_touched = rasterizer(**call_dict)

    render_result = torch.cat([im, depth], dim=0)
    render_result.backward(torch.ones_like(render_result))
    # hessian_dict = {}
    # for name in ['means3D', 'shs', 'opacities', 'scales', 'rotations']:
    #     hessian_dict[name] = call_dict[name].grad
    for k in ref_hessian_dict.keys():
        if torch.allclose(ref_hessian_dict[k], call_dict[k].grad):
            print(f"{k} ✅")
        else:
            norm = torch.norm(ref_hessian_dict[k] - call_dict[k].grad)
            avg_diff = norm / ref_hessian_dict[k].numel()
            print(f"{k} norm: {norm.item()}, avg_diff: {avg_diff.item()}")
    



if __name__ == "__main__":
    main()
