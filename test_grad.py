from types import SimpleNamespace
import torch
from diff_eig import GaussianRasterizer
from diff_gaussian_rasterization_w_pose import GaussianRasterizer as OriginalGaussianRasterizer
import os


def main():

    path = "./test_dict.pt"

    test_dict = torch.load(path)
    raster_settings = SimpleNamespace(**test_dict['raster_settings'])
    raster_settings.gradient_power = 1

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    call_dict = test_dict["call_dict"]

    orig_rasterizer = OriginalGaussianRasterizer(raster_settings=raster_settings)
    orig_call_dict = {k: v for k, v in call_dict.items() if k != 'pose_H'}
    for k, v in orig_call_dict.items():
        if isinstance(v, torch.Tensor):
            orig_call_dict[k] = torch.nn.Parameter(v.detach().clone())

    orig_results = orig_rasterizer(**orig_call_dict)

    im, depth = orig_results[-5], orig_results[-3]
    render_result = torch.cat([im, depth], dim=0)
    render_result.backward(torch.ones_like(render_result))

    # ref_hessian_dict = test_dict['H']

    # param_names = ['means3D', 'means2D', 'shs', 'opacities', 'scales', 'rotations']
    # for name in param_names:
    #     call_dict[name].requires_grad_()
    for k, v in call_dict.items():
        if v is not None:
            v.grad = None
            v.retain_grad()

    # breakpoint()
    grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations, grad_theta, grad_rho, JTJ_tau,\
        im, radii, depth, opacity, n_touched = rasterizer(**call_dict)
    
    grads = (grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations, grad_theta, grad_rho)

    # render_result = torch.cat([im, depth], dim=0)
    # render_result.backward(torch.ones_like(render_result))

    param_names = ['means3D', 'shs', 'opacities', 'scales', 'rotations', 'theta', 'rho']

    # breakpoint()
    for k, v in zip(param_names, grads):
        if torch.allclose(orig_call_dict[k].grad, v):
            print(f"{k} ✅")
        else:
            norm = torch.norm(orig_call_dict[k].grad - v)
            avg_diff = norm / orig_call_dict[k].grad.numel()
            print(f"{k:10s} norm: {norm.item():.3f},\t avg_diff: {avg_diff.item():3f},\t\
ref_norm: {torch.norm(orig_call_dict[k].grad).item():.3f},\t cur_norm: {torch.norm(v).item():.3f}")

    # loss = grad_means3D.sum() + grad_opacities.sum()
    # print(loss)
    # loss.backward()

    # for param_name in param_names:
    #     print(f"{param_name} grad: {call_dict[param_name].grad}"    )
    breakpoint()
    



if __name__ == "__main__":
    main()
