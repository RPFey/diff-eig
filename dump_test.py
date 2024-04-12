from types import SimpleNamespace
import torch
# from diff_eig import GaussianRasterizer
from modified_diff_gaussian_rasterization_depth import GaussianRasterizer
import os


def main():

    path = "/tmp/fwd-dict.pt"
    if not os.path.exists(path): # didn't hit cache
        path = "/mnt/kostas_home/wen/fwd-dict.pt"

    test_dict = torch.load(path)
    raster_settings = SimpleNamespace(**test_dict['raster_settings'])
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    call_dict = test_dict["call_dict"]

    param_names = ['means3D', 'means2D', 'shs', 'opacities', 'scales', 'rotations']
    for name in param_names:
        call_dict[name].requires_grad_()

    rendered_image, depth, radii = rasterizer(**call_dict)
    rendered_image.backward(gradient=torch.ones_like(rendered_image))
    breakpoint()

    hessian_dict = {}
    for name in ['means3D', 'shs', 'opacities', 'scales', 'rotations']:
        hessian_dict[name] = call_dict[name].grad
    
    test_dict['H'] = hessian_dict
    torch.save(test_dict, "/mnt/kostas_home/wen/test-dict.pt")
    



if __name__ == "__main__":
    main()
