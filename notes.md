# Notes for the computation graph

pixel -> EIG forward process:

dL_dpixel -> dL_dchannel -> dL_dalpha -> dL_dG -> cur_dL_dmean2D
                                               -> cur_dL_dconic2D
                         -> cur_dL_dcolors 
                                      -> cur_dL_dopacity
dL_dpixel_depth -> dL_dalpha
dL_dpixel_depth -> cur_dL_ddepth

cur_dL_dconic2D -> computeCov2DCUDARelocated() -> cur_dL_dmeans, cur_dL_dcov3D, dL_dtau

cur_dL_dmean2D, cur_dL_ddepth, cur_dL_dcolors -> preprocessCUDARelocated():
    dL_dmean2D -> dL_dmeans
    dL_dmean2d -> dL_dt -> dL_dtau
    cur_dL_ddepth -> dL_dmeans
    cur_dL_ddepth -> cur_dL_dtau
    dL_dcolor -> computeColorFromSH() -> dL_dsh, dL_dmeans, dL_dtau{
        dL_dcolor -> dL_dRGB -> dL_dsh
                     dL_dRGB -> dL_ddir -> dL_dmean -> dL_dtau
    }
    dL_dcov3Ds -> computeCov3D() -> cur_dL_dscale, cur_dL_drot 

-> cur_dL_dmeans, cur_dL_dcov3D, cur_dL_dscale, cur_dL_drot, dL_dtau


