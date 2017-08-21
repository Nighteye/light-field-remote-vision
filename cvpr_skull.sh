
# run the algo with n_iter = 0 just to reconstruct the depth maps
./unstructured_lf -config configs/cvpr2016/skull_reconstruction

# a = alpha, b = gamma and c = lambda in the paper

# skull
./unstructured_lf -config configs/cvpr2016/skull_a0.1_b0.0_c0.001
./unstructured_lf -config configs/cvpr2016/skull_a0.1_b0.1_c0.001
./unstructured_lf -config configs/cvpr2016/skull_a0.0_b0.1_c0.001

#skull2
./unstructured_lf -config configs/cvpr2016/skull2_a0.1_b0.0_c0.001
./unstructured_lf -config configs/cvpr2016/skull2_a0.1_b0.1_c0.001
./unstructured_lf -config configs/cvpr2016/skull2_a0.0_b0.1_c0.001
