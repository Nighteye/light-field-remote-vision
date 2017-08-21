# COMPILE PSNR FOR EVALUATION
./compile_psnr_main.sh

# BUDDHA

./unstructured_lf -config configs/pami/buddha_comp
./unstructured_lf -config configs/pami/buddha_zero
./unstructured_lf -config configs/pami/buddha_5x5

# STILL

./unstructured_lf -config configs/pami/still_comp
./unstructured_lf -config configs/pami/still_zero
./unstructured_lf -config configs/pami/still_5x5

# MARIA

./unstructured_lf -config configs/pami/maria_comp
./unstructured_lf -config configs/pami/maria_zero
./unstructured_lf -config configs/pami/maria_5x5

# COUPLE

./unstructured_lf -config configs/pami/couple_comp
./unstructured_lf -config configs/pami/couple_zero
./unstructured_lf -config configs/pami/couple_5x5

# TRUCK

./unstructured_lf -config configs/pami/truck_comp
./unstructured_lf -config configs/pami/truck_zero
./unstructured_lf -config configs/pami/truck_5x5

# GUM

./unstructured_lf -config configs/pami/gum_nuts_comp
./unstructured_lf -config configs/pami/gum_nuts_zero
./unstructured_lf -config configs/pami/gum_nuts_5x5

# TAROT FINE

./unstructured_lf -config configs/pami/tarot_fine_comp
./unstructured_lf -config configs/pami/tarot_fine_zero
./unstructured_lf -config configs/pami/tarot_fine_5x5

# EVALUATION

echo buddha comp
./psnr ../lightfields/out/disparity_epi/buddha/center_view.png out/pami/buddha/comp/buddha_comp.png
echo buddha zero
./psnr ../lightfields/out/disparity_epi/buddha/center_view.png out/pami/buddha/zero/buddha_zero.png
echo buddha 5x5
./psnr ../lightfields/out/disparity_epi/buddha/center_view.png out/pami/buddha/5x5/buddha_5x5.png
echo stillLife comp
./psnr ../lightfields/out/disparity_epi/stillLife/center_view.png out/pami/stillLife/comp/stillLife_comp.png
echo stillLife zero
./psnr ../lightfields/out/disparity_epi/stillLife/center_view.png out/pami/stillLife/zero/stillLife_zero.png
echo stillLife 5x5
./psnr ../lightfields/out/disparity_epi/stillLife/center_view.png out/pami/stillLife/5x5/stillLife_5x5.png
echo maria comp
./psnr ../lightfields/out/disparity_epi/maria/center_view.png out/pami/maria/comp/maria_comp.png
echo maria zero
./psnr ../lightfields/out/disparity_epi/maria/center_view.png out/pami/maria/zero/maria_zero.png
echo maria 5x5
./psnr ../lightfields/out/disparity_epi/maria/center_view.png out/pami/maria/5x5/maria_5x5.png
echo couple comp
./psnr ../lightfields/out/disparity_epi/couple/center_view.png out/pami/couple/comp/couple_comp.png
echo couple zero
./psnr ../lightfields/out/disparity_epi/couple/center_view.png out/pami/couple/zero/couple_zero.png
convert ../lightfields/out/disparity_epi/couple/center_view.png -crop 897x897+0+0 ../lightfields/out/disparity_epi/couple/center_view_crop.png
echo couple 5x5
./psnr ../lightfields/out/disparity_epi/couple/center_view_crop.png out/pami/couple/5x5/couple_5x5.png
echo truck comp
./psnr ../lightfields/in/stanford/truck_lf/lowRes_centerView.png out/pami/truck_lf/comp/truck_lf_comp.png
echo truck zero
./psnr ../lightfields/in/stanford/truck_lf/lowRes_centerView.png out/pami/truck_lf/zero/truck_lf_zero.png
convert ../lightfields/in/stanford/truck_lf/lowRes_centerView.png -crop 639x480+0+0 ../lightfields/in/stanford/truck_lf/lowRes_centerView_crop.png
echo truck 5x5
./psnr ../lightfields/in/stanford/truck_lf/lowRes_centerView_crop.png out/pami/truck_lf/5x5/truck_lf_5x5.png
echo gum nuts comp
./psnr ../lightfields/in/stanford/gum_nuts_lf/lowRes_centerView.png out/pami/gum_nuts_lf/comp/gum_nuts_lf_comp.png
echo gum nuts zero
./psnr ../lightfields/in/stanford/gum_nuts_lf/lowRes_centerView.png out/pami/gum_nuts_lf/zero/gum_nuts_lf_zero.png
convert ../lightfields/in/stanford/gum_nuts_lf/lowRes_centerView.png -crop 639x768+0+0 ../lightfields/in/stanford/gum_nuts_lf/lowRes_centerView_crop.png
echo gum nuts 5x5
./psnr ../lightfields/in/stanford/gum_nuts_lf/lowRes_centerView_crop.png out/pami/gum_nuts_lf/5x5/gum_nuts_lf_5x5.png
echo tarot fine comp
./psnr ../lightfields/in/stanford/tarot_fine_lf/lowRes_centerView.png out/pami/tarot_fine_lf/comp/tarot_fine_lf_comp.png
echo tarot fine zero
./psnr ../lightfields/in/stanford/tarot_fine_lf/lowRes_centerView.png out/pami/tarot_fine_lf/zero/tarot_fine_lf_zero.png
convert ../lightfields/in/stanford/tarot_fine_lf/lowRes_centerView.png -crop 510x510+0+0 ../lightfields/in/stanford/tarot_fine_lf/lowRes_centerView_crop.png
echo tarot fine 5x5
./psnr ../lightfields/in/stanford/tarot_fine_lf/lowRes_centerView_crop.png out/pami/tarot_fine_lf/5x5/tarot_fine_lf_5x5.png

