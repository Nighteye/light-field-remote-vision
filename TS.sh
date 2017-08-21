# "DATASET_SCALE_VIEW_alpha_beta_lambda"

#./fromMVE2coco.sh

#echo '######################'
#echo 'FOUNTAIN dataset'
#echo '######################'

# altered images
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_1_0_002_altered
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_01_1_002_altered

# view 5
# scale 0
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_5_01_1_002

# scale 1
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_1_5_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_1_5_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_1_5_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_1_5_01_1_002

# scale 2
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_2_5_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_2_5_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_2_5_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_2_5_01_1_002

# view 2
# scale 0

#mkdir ./in/strecha_datasets/fountain_coco/fountain_0_2

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0_2_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_2_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_2_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_2_01_1_002

# view 8 
# scale 0

#mkdir ./in/strecha_datasets/fountain_coco/fountain_0_8

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0_8_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_8_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_8_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_0_8_01_1_002

#echo '######################'
#echo 'HERZJESU dataset'
#echo '######################'

# altered images
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_1_0_002_altered
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_1_0_003_altered
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_01_1_002_altered

# view 4
# scale 0
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_4_01_1_002

# scale 1
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_1_4_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_1_4_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_1_4_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_1_4_01_1_002

# scale 2
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_2_4_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_2_4_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_2_4_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_2_4_01_1_002

# view 2
# scale 0

#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_0_2

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0_2_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_2_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_2_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_2_01_1_002

# view 6 
# scale 0

#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_0_6

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0_6_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_6_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_6_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_0_6_01_1_002

#echo '######################'
#echo 'CHARCE dataset'
#echo '######################'

# view 4
# scale 0

#mkdir -p ./in/greg_datasets/charce_coco/charce_0_4
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/charce_0_4_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_4_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_4_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_4_01_1_002

# view 6
# scale 0

#mkdir -p ./in/greg_datasets/charce_coco/charce_0_6
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/charce_0_6_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_6_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_6_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_6_01_1_002

# view 11 
# scale 0

#mkdir -p ./in/greg_datasets/charce_coco/charce_0_11
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/charce_0_11_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_11_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_11_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/charce_0_11_01_1_002

#echo '######################'
#echo 'LION dataset'
#echo '######################'

# view 1
# scale 0

#mkdir -p ./in/greg_datasets/lion_coco/lion_0_1
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/lion_0_1_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_1_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_1_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_1_01_1_002

# view 3
# scale 0

#mkdir -p ./in/greg_datasets/lion_coco/lion_0_3
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/lion_0_3_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_3_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_3_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_3_01_1_002

# view 5
# scale 0

#mkdir -p ./in/greg_datasets/lion_coco/lion_0_5
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/lion_0_5_fromMVE2coco

#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_5_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_5_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/lion_0_5_01_1_002

#echo '###### VIDEO ######'

#echo '######################'
#echo 'FOUNTAIN dataset'
#echo '######################'

#mkdir ./in/strecha_datasets/fountain_coco/fountain_small_0_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_medium_0_5

#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_small_0_5_1_0_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_small_0_5_01_1_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_medium_0_5_1_0_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/fountain_medium_0_5_01_1_002_video

#echo '######################'
#echo 'HERZJESU dataset'
#echo '######################'

#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_small_0_4
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_medium_0_4

#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_small_0_4_1_0_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_small_0_4_01_1_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_medium_0_4_1_0_002_video
#./ULF -algorithm IBR_color -config configs/gradientIBR/herzjesu_medium_0_4_01_1_002_video

echo '###### EVALUATION ######'

echo '######################'
echo 'FOUNTAIN dataset'
echo '######################'

# view 5

#echo "fountain view 5"
#../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_1_0_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_1_0_003.png
#../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_01_05_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_01_1_002.png

#../psnr in/strecha_datasets/fountain_coco/fountain_1_5/v_05.png out/gradientIBR/fountain/fountain_1_5/fountain_1_5_1_0_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_1_5/v_05.png out/gradientIBR/fountain/fountain_1_5/fountain_1_5_1_0_003.png
#../psnr in/strecha_datasets/fountain_coco/fountain_1_5/v_05.png out/gradientIBR/fountain/fountain_1_5/fountain_1_5_01_05_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_1_5/v_05.png out/gradientIBR/fountain/fountain_1_5/fountain_1_5_01_1_002.png

#../psnr in/strecha_datasets/fountain_coco/fountain_2_5/v_05.png out/gradientIBR/fountain/fountain_2_5/fountain_2_5_1_0_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_2_5/v_05.png out/gradientIBR/fountain/fountain_2_5/fountain_2_5_1_0_003.png
#../psnr in/strecha_datasets/fountain_coco/fountain_2_5/v_05.png out/gradientIBR/fountain/fountain_2_5/fountain_2_5_01_05_002.png
#../psnr in/strecha_datasets/fountain_coco/fountain_2_5/v_05.png out/gradientIBR/fountain/fountain_2_5/fountain_2_5_01_1_002.png

echo "fountain view 2"
../psnr in/strecha_datasets/fountain_coco/fountain_0_2/v_02.png out/gradientIBR/fountain/fountain_0_2/fountain_0_2_1_0_002.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_2/v_02.png out/gradientIBR/fountain/fountain_0_2/fountain_0_2_1_0_003.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_2/v_02.png out/gradientIBR/fountain/fountain_0_2/fountain_0_2_01_1_002.png

echo "fountain view 5"
../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_1_0_002.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_1_0_003.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_5/v_05.png out/gradientIBR/fountain/fountain_0_5/fountain_0_5_01_1_002.png

echo "fountain view 8"
../psnr in/strecha_datasets/fountain_coco/fountain_0_8/v_08.png out/gradientIBR/fountain/fountain_0_8/fountain_0_8_1_0_002.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_8/v_08.png out/gradientIBR/fountain/fountain_0_8/fountain_0_8_1_0_003.png
../psnr in/strecha_datasets/fountain_coco/fountain_0_8/v_08.png out/gradientIBR/fountain/fountain_0_8/fountain_0_8_01_1_002.png

echo '######################'
echo 'HERZJESU dataset'
echo '######################'

#echo "herzjesu view 5"

#echo "altered"
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_altered_0_4/herzjesu_altered_0_4_1_0_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_altered_0_4/herzjesu_altered_0_4_1_0_003.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_altered_0_4/herzjesu_altered_0_4_01_05_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_altered_0_4/herzjesu_altered_0_4_01_1_002.png

#echo "scale 0"
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_1_0_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_1_0_003.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_01_05_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_01_1_002.png

#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_1_4/v_04.png out/gradientIBR/herzjesu/herzjesu_1_4/herzjesu_1_4_1_0_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_1_4/v_04.png out/gradientIBR/herzjesu/herzjesu_1_4/herzjesu_1_4_1_0_003.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_1_4/v_04.png out/gradientIBR/herzjesu/herzjesu_1_4/herzjesu_1_4_01_05_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_1_4/v_04.png out/gradientIBR/herzjesu/herzjesu_1_4/herzjesu_1_4_01_1_002.png

#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_2_4/v_04.png out/gradientIBR/herzjesu/herzjesu_2_4/herzjesu_2_4_1_0_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_2_4/v_04.png out/gradientIBR/herzjesu/herzjesu_2_4/herzjesu_2_4_1_0_003.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_2_4/v_04.png out/gradientIBR/herzjesu/herzjesu_2_4/herzjesu_2_4_01_05_002.png
#../psnr in/strecha_datasets/herzjesu_coco/herzjesu_2_4/v_04.png out/gradientIBR/herzjesu/herzjesu_2_4/herzjesu_2_4_01_1_002.png

echo "herzjesu view 2"
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_2/v_02.png out/gradientIBR/herzjesu/herzjesu_0_2/herzjesu_0_2_1_0_002.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_2/v_02.png out/gradientIBR/herzjesu/herzjesu_0_2/herzjesu_0_2_1_0_003.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_2/v_02.png out/gradientIBR/herzjesu/herzjesu_0_2/herzjesu_0_2_01_1_002.png

echo "herzjesu view 4"
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_1_0_002.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_1_0_003.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_4/v_04.png out/gradientIBR/herzjesu/herzjesu_0_4/herzjesu_0_4_01_1_002.png

echo "herzjesu view 6"
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_6/v_06.png out/gradientIBR/herzjesu/herzjesu_0_6/herzjesu_0_6_1_0_002.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_6/v_06.png out/gradientIBR/herzjesu/herzjesu_0_6/herzjesu_0_6_1_0_003.png
../psnr in/strecha_datasets/herzjesu_coco/herzjesu_0_6/v_06.png out/gradientIBR/herzjesu/herzjesu_0_6/herzjesu_0_6_01_1_002.png

echo '######################'
echo 'CHARCE dataset'
echo '######################'

echo "charce view 4"
../psnr in/greg_datasets/charce_coco/charce_0_4/v_04.png out/gradientIBR/charce/charce_0_4/charce_0_4_1_0_002.png
../psnr in/greg_datasets/charce_coco/charce_0_4/v_04.png out/gradientIBR/charce/charce_0_4/charce_0_4_1_0_003.png
../psnr in/greg_datasets/charce_coco/charce_0_4/v_04.png out/gradientIBR/charce/charce_0_4/charce_0_4_01_1_002.png

echo "charce view 6"
../psnr in/greg_datasets/charce_coco/charce_0_6/v_06.png out/gradientIBR/charce/charce_0_6/charce_0_6_1_0_002.png
../psnr in/greg_datasets/charce_coco/charce_0_6/v_06.png out/gradientIBR/charce/charce_0_6/charce_0_6_1_0_003.png
../psnr in/greg_datasets/charce_coco/charce_0_6/v_06.png out/gradientIBR/charce/charce_0_6/charce_0_6_01_1_002.png

echo "charce view 11"
../psnr in/greg_datasets/charce_coco/charce_0_11/v_11.png out/gradientIBR/charce/charce_0_11/charce_0_11_1_0_002.png
../psnr in/greg_datasets/charce_coco/charce_0_11/v_11.png out/gradientIBR/charce/charce_0_11/charce_0_11_1_0_003.png
../psnr in/greg_datasets/charce_coco/charce_0_11/v_11.png out/gradientIBR/charce/charce_0_11/charce_0_11_01_1_002.png

echo '######################'
echo 'LION dataset'
echo '######################'

echo "lion view 1"
../psnr in/greg_datasets/lion_coco/lion_0_1/v_01.png out/gradientIBR/lion/lion_0_1/lion_0_1_1_0_002.png
../psnr in/greg_datasets/lion_coco/lion_0_1/v_01.png out/gradientIBR/lion/lion_0_1/lion_0_1_1_0_003.png
../psnr in/greg_datasets/lion_coco/lion_0_1/v_01.png out/gradientIBR/lion/lion_0_1/lion_0_1_01_1_002.png

echo "lion view 3"
../psnr in/greg_datasets/lion_coco/lion_0_3/v_03.png out/gradientIBR/lion/lion_0_3/lion_0_3_1_0_002.png
../psnr in/greg_datasets/lion_coco/lion_0_3/v_03.png out/gradientIBR/lion/lion_0_3/lion_0_3_1_0_003.png
../psnr in/greg_datasets/lion_coco/lion_0_3/v_03.png out/gradientIBR/lion/lion_0_3/lion_0_3_01_1_002.png

echo "lion view 5"
../psnr in/greg_datasets/lion_coco/lion_0_5/v_05.png out/gradientIBR/lion/lion_0_5/lion_0_5_1_0_002.png
../psnr in/greg_datasets/lion_coco/lion_0_5/v_05.png out/gradientIBR/lion/lion_0_5/lion_0_5_1_0_003.png
../psnr in/greg_datasets/lion_coco/lion_0_5/v_05.png out/gradientIBR/lion/lion_0_5/lion_0_5_01_1_002.png

#echo '###### IMAGETTE EVALUATION ######'

#echo '#########################'
#echo 'FOUNTAIN dataset (part 1)'
#echo '#########################'

#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_01_1_002_1.png

#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_01_1_002_1.png

#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_01_1_002_1.png

#echo '#########################'
#echo 'FOUNTAIN dataset (part 2)'
#echo '#########################'

#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_01_1_002_2.png

#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_01_1_002_2.png

#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_01_1_002_2.png

#echo '#########################'
#echo 'HERZJESU dataset (part 1)'
#echo '#########################'

#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_01_1_002_1.png

#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_01_1_002_1.png

#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_002_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_003_1.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_01_1_002_1.png

#echo '#########################'
#echo 'HERZJESU dataset (part 2)'
#echo '#########################'

#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_01_1_002_2.png

#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_01_1_002_2.png

#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_002_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_003_2.png
#../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_01_1_002_2.png




