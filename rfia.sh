# "DATASET_SCALE_VIEW_alpha_beta_lambda"

#echo '######################'
#echo 'FOUNTAIN dataset'
#echo '######################'

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

#echo '######################'
#echo 'HERZJESU dataset'
#echo '######################'

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

#echo '######################'
#echo 'HERCULES dataset'
#echo '######################'

# scale 0
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_0_0_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_0_0_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_0_0_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_0_0_01_1_002

# scale 1
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_1_0_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_1_0_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_1_0_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_1_0_01_1_002

# scale 2
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_2_0_1_0_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_2_0_1_0_003
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_2_0_01_05_002
#./ULF -algorithm IBR_color -config configs/gradientIBR/hercule_2_0_01_1_002

#echo '###### EVALUATION ######'

#echo '######################'
#echo 'FOUNTAIN dataset'
#echo '######################'

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

#echo '######################'
#echo 'HERZJESU dataset'
#echo '######################'

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

#echo '######################'
#echo 'HERCULES dataset'
#echo '######################'

#../psnr in/greg_datasets/hercule_coco/hercule_0_0/v_00.png out/gradientIBR/hercule/hercule_0_0/hercule_0_0_1_0_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_0_0/v_00.png out/gradientIBR/hercule/hercule_0_0/hercule_0_0_1_0_003.png
#../psnr in/greg_datasets/hercule_coco/hercule_0_0/v_00.png out/gradientIBR/hercule/hercule_0_0/hercule_0_0_01_05_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_0_0/v_00.png out/gradientIBR/hercule/hercule_0_0/hercule_0_0_01_1_002.png

#../psnr in/greg_datasets/hercule_coco/hercule_1_0/v_00.png out/gradientIBR/hercule/hercule_1_0/hercule_1_0_1_0_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_1_0/v_00.png out/gradientIBR/hercule/hercule_1_0/hercule_1_0_1_0_003.png
#../psnr in/greg_datasets/hercule_coco/hercule_1_0/v_00.png out/gradientIBR/hercule/hercule_1_0/hercule_1_0_01_05_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_1_0/v_00.png out/gradientIBR/hercule/hercule_1_0/hercule_1_0_01_1_002.png

#../psnr in/greg_datasets/hercule_coco/hercule_2_0/v_00.png out/gradientIBR/hercule/hercule_2_0/hercule_2_0_1_0_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_2_0/v_00.png out/gradientIBR/hercule/hercule_2_0/hercule_2_0_1_0_003.png
#../psnr in/greg_datasets/hercule_coco/hercule_2_0/v_00.png out/gradientIBR/hercule/hercule_2_0/hercule_2_0_01_05_002.png
#../psnr in/greg_datasets/hercule_coco/hercule_2_0/v_00.png out/gradientIBR/hercule/hercule_2_0/hercule_2_0_01_1_002.png

echo '###### IMAGETTE EVALUATION ######'

echo '#########################'
echo 'FOUNTAIN dataset (part 1)'
echo '#########################'

../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_0_5_01_1_002_1.png

../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_1_5_01_1_002_1.png

../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/fountain_1.png ../papers/figuresRFIA/exper/fountain_2_5_01_1_002_1.png

echo '#########################'
echo 'FOUNTAIN dataset (part 2)'
echo '#########################'

../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_0_5_01_1_002_2.png

../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_1_5_01_1_002_2.png

../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/fountain_2.png ../papers/figuresRFIA/exper/fountain_2_5_01_1_002_2.png

echo '#########################'
echo 'HERZJESU dataset (part 1)'
echo '#########################'

../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_0_4_01_1_002_1.png

../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_1_4_01_1_002_1.png

../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_002_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_003_1.png
../psnr ../papers/figuresRFIA/exper/herzjesu_1.png ../papers/figuresRFIA/exper/herzjesu_2_4_01_1_002_1.png

echo '#########################'
echo 'HERZJESU dataset (part 2)'
echo '#########################'

../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_0_4_01_1_002_2.png

../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_1_4_01_1_002_2.png

../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_002_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_1_0_003_2.png
../psnr ../papers/figuresRFIA/exper/herzjesu_2.png ../papers/figuresRFIA/exper/herzjesu_2_4_01_1_002_2.png




