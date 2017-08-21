# "DATASET_SCALE_VIEW"

#echo '######################'
#echo 'FOUNTAIN dataset'
#echo '######################'

#mkdir ./in/strecha_datasets/fountain_coco/fountain_altered_0_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_0_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_1_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_2_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_0to3_5
#mkdir ./in/strecha_datasets/fountain_coco/fountain_0to5_5

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0_5_fromMVE2coco_altered
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0_5_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_1_5_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_2_5_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0to3_5_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/fountain_0to5_5_fromMVE2coco

#echo '######################'
#echo 'CASTLE dataset'
#echo '######################'

#mkdir ./in/strecha_datasets/castle_coco/castle_0_6
#mkdir ./in/strecha_datasets/castle_coco/castle_1_6
#mkdir ./in/strecha_datasets/castle_coco/castle_2_6

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/castle_0_6_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/castle_1_6_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/castle_2_6_fromMVE2coco

#echo '######################'
#echo 'HERZJESU dataset'
#echo '######################'

#mkdir ./in/strecha_datasets/fountain_coco/herzjesu_0_4_altered
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_0_4
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_1_4
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_2_4
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_0to3_4
#mkdir ./in/strecha_datasets/herzjesu_coco/herzjesu_0to5_4

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0_4_fromMVE2coco_altered
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0_4_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_1_4_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_2_4_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0to3_4_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/herzjesu_0to5_4_fromMVE2coco

#echo '######################'
#echo 'HERCULES dataset'
#echo '######################'

#mkdir ./in/greg_datasets/hercule_coco/hercule_0_0
#mkdir ./in/greg_datasets/hercule_coco/hercule_1_0
#mkdir ./in/greg_datasets/hercule_coco/hercule_2_0

#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/hercule_0_0_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/hercule_1_0_fromMVE2coco
#./ULF -algorithm fromMVE2coco -config configs/gradientIBR/hercule_2_0_fromMVE2coco

echo '######################'
echo 'PLANE dataset'
echo '######################'

mkdir $HOME/datasets/greg_datasets/plane_coco

./ULF -algorithm fromMVE2coco -config configs/IBR_optical/plane_24_3360_2240


