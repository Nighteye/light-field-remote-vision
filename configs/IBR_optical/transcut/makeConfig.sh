for i in {1..7}
do
	for j in {1..7}
	do
		touch obj"$i"scn"$j"
		echo 'config 			./configs/IBR_optical/transcut/common_parameters

# INPUT
imageName		./in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/%u_rect.ppm

# OUTPUT
outdir			out/IBR_optical/transcut/obj'$i'_scn'$j'' >obj"$i"scn"$j"
	done
done
