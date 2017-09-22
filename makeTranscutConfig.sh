for i in {1..7}
do
	for j in {1..7}
	do
		touch configs/IBR_optical/transcut/obj"$i"scn"$j"view12
		echo 'config 			./configs/IBR_optical/transcut/common_parameters

# VIEW RANGE
s_rmv			2
t_rmv			2

# INPUT
imageName		./in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/%u_rect.ppm

# OUTPUT
outdir			out/IBR_optical/transcut/obj'$i'_scn'$j'' >configs/IBR_optical/transcut/obj"$i"scn"$j"view12
	done
done

for i in {1..7}
do
	for j in {1..7}
	do
		touch configs/IBR_optical/transcut/obj"$i"scn"$j"view04
		echo 'config 			./configs/IBR_optical/transcut/common_parameters

# VIEW RANGE
s_rmv			4
t_rmv			0

# INPUT
imageName		./in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/%u_rect.ppm

# OUTPUT
outdir			out/IBR_optical/transcut/obj'$i'_scn'$j'' >configs/IBR_optical/transcut/obj"$i"scn"$j"view04
	done
done

for i in {1..7}
do
	for j in {1..7}
	do
		touch configs/IBR_optical/transcut/obj"$i"scn"$j"view22
		echo 'config 			./configs/IBR_optical/transcut/common_parameters

# VIEW RANGE
s_rmv			2
t_rmv			4

# INPUT
imageName		./in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/%u_rect.ppm

# OUTPUT
outdir			out/IBR_optical/transcut/obj'$i'_scn'$j'' >configs/IBR_optical/transcut/obj"$i"scn"$j"view22
	done
done

for i in {1..7}
do
	for j in {1..7}
	do
		touch configs/IBR_optical/transcut/obj"$i"scn"$j"allViews
		echo 'config 			./configs/IBR_optical/transcut/common_parameters

# VIEW RANGE
s_rmv			-1
t_rmv			-1

# INPUT
imageName		./in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/%u_rect.ppm

# OUTPUT
outdir			out/IBR_optical/transcut/obj'$i'_scn'$j'' >configs/IBR_optical/transcut/obj"$i"scn"$j"allViews
	done
done
