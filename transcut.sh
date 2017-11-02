# CREATE CONFIG FILES

#./makeTranscutConfig.sh

# ESTIMATE MODELS AND RENDER NOVEL VIEWS

for i in {1..7}
do
	for j in {1..7}
	do
		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/model_9p_LIN_12.pfm'
		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/model_9p_LIN_04.pfm'
		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/model_9p_LIN_22.pfm'
		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/model_9p_LIN_allViews.pfm'
		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view12'
		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view04'
		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view22'
		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'allViews'
	done
done

# UNIT TEST

#./ULF -config configs/IBR_optical/unitTest

# CONVERT IMAGE SEQUENCE TO VIDEO

#for i in {1..7}
#do
#	for j in {1..7}
#	do
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_panning.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_panning_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_panning.mp4'
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_panning.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_panning_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_panning.mp4'
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_panning.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_panning_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_panning.mp4'		
#	done
#done

# EVALUATE RESULT QUALITY

timestamp=$(date)
filename="psnr/results/$timestamp"
touch "$filename"

for i in {1..7}
do
	for j in {1..7}
	do
		psnr3g12=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_12.png')
		psnr4g12=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_12.png')
		psnr6g12=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_12.png')
		psnr3g04=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/4_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_04.png')
		psnr4g04=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/4_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_04.png')
		psnr6g04=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/4_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_04.png')
		psnr3g22=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/22_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_22.png')
		psnr4g22=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/22_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_22.png')
		psnr6g22=$(psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/22_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_22.png')

		echo 'obj'$i'/scn'$j >>"$filename"
		echo 'view 12' >>"$filename"
		echo $psnr3g12 >>"$filename"
		echo $psnr4g12 >>"$filename"
		echo $psnr6g12 >>"$filename"
		echo 'view 04' >>"$filename"
		echo $psnr3g04 >>"$filename"
		echo $psnr4g04 >>"$filename"
		echo $psnr6g04 >>"$filename"
		echo 'view 22' >>"$filename"
		echo $psnr3g22 >>"$filename"
		echo $psnr4g22 >>"$filename"
		echo $psnr6g22 >>"$filename"
	done
done









