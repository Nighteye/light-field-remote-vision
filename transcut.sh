# CREATE CONFIG FILES

#./makeTranscutConfig.sh

# ESTIMATE MODELS AND RENDER NOVEL VIEWS

#for i in {1..7}
#do
#	for j in {1..7}
#	do
#		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view12'
#		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view04'
#		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'view22'
#		./ULF -config 'configs/IBR_optical/transcut/obj'$i'scn'$j'allViews'
#	done
#done

# CONVERT IMAGE SEQUENCE TO VIDEO

#for i in {1..7}
#do
#	for j in {1..7}
#	do
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning3param_02_02.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning3param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning3param_02_02.mp4'
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning4param_02_02.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning4param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning4param_02_02.mp4'
#		rm 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning6param_02_02.mp4'
#		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning6param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning6param_02_02.mp4'		
#	done
#done

# EVALUATE RESULT QUALITY

for i in {1..7}
do
	for j in {1..7}
	do
		echo 'obj'$i'/scn'$j
		psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/3g_IBR_12.png'
		psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/4g_IBR_12.png'
		psnr/psnr 'in/transcut_dataset/obj'$i'/scn'$j'/images_25_629_469/12_rect.ppm' 'out/IBR_optical/transcut/obj'$i'_scn'$j'/6g_IBR_12.png'
	done
done









