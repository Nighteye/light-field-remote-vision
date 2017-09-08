# ESTIMATE MODELS AND RENDER NOVEL VIEWS

for i in {1..7}
do
	for j in {1..7}
	do
		config_file='configs/IBR_optical/transcut/obj'$i'scn'$j
		./ULF -config $config_file
	done
done

# CONVERT IMAGE SEQUENCE TO VIDEO

for i in {1..7}
do
	for j in {1..7}
	do
		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning3param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning3param_02_02.mp4'
		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning4param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning4param_02_02.mp4'
		ffmpeg -r 60 -f image2 -s 629x469 -i 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning6param_02_02_%03d.png' -vcodec libx264 -crf 15 'out/IBR_optical/transcut/obj'$i'_scn'$j'/panning6param_02_02.mp4'		
	done
done











