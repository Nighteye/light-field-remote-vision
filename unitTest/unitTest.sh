./ULF -config configs/IBR_optical/unitTest

#for i in {0..24}
#do
#    cp ../out/IBR_optical/unitTest/ref_flow$(printf %02d $i).pfm ../out/IBR_optical/unitTest/flow$(printf %02d $i).pfm
#done

cp ../out/IBR_optical/transcut/obj1_scn1/flow$(printf %02d $i).pfm ../out/IBR_optical/unitTest/ref_flow$(printf %02d $i).pfm

for i in {0..24}
do
    if [ $i != 12 ]
    then
        ./unitTest ../out/IBR_optical/unitTest/flow$(printf %02d $i).pfm ../out/IBR_optical/unitTest/ref_flow$(printf %02d $i).pfm
    fi
done

