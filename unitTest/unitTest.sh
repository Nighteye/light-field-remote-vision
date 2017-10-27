# CREATE reference data from transcut/obj1_scn1

#rm ../out/IBR_optical/unitTest/
#cp ../out/IBR_optical/transcut/obj1_scn1/* ../out/IBR_optical/unitTest/
#cd ../out/IBR_optical/unitTest
#for f in *
#do
#    mv $f 'ref_'$f 
#done
#cd ../../../unitTest

# RUN tests

cd ..
./ULF -config configs/IBR_optical/unitTest
cd unitTest

# COMPARE with reference

for f in $(ls ../out/IBR_optical/unitTest/ | grep -v '^ref')
do
    if [ $f != 'config' ]
    then
        ./unitTest '../out/IBR_optical/unitTest/'$f '../out/IBR_optical/unitTest/ref_'$f
    fi
done
