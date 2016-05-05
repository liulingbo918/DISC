#!/usr/bin/env sh   
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib

########################################
## generate the test list 
########################################
ls DISC_Data/DISC_Input/ | grep jpg > DISC_Data/.DISC_test.list
sed -n 's/\.jpg//p' DISC_Data/.DISC_test.list > DISC_Data/DISC_test.list

########################################
## preprocess
## generate seg info
########################################
cd Seg_ERS/ers-master/
matlab -nodisplay -nosplash -nodesktop -r seg_demo
cd ../../


########################################
## computing saliency map
########################################
model=DISC_Net/DISC_Prototxt/DISC.prototxt
weights=DISC_Net/DISC_Model/DISC.caffemodel

gpu=0
iterations=`wc -l DISC_Data/DISC_test.list | cut -d ' ' -f 1`
log=DISC_Log/DISC_Test.log

DISC_Bin/tools/caffe test \
    --model=$model \
    --weights=$weights \
    --gpu=$gpu \
   --iterations=$iterations 2>&1 | tee $log

########################################
## remove temp files
######################################## 
rm DISC_Data/DISC_Input/*_seg.list


