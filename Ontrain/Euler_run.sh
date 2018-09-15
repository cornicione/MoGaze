SOLVER=/home/mc/Caffe/Eyepose_solver.prototxt
WEIGHTS=/home/mc/Caffe/EyePose_ini.caffemodel

caffe train -solver $SOLVER -weights $WEIGHTS  -gpu 4,5,6,7 2>&1| tee /home/mc/Caffe/EyePose_913.log