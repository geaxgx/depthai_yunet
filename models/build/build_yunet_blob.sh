# This script has to be run from the docker container started by ./docker_openvino2tensorflow.sh

usage ()
{
	echo -e "Usage:\nGenerate a new YuNet blob with a specified model input resolution and number of shaves"
	echo -e "\nUsage: ${0} -r HxW [-s nb_shaves]"
	echo -e "\nHxW: height x width, example 120x160"
	echo -e "nb_shaves must be between 1 and 13. If not specified, default=4\n"
}

while getopts ":hr:s:" opt; do
	case ${opt} in
		h )
			usage
			exit 0
			;;
		r )
			H=$(echo $OPTARG | sed 's/x.*//')
            W=$(echo $OPTARG | sed 's/.*x//')
			;;
		s )
			nb_shaves=$OPTARG
			;;
		: )
			echo "Error: -$OPTARG requires an argument."
			usage
			exit 1
			;;
		\? )
			echo "Invalid option: -$OPTARG" 
			usage
			exit 1
			;;
	esac
done

nb_shaves=${nb_shaves:-4}
if [ $nb_shaves -lt 1 -o $nb_shaves -gt 13 ]
then
	echo "Invalid number of shaves !"
	usage
	exit 1
fi
# Check $H and $W are integers
if [[ ! ( $H =~ ^-?[0-9]+$  && $W =~ ^-?[0-9]+$ ) ]]
then
	usage
	exit 1
fi


MODEL=face_detection_yunet

# Optimize onnx model with chosen resolution
# ONNX Simplifier: pip3 install onnx-simplifier
python3 -m onnxsim ${MODEL}.onnx face_detection_yunet_${H}x${W}.onnx --input-shape 1,3,${H},${W}                                                                                                                                 
if [ $? -ne 0 ]
then
	echo "Exit on error !!!"	
	exit 1
fi

$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir openvino/FP16
mkdir -p openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES $nb_shaves \
-VPU_NUMBER_OF_CMX_SLICES $nb_shaves \
-o openvino/myriad/${MODEL}_${H}x${W}_sh${nb_shaves}.blob

if [ -f openvino/myriad/${MODEL}_${H}x${W}_sh${nb_shaves}.blob ]
then
	cp -p openvino/myriad/${MODEL}_${H}x${W}_sh${nb_shaves}.blob ..
	if [ $? -eq 0 ]
	then
		echo "${MODEL}_${H}x${W}_sh${nb_shaves}.blob has been successfully copied in models"
	fi
fi
