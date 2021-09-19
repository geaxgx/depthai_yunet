#!/usr/bin/env python3

from YuNetRenderer import YuNetRenderer
import argparse
import sys

# List available models and their input resolution
# The name of a model is face_detection_yunet_{H}x{W}.blob
# where H and W are the model input height and weight
# Extract the resolutions and set them as the default choice list
# of --model_resolution argument
from glob import glob
import re
model_list = glob("models/face_detection_yunet_*.blob")
res_list = list(set([ re.sub(r'models/face_detection_yunet_(.*)_.*\.blob', r'\1', model) for model in model_list]))

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_det = parser.add_argument_group("Tracker arguments")
parser_det.add_argument('-myu', '--model_yunet', type=str, 
                    help="Path to Yunet blob")
parser_det.add_argument('-mpp', '--model_postproc', type=str, 
                    help="Path to Post Processing model blob (only in edge node)")

parser_det.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")

parser_det.add_argument("-mr", "--model_resolution", type=str, choices=res_list, default="180x320",
                    help="Select YuNet model input resolution (default=%(default)s)")
parser_det.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps")                    
# parser_det.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
#                     help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_det.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels") 
parser_det.add_argument('-s', '--sync', action="store_true",
                    help="Synchronize video frame and Yunet inference (only in Edge mode)")
parser_det.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()

if args.model_yunet is not None:
    model_yunet = args.model_yunet
else:
    # List the models having the specified resolution
    # If found exactly one, use that one
    # If more than one, let the user choose
    models = glob(f"models/face_detection_yunet_*{args.model_resolution}*.blob")
    if len(models) == 0:
        print(f"There is no YuNet model with resolution {args.model_resolution} !")
        sys.exit()
    elif len(models) == 1:
        model_yunet = models[0]
    else:
        print(f"There are several YuNet model with resolution {args.model_resolution}.")
        print("To select the model, enter its # :")
        for i, m in enumerate(models):
            print(f"{i+1}) {m}")
        print(f"Your selection [1] ? ", end='')
        sel = input()
        try:
            sel = 0 if sel == "" else int(sel) - 1
            model_yunet = models[sel]
        except:
            print("Invalid selection !")
            sys.exit()

if args.edge:
    # Selection of the Post Processing model
    if args.model_postproc is not None:
        model_postproc = args.model_postproc
    else:
        # List the models having the specified resolution
        # If found exactly one, use that one
        # If more than one, let the user choose
        models = glob(f"models/postproc_yunet_*{args.model_resolution}*.blob")
        if len(models) == 0:
            print(f"There is no Post Processing model with resolution {args.model_resolution} !")
            sys.exit()
        elif len(models) == 1:
            model_postproc = models[0]
        else:
            print(f"There are several Post Processing model with resolution {args.model_resolution}.")
            print("To select the model, enter its # :")
            for i, m in enumerate(models):
                print(f"{i+1}) {m}")
            print(f"Your selection [1] ? ", end='')
            sel = input()
            try:
                sel = 0 if sel == "" else int(sel) - 1
                model_postproc = models[sel]
            except:
                print("Invalid selection !")
                sys.exit()

dargs = vars(args)

detector_args = {a:dargs[a] for a in ['internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from YuNetEdge import YuNet
    detector_args["model_postproc"] = model_postproc
    detector_args["sync"] = args.sync
else:
    from YuNet import YuNet      
    

detector = YuNet(
        model=model_yunet,
        model_resolution=args.model_resolution,
        input_src=args.input,
        # resolution=args.resolution,
        stats=True,
        trace=args.trace,
        **detector_args
        )

renderer = YuNetRenderer(
        detector=detector,
        output=args.output)

while True:
    # Run face detector on next frame
    frame, faces = detector.next_frame()
    if frame is None: break
    # Draw faces
    frame = renderer.draw(frame, faces)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

detector.exit()
renderer.exit()
