import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
from itertools import product
import onnx


import argparse

iou_threshold = 0.3
list_min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
steps = [8, 16, 32, 64]
variance = torch.from_numpy(np.array([0.1, 0.2]))

def prior_gen(w, h):    
    feature_map_2th = [int(int((h + 1) / 2) / 2),
                        int(int((w + 1) / 2) / 2)]
    feature_map_3th = [int(feature_map_2th[0] / 2),
                        int(feature_map_2th[1] / 2)]
    feature_map_4th = [int(feature_map_3th[0] / 2),
                        int(feature_map_3th[1] / 2)]
    feature_map_5th = [int(feature_map_4th[0] / 2),
                        int(feature_map_4th[1] / 2)]
    feature_map_6th = [int(feature_map_5th[0] / 2),
                        int(feature_map_5th[1] / 2)]

    feature_maps = [feature_map_3th, feature_map_4th,
                    feature_map_5th, feature_map_6th]

    priors = []
    for k, f in enumerate(feature_maps):
        min_sizes = list_min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])): # i->h, j->w
            for min_size in min_sizes:
                s_kx = min_size / w
                s_ky = min_size / h

                cx = (j + 0.5) * steps[k] / w
                cy = (i + 0.5) * steps[k] / h

                priors.append([cx, cy, s_kx, s_ky])

    priors = torch.from_numpy(np.array(priors, dtype=np.float32))
    return priors
    

class YunetPostProcessing(nn.Module):
    def __init__(self, w, h, top_k, priors):
        super(YunetPostProcessing, self).__init__()
        self.top_k = top_k
        self.priors = priors

    def forward(self, loc, conf, iou):
        # loc.shape: Nx14
        # conf.shape: Nx2
        # iou.shape: Nx1

        # get score
        cls_scores = conf[:,1]
        iou_scores = torch.squeeze(iou, 1)
        # clamp
        iou_scores[iou_scores < 0.] = 0.
        iou_scores[iou_scores > 1.] = 1.
        scores = torch.sqrt(cls_scores * iou_scores)
        # scores.unsqueeze_(1) # Nx1

        # get bboxes
        bb_cx_cy = self.priors[:, 0:2] + loc[:, 0:2] * variance[0] * self.priors[:, 2:4]
        bb_wh_half = self.priors[:, 2:4] * torch.exp(loc[:, 2:4] * variance) * 0.5
        bb_x1_y1 = bb_cx_cy - bb_wh_half
        bb_x2_y2 = bb_cx_cy + bb_wh_half
        bboxes = torch.cat((bb_x1_y1, bb_x2_y2), dim=1).float()

        # get landmarks
        landmarks = torch.cat((
            self.priors[:, 0:2] + loc[:,  4: 6] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:,  6: 8] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:,  8:10] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:, 10:12] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:, 12:14] * variance[0] * self.priors[:, 2:4]),
            dim=1)

        # NMS
        keep_idx = nms(bboxes, scores, iou_threshold)[:self.top_k]

        # scores = scores.unsqueeze(1)[keep_idx]
        dets = torch.cat((bboxes[keep_idx], landmarks[keep_idx], scores[keep_idx].unsqueeze_(1)), dim=1)
        return dets


def test(w, h, top_k, priors):

    model = YunetPostProcessing(w, h, top_k, priors)
    N = priors.shape[0]
    loc = torch.randn(N, 14, dtype=torch.float)
    conf = torch.randn(N, 2, dtype=torch.float)
    iou = torch.randn(N, 1, dtype=torch.float)
    result = model(loc, conf, iou)
    print("Result shape:", result.shape)

def export_onnx(w, h, top_k, priors, onnx_name):
    """
    Exports the model to an ONNX file.
    """
    model = YunetPostProcessing(w, h, top_k, priors)
    N = priors.shape[0]
    loc = torch.randn(N, 14, dtype=torch.float)
    conf = torch.randn(N, 2, dtype=torch.float)
    iou = torch.randn(N, 1, dtype=torch.float)

    print(f"Generating {onnx_name}")
    torch.onnx.export(
        model,
        (loc, conf, iou),
        onnx_name,
        opset_version=11,
        do_constant_folding=True,
        # verbose=True,
        input_names=['loc', 'conf', 'iou'],
        output_names=['dets']
    )

def simplify(model):
    import onnxsim
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    print("Model has been simplified.")
    return model_simp

def patch_nms(model, top_k, score_thresh=None):
    import onnx_graphsurgeon as gs  
    import struct
    graph = gs.import_onnx(model)

    # Search for NMS node
    nms_not_found = True
    for node in graph.nodes:
        if node.op == "NonMaxSuppression":
            # print(vars(node))
            print("NonMaxSuppression found.")
            # Inputs of NonMaxSuppression:
            # 0: boxes
            # 1: scores
            # 2: max_output_boxes_per_class (opt)
            # 3: iou_threshold
            # 4: score_threshold

            # mobpc = max_out_boxes_per_class
            mobpc_input = node.inputs[2]
            # print(mobpc_input)
            # print(vars(mobpc_input))
            # print(mobpc_input._values)
            # print(vars(mobpc_input._values))
            mobpc = mobpc_input._values.tensor.raw_data
            mobpc = struct.unpack("q", mobpc)
            print("Current value of max_out_boxes_per_class", mobpc)
            new_mobpc = top_k
            mobpc = struct.pack("q", new_mobpc)
            mobpc_input._values.tensor.raw_data = mobpc
            print(f"max_out_boxes_per_class value changed to {top_k}")
            nms_not_found = False
            if score_thresh is not None:
                score_threshold_input = gs.Constant("_score_thresh", values=np.array(score_thresh, dtype=np.float32))
                node.inputs.append(score_threshold_input)
                print(f"score_threshold={score_thresh} added ")
                # print(vars(node.inputs[4]))

            break
    assert nms_not_found==False, "NonMaxSuppression could not be found in the graph !"
    graph.cleanup().toposort()
    print("NonMaxSuppression has benn patched")
    return gs.export_onnx(graph)

parser = argparse.ArgumentParser()
parser.add_argument('-W', type=int, required=True, help="yunet model input width")
parser.add_argument('-H', type=int, required=True, help="yunet model input height")
parser.add_argument('-top_k', type=int, default=50, help="max number of detections (default=%(default)i)")
parser.add_argument('-score_thresh', type=float, help="patch NMS op to use 'score_threshold' with given value")
parser.add_argument('-no_simp', action="store_true", help="do not run simplifier")
args = parser.parse_args()

w = args.W
h = args.H
top_k = args.top_k
run_simp = not args.no_simp

priors = prior_gen(w, h)

# test(w, h, top_k, priors)

raw_onnx_name = f"postproc_yunet_top{top_k}_{h}x{w}_raw.onnx"
export_onnx(args.W, args.H, args.top_k, priors, raw_onnx_name)

model = onnx.load(raw_onnx_name)
print("Model IR version:", model.ir_version)
if run_simp:
    model = simplify(model)
model = patch_nms(model, top_k, args.score_thresh)
print("Model IR version:", model.ir_version)



onnx_name = f"postproc_yunet_top{top_k}_{h}x{w}.onnx"
onnx.save(model, onnx_name)
print(f"Model saved in {onnx_name}")

