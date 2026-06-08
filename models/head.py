# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.init as init
import torch.nn.functional as F
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto, Simam_module
from .conv import Conv, DWConv, autopad
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
from .rep_block import *

__all__ = ("Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect",
           'DetectDeepDBB','DetectWDBB','DetectV8','DetectAux',
            'Detect_LSCD', 'Segment_LSCD', 'Pose_LSCD', 'OBB_LSCD',
)



hier_names = {  #Fair_CSAR
    # Level 0: 
    "0": {
      "0": "Aircraft",
    },
    # Level 2: 
    "1": {
        "0": "Boeing",
        "1": "Airbus",
        "2": "Gulfstream",
        "3": "Fokker",
        "4": "Airfreighter",
        "5": "Helicopter",
        "6": "Other_Aircraft"
    },
    # Level 3: Specific aircraft model (ultimate fine-grained)
    "2": {
    "0": 'Airbus_A220',
    "1": 'Boeing777',
    "2": 'Boeing737',
    "3": 'Airbus_A330',
    "4": 'Airbus_A320',
    "5": 'Boeing767',
    "6": 'Helicopter',
    "7": 'Airfreighter',
    "8": 'Boeing747',
    "9": 'Fokker-50',
    "10": 'Gulfstream',
    "11": 'Other_Aircraft'
    }
}




import torch
import torch.nn as nn

class ECAModule(nn.Module):
    def __init__(self,  k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# Add different versions of the hierarchical architecture
class Detect_ORI(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:

            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Detect1(nn.Module):  # Hierarchical Arch Version 1
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        ####################
        if hier == {}:
              hier = hier_names

        self.hier = hier

        c2, c3 = max(
            (16, ch[0] // 4, self.reg_max * 4)), max(ch[0],
                                                     self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])

        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    prev_h_len = len(self.hier[str(k - 1)])
                    mod_list += [
                        nn.Sequential(Conv(x + prev_h_len, c3_h, 3),
                                      Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x + len(self.hier[str(len(self.hier) -2)]), c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3, len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        out_dict = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []

        for i in range(self.nl):
            for l in range(len(self.hier) - 1):
                if l == 0:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](x[i]))
                else:
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](
                        torch.cat((x[i], out_dict[f'cv3_level{l - 1}_out'][i]), 1)))

            out_dict[f"cv3_level{len(self.hier) - 1}_out"].append(
                self.cv3[len(self.hier) - 1][i](torch.cat(
                    (x[i], out_dict[f'cv3_level{len(self.hier) - 2}_out'][i]), 1)))
            x[i] = torch.cat(
                (self.cv2[i](x[i]), self.cv3[len(self.hier) - 1][i](torch.cat(
                    (x[i], out_dict[f'cv3_level{len(self.hier) - 2}_out'][i]), 1))), 1)

        if self.training:
            return [x, out_dict]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides
        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier) - 1}"] = y_last_level
        return y_last_level if self.export else (y_last_level , x)

 

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s) ** 2)


class Detect2(nn.Module):  # Hierarchical Arch Version 2
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        #####################
        if hier == {}:
              hier = hier_names

        self.hier = hier

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])
        self.eca = ECAModule()
        # self.Simam_module = Simam_module()
        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                c3_h = max(x // 4, h_len)
                # c3_h = c3
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    prev_h_len = len(self.hier[str(k - 1)])
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h + prev_h_len, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3 + len(self.hier[str(len(self.hier) - 2)]),
                              len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
                self.one2one_cv2 = copy.deepcopy(self.cv2)
                self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x, x_input):
        if self.end2end:
            return self.forward_end2end(x)

        # print(111)
        shape = x[0].shape  # BCHW
        out_dict = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
        for i in range(self.nl):
            for l in range(len(self.hier) - 1):
                if l == 0:
                    x_l0 = self.cv3[l][i][1](self.cv3[l][i][0](x[i]))
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i](x[i]))
                else:
                    out0 = self.cv3[l][i][0](x[i])
                    out1 = self.cv3[l][i][1](out0)
                    out_dict[f"cv3_level{l}_out"].append(self.cv3[l][i][2](
                        torch.cat((out1, self.eca(out_dict[f'cv3_level{l - 1}_out'][i])), 1)))
                        # torch.cat((out1, self.Simam_module(out_dict[f'cv3_level{l - 1}_out'][i])), 1)))
                        # torch.cat((out1, (out_dict[f'cv3_level{l - 1}_out'][i])), 1)))

            out_last0 = self.cv3[len(self.hier) - 1][i][0](x[i])
            out_last1 = self.cv3[len(self.hier) - 1][i][1](out_last0)
            out_dict[f"cv3_level{len(self.hier) - 1}_out"].append(
                self.cv3[len(self.hier) - 1][i][2](
                   torch.cat((out_last1, self.eca(out_dict[f'cv3_level{len(self.hier) - 2}_out'][i])), 1)))
                   # torch.cat((out_last1, self.Simam_module(out_dict[f'cv3_level{len(self.hier) - 2}_out'][i])), 1)))
                   # torch.cat((out_last1, (out_dict[f'cv3_level{len(self.hier) - 2}_out'][i])), 1)))
            x[i] = torch.cat((self.cv2[i](x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]

        y = self._inference(x, out_dict)

        return y if self.export else (y, x)



    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x, out_dict):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)

        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier) - 1}"] = y_last_level



        merged_cls = torch.cat([level2_cls, level1_cls, level0_cls], dim=1)

        final_output = torch.cat([dbox, merged_cls], dim=1)


        return final_output  #
        # return y_last_level  #
        # return y_hier[f"y_hier_level{2}"] # y_hier[f"y_hier_level{0}"] y_hier[f"y_hier_level{1}"]  y_hier[f"y_hier_level{2}"]

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s) ** 2)
    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)



class Detect3(nn.Module): ## Hierarchical Arch Version 3
    """YOLO Detect head v3  for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models


    def __init__(self, nc=80, ch=(), hier={}):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        #####################
        if hier == {}:
              hier = hier_names

        self.hier = hier
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList([])



        for k in range(len(self.hier) - 1):
            h_len = len(self.hier[str(k)])
            mod_list = []
            for x in ch:
                # c3_h = max(x // 4, h_len)
                c3_h = max(ch[0], h_len)
                if k == 0:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3), Conv(c3_h, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]
                else:
                    mod_list += [
                        nn.Sequential(Conv(x, c3_h, 3),
                                      Conv(c3_h * 2, c3_h, 3),
                                      nn.Conv2d(c3_h, h_len, 1))
                    ]

            self.cv3.append(nn.ModuleList(mod_list))

        self.cv3.append(
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3 * 2, c3, 3),
                    nn.Conv2d(c3, len(self.hier[str(len(self.hier) - 1)]), 1))
                for x in ch))


        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):

        shape = x[0].shape  # BCHW
        out_dict = {}
        in_2nd_conv = {}
        for l in range(len(self.hier)):
            out_dict[f"cv3_level{l}_out"] = []
            in_2nd_conv[f"cv3_level{l}_in_2nd_conv"] = []
        for i in range(self.nl):
            for l in range(len(self.hier)):
                if l == 0:
                    out0 = self.cv3[l][i][0](x[i])
                    in_2nd_conv[f"cv3_level{l}_in_2nd_conv"].append(out0)
                    out1 = self.cv3[l][i][1](out0)
                    out_dict[f"cv3_level{l}_out"].append(
                        self.cv3[l][i][2](out1))
                else:
                    out0 = self.cv3[l][i][0](x[i])
                    in_2nd_conv[f"cv3_level{l}_in_2nd_conv"].append(out0)
                    out1 = self.cv3[l][i][1](torch.cat(
                        (out0, in_2nd_conv[f"cv3_level{l}_in_2nd_conv"][i]),
                        1))
                    out_dict[f"cv3_level{l}_out"].append(
                        self.cv3[l][i][2](out1))
            x[i] = torch.cat((self.cv2[i](
                x[i]), out_dict[f"cv3_level{len(self.hier) - 1}_out"][-1]), 1)

        if self.training:
            return [x, out_dict]

        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat_last_level = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2)
        x_cat_hier = {}
        for l in range(len(self.hier) - 1):
            x_cat_hier[f"x_cat_level{l}"] = torch.cat([
                xi.view(shape[0], len(self.hier[str(l)]), -1)
                for xi in out_dict[f"cv3_level{l}_out"]
            ], 2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite',
                                           'edgetpu',
                                           'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat_last_level[:, :self.reg_max * 4]
            cls = x_cat_last_level[:, self.reg_max * 4:]
        else:
            box, cls = x_cat_last_level.split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True,
            dim=1) * self.strides

        y_last_level = torch.cat((dbox, cls.sigmoid()), 1)
        y_hier = {}
        for l in range(len(self.hier) - 1):
            y_hier[f"y_hier_level{l}"] = torch.cat(
                (dbox, x_cat_hier[f"x_cat_level{l}"].sigmoid()), 1)
        y_hier[f"y_hier_level{len(self.hier)-1}"] = y_last_level

        return y_last_level if self.export else (y_last_level , x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        for l in range(len(self.hier)):
            for a, b, s in zip(m.cv2, m.cv3[l], m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                nc_curr_level = len(self.hier[str(l)])
                b[-1].bias.data[:nc_curr_level] = math.log(5 / nc_curr_level /
                                                           (640 / s) ** 2)


    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)




import sys
# To select the correct Detect architecture, match hier_arch_version to Detect class
# hier_arch_version could be located in default.yaml, or could be passed through CLI
# CLI should override default.yaml
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

# Get the value of hier_arch_version from default.yaml
hier_arch_version = DEFAULT_CFG.hier_arch_version


# Get the architecture corresponding to hier_arch_version
def get_detect_class(hier_arch_version):
    if hier_arch_version == 1:
        return Detect1
    elif hier_arch_version == 2:
        return Detect2
    elif hier_arch_version == 3:
        return Detect3
    else:
        return Detect_ORI

Detect = get_detect_class(hier_arch_version)

class DetectDeepDBB(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(DeepDiverseBranchBlock(x, c2, 3), DeepDiverseBranchBlock(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        # x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1) for i in range(self.nl)
        ]
        if hasattr(self, 'cv2') and hasattr(self, 'cv3'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

class DetectWDBB(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(WideDiverseBranchBlock(x, c2, 3), WideDiverseBranchBlock(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        # x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1) for i in range(self.nl)
        ]
        if hasattr(self, 'cv2') and hasattr(self, 'cv3'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)




import cv2,os
import numpy as np

from .conv import Focus
from .block import  Proto_v2
class OBB_Hier(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    # def __init__(self, nc=80, ne=1, ch=()):
    def __init__(self, nc=80, ne=1, ch=(),  hier={}): 
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch, hier)
        self.ne = ne  # number of extra parameters

        self.Proto_v2 = Proto_v2(c1= 128, c_=256, c2=128)
        # self.final_conv = nn.Conv2d(256, 1, kernel_size=1)# 
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

        self.Focus = Focus(3, [256], k=1, s=1, num_layers=2) # input feature map dim
        self.proj_sar = nn.Sequential(nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU()) # input feature map dim
        self.proj_shape = nn.Sequential(nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU()) # input feature map dim

        dict_cls = ['Airbus_A220', 'Boeing777', 'Boeing737', 'Airbus_A330', 'Airbus_A320', 'Boeing767', 'Helicopter', 'Airfreig
hter', 'Boeing747', 'Fokker-50', 'Gulfstream', 'Other_Aircraft']   # same index with data.yaml
        img_list = []
        for cls in dict_cls:
            img_path = os.path.join('/data/template', cls + '.png')
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (112, 112))
            img_list.append(img)

        batch_np = np.stack(img_list, axis=0)
        self.batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2) / 255.0

    # def forward(self, x):
    def forward(self, x): 
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        image_features = x[3:-1][0]
        x_input = x[-1]
        x = x [:3]
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x_out = Detect.forward(self, x, x_input)

        if self.training:
            shape_features = self.Focus(self.batch_tensor.to(x[0]))
            mask_pred = shape_features
            # return x_out, angle, [self.Proto_v2(x) for x in image_features], shape_features, mask_pred #
            # return x_out, angle, image_features, shape_features, mask_pred #
            return x_out, angle, [self.proj_sar(image_features[0])], [self.proj_shape(shape_features[0])], mask_pred #
        return torch.cat([x_out, angle], 1) if self.export else (torch.cat([x_out[0], angle], 1), (x_out[1], angle))


    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

class OBB_shape(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    # def __init__(self, nc=80, ne=1, ch=()):
    def __init__(self, nc=80, ne=1, ch=(),  n_features=0): 
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.n_features = n_features  # 额外特征数量
        # self.Proto_v2 = Proto_v2(c1= 128, c_=64, c2=1)
        # self.final_conv = nn.Conv2d(256, 1, kernel_size=1)
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

        self.Focus = Focus(3, [256], k=1, s=1, num_layers=2)

    # def forward(self, x):
    def forward(self, x): 
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        image_features = x[3:][0]
        x_input = x[-1]
        x = x [:3]
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x_out = Detect.forward(self, x)
        if self.training:
            # mask_pred = self.final_conv(shape_features)
            # return x, angle #ori
            dict_cls = ['Airbus_A220', 'Boeing777', 'Boeing737', 'Airbus_A330', 'Airbus_A320', 'Boeing767', 'Helicopter', 'Airfreig
hter', 'Boeing747', 'Fokker-50', 'Gulfstream', 'Other_Aircraft'] # same index with data.yaml
            img_list = []
            for cls in dict_cls:
                img_path = os.path.join('/data/20250115/yolo_obb_format49-17/template', cls + '.png')
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (112, 112))
                img_list.append(img)

            batch_np = np.stack(img_list, axis=0)
            batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2) / 255.0
            shape_features = self.Focus(batch_tensor.to(x[0]))

            return x_out, angle, image_features, shape_features, shape_features 


        return torch.cat([x_out, angle], 1) if self.export else (torch.cat([x_out[0], angle], 1), (x_out[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)



class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)



use_Hier_head = DEFAULT_CFG.use_Hier_head
def get_Hierhead(use_Hier_loss):
    if use_Hier_head:
        return OBB_Hier
    else:
        return  OBB_shape
OBB = get_Hierhead(use_Hier_head)