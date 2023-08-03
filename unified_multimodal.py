# Copyright (c) 2023 Seeking AI Authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from super_gradients.training import models as yolon
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2


class UniMultiModalFramework:

    def __init__(
            self, 
            sam_checkpoint: str, # "sam_vit_h_4b8939.pth"
            sam_model_type: str, # "vit_h"
            yolo_model: str, # "yolo_nas_l"
            yolo_pretrained_weights: str, # "coco"
            yolo_conf_threshold: int, # 0.25
            device: str = "cuda",
            name: str = 'ummf',
        ) -> None:
        self._name = name
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type
        self._device = device
        self._register_sam()
        self._yolo_model = yolo_model
        self._yolo_pretrained_weights = yolo_pretrained_weights
        self._yolo_conf_threshold = yolo_conf_threshold

    def _register_sam(self):
        self._sam = sam_model_registry[self._model_type](checkpoint=self._sam_checkpoint)
        self._sam.to(device=self._device)

    def sa_to_masks(
            self, 
            image_path: str,
        ) -> List[Dict[str, Any]]:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(self._sam)
        masks = mask_generator.generate(image)
        return masks
    
    def sa_yolo_detect(
            self,
            image_path: str,
            conf_threshold: int = None,
        ) -> Any:
        model = yolon.get(self._yolo_model, pretrained_wrights=self._yolo_pretrained_weights)
        if conf_threshold is None:
            conf_threshold = self._conf_threshold
        detection = model.predict(image_path, conf=conf_threshold)
        return detection
    
    def sa_target_single_mask(
            self, 
            image_path: str,
            objIndex: int = 0, # default the first object, it can be selected by using sa_yolo_detect together
        ) -> List[Dict[str, Any]]:

        detection_pred = self.sa_yolo_detect(image_path)._images_prediction_lst

        bboxes_xyxy = detection_pred[objIndex].prediction.bboxes_xyxy.tolist()
        confidence = detection_pred[objIndex].prediction.confidence.tolist()
        labels = detection_pred[objIndex].prediction.labels.tolist()

        image = cv2.imread(image_path)

        predictor = SamPredictor(self._sam)
        predictor.set_image(image)

        image = image.transpose((2, 0, 1)) # Tranpose to match SAM input format
        image = image / 255.0 # Normalize image values to [0, 1]
        image = np.expand_dims(image, axis=0) # Add batch dimension

        input_box = np.array(bboxes_xyxy[objIndex])

        if labels[objIndex] == 0:
            # predict masks using SAM
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
        return masks # may return None

    ## helpers
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], aixs=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=market_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=market_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle(x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones(m.shape[0], m.shape[1], 3)
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack((img, m*0.35)))

