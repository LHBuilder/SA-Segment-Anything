from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from super_gradients.training import yolon
from typing import Any, Dict, List, Optional, Tuple
import cv2


class UniMultiModalFramework:

    def __init__(
            self, 
            sam_checkpoint, # "sam_vit_h_4b8939.pth"
            sam_model_type, # "vit_h"
            yolo_model, # "yolo_nas_l"
            yolo_pretrained_weights, # "coco"
            yolo_conf_threshold, # 0.25
            device, # "cuda"
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
        ) -> Any:
        model = yolon.get(self._yolo_model, pretrained_wrights=self._yolo_pretrained_weights) 
        detection = model.predict(image_path, conf=self._conf_threshold)
        return detection

