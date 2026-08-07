"""Batched YOLOX confidence filtering, class-aware NMS, and scale recovery."""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


def _nms(boxes, scores, class_ids, threshold):
    selected = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = order[0]
            selected.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            current_area = np.maximum(0.0, boxes[current, 2] - boxes[current, 0]) * np.maximum(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            remaining_area = np.maximum(0.0, boxes[remaining, 2] - boxes[remaining, 0]) * np.maximum(
                0.0, boxes[remaining, 3] - boxes[remaining, 1]
            )
            union = current_area + remaining_area - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= threshold]

    return np.array(selected, dtype=np.int64)


def postprocess_batch(detections, metadata, confidence_threshold, nms_threshold, max_detections):
    if detections.ndim != 3 or detections.shape[-1] != 7:
        raise ValueError(f"DETECTIONS must have shape [batch, candidates, 7], got {detections.shape}")
    if metadata.shape != (detections.shape[0], 5):
        raise ValueError(f"IMAGE_METADATA must have shape [batch, 5], got {metadata.shape}")

    batch_size = detections.shape[0]
    output_boxes = np.zeros((batch_size, max_detections, 4), dtype=np.float32)
    output_scores = np.zeros((batch_size, max_detections), dtype=np.float32)
    output_classes = np.full((batch_size, max_detections), -1, dtype=np.int32)
    output_counts = np.zeros((batch_size, 1), dtype=np.int32)

    for batch_index in range(batch_size):
        rows = detections[batch_index]
        scores = rows[:, 4] * rows[:, 5]
        valid = np.isfinite(rows).all(axis=1) & (scores >= confidence_threshold)
        boxes = rows[valid, :4].astype(np.float32, copy=True)
        filtered_scores = scores[valid].astype(np.float32)
        class_ids = rows[valid, 6].astype(np.int32)
        if boxes.size == 0:
            continue

        selected = _nms(boxes, filtered_scores, class_ids, nms_threshold)
        selected = selected[np.argsort(filtered_scores[selected])[::-1]][:max_detections]
        boxes = boxes[selected]
        filtered_scores = filtered_scores[selected]
        class_ids = class_ids[selected]

        scale, pad_x, pad_y, input_width, input_height = metadata[batch_index]
        if scale <= 0 or input_width <= 0 or input_height <= 0:
            raise ValueError("IMAGE_METADATA contains invalid scale or image dimensions")
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_height)

        count = len(selected)
        output_boxes[batch_index, :count] = boxes
        output_scores[batch_index, :count] = filtered_scores
        output_classes[batch_index, :count] = class_ids
        output_counts[batch_index, 0] = count

    return output_boxes, output_scores, output_classes, output_counts


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})
        self.confidence_threshold = float(parameters["confidence_threshold"]["string_value"])
        self.nms_threshold = float(parameters["nms_threshold"]["string_value"])
        self.max_detections = int(parameters["max_detections"]["string_value"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                detections = pb_utils.get_input_tensor_by_name(request, "DETECTIONS")
                metadata = pb_utils.get_input_tensor_by_name(request, "IMAGE_METADATA")
                if detections is None or metadata is None:
                    raise ValueError("missing required input DETECTIONS or IMAGE_METADATA")
                outputs = postprocess_batch(
                    detections.as_numpy(),
                    metadata.as_numpy(),
                    self.confidence_threshold,
                    self.nms_threshold,
                    self.max_detections,
                )
                names = ("BBOXES", "SCORES", "CLASS_IDS", "NUM_DETECTIONS")
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor(name, value) for name, value in zip(names, outputs)]
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
