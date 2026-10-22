import intent
import intent.language as I


BATCH = 32
BOXES = 1024


@intent.kernel
def greedy_nms(
    boxes: I.In[I.f32, (BATCH, BOXES, 4)],
    keep: I.Out[I.bool, (BATCH, BOXES)],
    threshold: I.f32,
):
    for batch in I.parallel(I.domain(0, BATCH)):
        suppressed = I.buffer((BOXES,), I.bool, init=False)
        for candidate in range(BOXES):
            if I.mutable_load(suppressed, candidate):
                keep[batch, candidate] = False
                continue
            keep[batch, candidate] = True
            x1 = boxes[batch, candidate, 0]
            y1 = boxes[batch, candidate, 1]
            x2 = boxes[batch, candidate, 2]
            y2 = boxes[batch, candidate, 3]
            candidate_area = I.maximum(x2 - x1, 0.0) * I.maximum(
                y2 - y1,
                0.0,
            )
            for other in range(candidate + 1, BOXES):
                if I.mutable_load(suppressed, other):
                    continue
                other_x1 = boxes[batch, other, 0]
                other_y1 = boxes[batch, other, 1]
                other_x2 = boxes[batch, other, 2]
                other_y2 = boxes[batch, other, 3]
                intersection_width = I.maximum(
                    I.minimum(x2, other_x2) - I.maximum(x1, other_x1),
                    0.0,
                )
                intersection_height = I.maximum(
                    I.minimum(y2, other_y2) - I.maximum(y1, other_y1),
                    0.0,
                )
                intersection = intersection_width * intersection_height
                other_area = I.maximum(other_x2 - other_x1, 0.0) * I.maximum(
                    other_y2 - other_y1,
                    0.0,
                )
                union = candidate_area + other_area - intersection
                if intersection / union > threshold:
                    I.store(suppressed, other, True)
