import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from decord import VideoReader, cpu
from sam2.build_sam import build_sam2_video_predictor

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def segment_two_objects(video_path, object1_points, object2_points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    vr = VideoReader(video_path, ctx=cpu(0))
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    prompts = {}
    points_obj1 = np.array(object1_points, dtype=np.float32)
    labels_obj1 = np.array([1] * len(object1_points), np.int32)
    points_obj2 = np.array(object2_points, dtype=np.float32)
    labels_obj2 = np.array([1] * len(object2_points), np.int32)

    prompts[0] = (points_obj1, labels_obj1)
    prompts[1] = (points_obj2, labels_obj2)

    all_points = np.vstack([points_obj1, points_obj2])
    all_labels = np.hstack([labels_obj1, labels_obj2])

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        points=points_obj1,
        labels=labels_obj1,
    )
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points_obj2,
        labels=labels_obj2,
    )

    plt.figure(figsize=(9, 6))
    plt.title("Frame 0 Segmentation")
    plt.imshow(vr[0].numpy())
    show_points(all_points, all_labels, plt.gca())

    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.axis('on')
        plt.show()

    plt.axis('on')
    plt.show()

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    vis_frame_stride = 30
    for out_frame_idx in range(0, len(vr), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx} Segmentation")
        plt.imshow(vr[out_frame_idx].numpy())

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id, random_color=False)

        plt.axis('on')
        plt.show()

# Example usage
video_file = "../notebooks/videos/bedroom.mp4"
object1_coords = [[210, 350], [250, 220]]
object2_coords = [[400, 150], [400, 180]]
segment_two_objects(video_file, object1_coords, object2_coords)
