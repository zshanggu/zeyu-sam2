# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from decord import VideoReader
from decord import cpu

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

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
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "../notebooks/videos/failed.mp4"

# Load the video using decord
vr = VideoReader(video_dir, ctx=cpu(0))

# Get the first frame
frame_rgb = vr[0].asnumpy()

# Convert the frame from HWC (height, width, channels) to RGB
# frame_rgb = first_frame[..., ::-1]  # Decord loads as BGR by default

# Define the callback for mouse click events
clicked_points = []
labels = []

def on_click(event):
    # Ignore clicks outside the axes (e.g., on the button)
    if event.inaxes is not None and event.inaxes != finalize_button_ax:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append([x, y])
        labels.append(1)  # Assume positive click for simplicity
        print(f"Clicked point: ({x}, {y})")
        # Show the clicked point on the plot
        plt.scatter(x, y, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.draw()

# take a look the first video frame
frame_idx = 0

# image_0 = np.array(image_0.convert("RGB"))
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title(f"frame {frame_idx}")
ax.imshow(frame_rgb)
ax.axis('on')

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Add a button to finalize points
def finalize_points(event):
    plt.close(fig)
    print(f"Finalized points: {clicked_points}")

finalize_button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # Button position in the figure
finalize_button = Button(finalize_button_ax, "Finalize")
finalize_button.on_clicked(finalize_points)

plt.show()

# sys.exit()

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[210, 350]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.axis('on')
# plt.show()


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
# points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 1], np.int32)
# After the interaction, use the points and labels
points = np.array(clicked_points, dtype=np.float32)
labels = np.array(labels, np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(vr[ann_frame_idx])
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.axis('on')
plt.show()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(vr), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(vr[out_frame_idx])

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.axis('on')
    plt.show()

# ann_frame_idx = 150  # further refine some details on this frame
# ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)

# # show the segment before further refinement
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx} -- before refinement")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)
# plt.axis('on')
# plt.show()

# # Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
# points = np.array([[82, 410]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([0], np.int32)
# _, _, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the segment after the further refinement
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx} -- after refinement")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
# plt.axis('on')
# plt.show()

# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# # render the segmentation results every few frames
# vis_frame_stride = 30
# plt.close("all")
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()

# predictor.reset_state(inference_state)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
# box = np.array([300, 0, 500, 400], dtype=np.float32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     box=box,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_box(box, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.axis('on')
# plt.show()

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (460, 60) to refine the mask
# points = np.array([[460, 60]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# # note that we also need to send the original box input along with
# # the new refinement click together into `add_new_points_or_box`
# box = np.array([300, 0, 500, 400], dtype=np.float32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
#     box=box,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_box(box, plt.gca())
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.axis('on')
# plt.show()

# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# # render the segmentation results every few frames
# vis_frame_stride = 30
# plt.close("all")
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()

# predictor.reset_state(inference_state)

# prompts = {}  # hold all the clicks we add for visualization

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (200, 300) to get started on the first object
# points = np.array([[200, 300]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# for i, out_obj_id in enumerate(out_obj_ids):
#     show_points(*prompts[out_obj_id], plt.gca())
#     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()

# # add the first object
# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# # sending all clicks (and their labels) to `add_new_points_or_box`
# points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# for i, out_obj_id in enumerate(out_obj_ids):
#     show_points(*prompts[out_obj_id], plt.gca())
#     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

# # Let's now move on to the second object we want to track (giving it object id `3`)
# # with a positive click at (x, y) = (400, 150)
# points = np.array([[400, 150]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels

# # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame on all objects
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# for i, out_obj_id in enumerate(out_obj_ids):
#     show_points(*prompts[out_obj_id], plt.gca())
#     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()

# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# # render the segmentation results every few frames
# vis_frame_stride = 30
# plt.close("all")
# for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
#     plt.axis('on')
#     plt.show()
