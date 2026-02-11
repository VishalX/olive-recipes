import os
import sys
import torch

from transformers import Qwen3VLConfig

# Add parent directory to sys.path to import codes module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from codes.modeling_qwen3_vl import Qwen3VLModel

model_name = "Qwen/Qwen3-VL-4B-Instruct"
config = Qwen3VLConfig.from_pretrained(model_name)


### Embedding
# Dynamo export

def get_embedding_model(model_path=None):
    model = Qwen3VLModel.from_pretrained(
        model_path or model_name,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model


def get_embedding_io_config(model_path=None):
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
    }
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_shapes": dynamic_shapes,
    }


def get_embedding_dummy_inputs(model=None):
    # 4B: out_hidden_size 2560; assume 2 batches, each with 1 image (3577 logical patches)
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        1,
        3606,
        3577,
        2560,  # 4B model hidden_size
    )
    num_logical_patches = batch_size * patches_per_image

    vision_start_token_id = config.vision_start_token_id  # 151652
    vision_end_token_id = config.vision_end_token_id  # 151653
    image_token_id = config.image_token_id  # 151655

    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=image_token_id,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_logical_patches,
            out_hidden_size,
            dtype=torch.float32,
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    inputs["input_ids"][0][2] = vision_start_token_id
    inputs["input_ids"][0][img_start_index:img_end_index] = image_token_id
    inputs["input_ids"][0][img_end_index] = vision_end_token_id

    return {
        "input_ids": inputs["input_ids"],
        "image_features": inputs["image_features"],
    }


### Vision
# Wrapper returns single image-feature tensor for ONNX (same pattern as builder_simple).
class VisionWrapper(torch.nn.Module):
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model

    def forward(self, pixel_values, image_grid_thw):
        outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            if isinstance(outputs[1], list):
                return outputs[0]
            return outputs[1]
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def get_vision_model(model_path=None):
    model = Qwen3VLModel.from_pretrained(
        model_path or model_name,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    visual = model.visual
    visual._skip_deepstack_export = True
    if hasattr(visual.config, "_attn_implementation"):
        visual.config._attn_implementation = "eager"
    wrapped = VisionWrapper(visual)
    wrapped.eval()
    return wrapped


# Vision: pixel_values shape [num_patches, 1536] with num_patches dynamic.
VISION_PATCH_FEATURES = 3 * 2 * 16 * 16  # 1536


def get_vision_io_config(model_path=None):
    # First dim of pixel_values is dynamic ("num_patches"); second dim fixed 1536.
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
            "image_features": {0: "num_logical_patches"},
        },
        "dynamic_axes": {
            "pixel_values": {"0": "num_patches"},
            "image_grid_thw": {"0": "num_images"},
            "image_features": {"0": "num_logical_patches"},
        },
    }


def get_vision_dummy_inputs(model=None):
    # Example shapes for tracing only; exported ONNX has dynamic num_patches.
    # pixel_values: [num_patches, 1536], image_grid_thw: [num_images, 3].
    num_patches = 576  # one example (e.g. 24×24 for 384×384)
    pixel_values = torch.randn(num_patches, VISION_PATCH_FEATURES, dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 24, 24]], dtype=torch.int64)
    return (pixel_values, image_grid_thw)
