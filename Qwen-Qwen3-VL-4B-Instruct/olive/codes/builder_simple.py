"""
Simplified Qwen3-VL ONNX Model Builder

Exports three separate ONNX models:
1. Vision encoder (FP32) - fixed 384×384 input
2. Embeddings (FP32) - simple text embedding layer
3. Text decoder (INT4-RTN) - using onnxruntime_genai builder
"""

import argparse
import importlib.util
import os
import shutil
import sys
import torch
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from onnxruntime_genai.models.builder import create_model


def _use_pytorch_modeling_file(args):
    """
    Copy pytorch_backup/modeling_qwen3_vl.py (repo source of truth) into the
    transformers package (site-packages) before loading/exporting. The model
    is then loaded from args.input (e.g. pytorch/ = user's downloaded weights/config);
    we do not use the installed transformers modeling file.
    Call this before importing Qwen3VLForConditionalGeneration.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(script_dir, "pytorch_backup", "modeling_qwen3_vl.py")
    if not os.path.isfile(src):
        print(f"  [WARNING] Local modeling file not found: {src}")
        return
    spec = importlib.util.find_spec("transformers.models.qwen3_vl")
    if spec is None or not spec.origin:
        print("  [WARNING] Could not find transformers.models.qwen3_vl package")
        return
    pkg_dir = os.path.dirname(spec.origin)
    dst = os.path.join(pkg_dir, "modeling_qwen3_vl.py")
    shutil.copy2(src, dst)
    print(f"  - Copied: {src}")
    print(f"  - To:     {dst}")
    # Force reload on next import so the copied file is used
    for mod in list(sys.modules):
        if mod == "transformers.models.qwen3_vl.modeling_qwen3_vl" or mod == "transformers.models.qwen3_vl":
            del sys.modules[mod]


def prepare_model(args, load_processor=False):
    """Load model from args.input (e.g. pytorch/ = user's downloaded Qwen3 VL).
    
    Modeling code is injected by _use_pytorch_modeling_file() in main() before this runs:
    pytorch_backup/modeling_qwen3_vl.py is copied to site-packages so the loaded model uses it.
    """
    print("=" * 80)
    print("Preparing Qwen3-VL Model for ONNX Export")
    print("=" * 80)
    
    # Load config
    print("\n[1/2] Loading configuration...")
    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    print(f"  - Model type: {config.model_type}")
    print(f"  - Vision hidden size: {config.vision_config.hidden_size}")
    print(f"  - Text hidden size: {config.text_config.hidden_size}")
    
    # Load processor only if needed (for text model export)
    processor = None
    if load_processor:
        print("\n[2/2] Loading processor and model...")
        processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True)
    else:
        print("\n[2/2] Loading model (skipping processor)...")
    
    # Import (uses repo's modeling file after _use_pytorch_modeling_file copies it to site-packages)
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.input,
        trust_remote_code=True,
        torch_dtype=args.precision,
    )
    model.to(args.device)
    model.eval()
    
    print(f"  [OK] Model loaded")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return config, processor, model


def build_vision(args, config, processor, model, output_dir):
    """
    Export vision encoder to ONNX - single output pooled_embeds (merged patches).
    
    Target: 384×384 input (processor reshapes to 384×384).
    Patch size 16 → grid 24×24 = 576 patches; patch features 3×2×16×16 = 1536.
    Some processor configs output 20×20 = 400 patches; use --vision_grid 20 to match.
    """
    print("\n" + "=" * 80)
    print("Building Vision Model (Simplified)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print("\n[1/3] Preparing dummy inputs...")
    
    # Grid for 384×384: 384/16 = 24 → 24×24 = 576 patches (or --vision_grid 20 for 400 patches)
    grid_size = args.vision_grid
    grid_t, grid_h, grid_w = 1, grid_size, grid_size
    num_patches = grid_t * grid_h * grid_w
    patch_features = 3 * 2 * 16 * 16  # RGB × temporal_patch × spatial_patch²
    
    dummy_pixel_values = torch.randn(num_patches, patch_features)
    dummy_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    
    print(f"  - Image size: 384×384 (processor standard)")
    print(f"  - Grid: {grid_t}×{grid_h}×{grid_w} = {num_patches} patches")
    print(f"  - Patch features: {patch_features}")
    print(f"  - pixel_values: {dummy_pixel_values.shape}")
    print(f"  - grid_thw: {dummy_grid_thw.shape}")
    
    print("\n[2/3] Creating vision wrapper (pooler_output only, no deepstack)...")
    
    # Wrapper returns only pooler_output (merged hidden states) to match working ONNX.
    # ModelOutput order: last_hidden_state, pooler_output, deepstack_features.
    class VisionWrapper(torch.nn.Module):
        def __init__(self, visual_model):
            super().__init__()
            self.visual = visual_model
        
        def forward(self, pixel_values, image_grid_thw):
            outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Return single tensor (pooled embeddings) to match working ONNX (one output).
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                # Vision returns (merged_tensor, deepstack_list); return the tensor only.
                if isinstance(outputs[1], list):
                    return outputs[0]
                return outputs[1]  # BaseModelOutputWithPooling style: pooler is second
            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    
    # Skip deepstack during export so graph has single merger path (matches working ONNX).
    visual = model.model.visual
    old_skip = getattr(visual, '_skip_deepstack_export', None)
    visual._skip_deepstack_export = True
    
    wrapped_model = VisionWrapper(visual)
    wrapped_model.eval()
    
    # Force eager attention (workaround for GQA/SDPA export issues)
    original_attn = getattr(model.config, '_attn_implementation', None)
    model.config._attn_implementation = 'eager'
    
    if hasattr(model.model.visual.config, '_attn_implementation'):
        original_vision_attn = model.model.visual.config._attn_implementation
        model.model.visual.config._attn_implementation = 'eager'
    
    print(f"  - Forced attention: eager (for ONNX compatibility)")
    
    print("\n[3/3] Exporting to ONNX...")
    
    output_path = os.path.join(output_dir, "qwen3vl-vision.onnx")
    
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_pixel_values, dummy_grid_thw),
            output_path,
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["pooled_embeds"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "image_grid_thw": {0: "num_images"},
                "pooled_embeds": {0: "num_merged_patches"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [OK] Vision model exported: {output_path}")
        
    except Exception as e:
        print(f"  [FAIL] Vision export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Restore original attention and deepstack flag
        if original_attn:
            model.config._attn_implementation = original_attn
        if 'original_vision_attn' in locals():
            model.model.visual.config._attn_implementation = original_vision_attn
        if old_skip is not None:
            visual._skip_deepstack_export = old_skip
        else:
            try:
                del visual._skip_deepstack_export
            except AttributeError:
                pass


def build_embedding(args, config, processor, model, output_dir):
    """
    Export embedding layer to ONNX - simplified direct export.
    
    Just the text embedding layer, no vision merging.
    """
    print("\n" + "=" * 80)
    print("Building Embedding Model (Simplified)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print("\n[1/2] Preparing embedding layer...")
    
    # Get the text embedding layer directly
    embeddings = model.model.language_model.embed_tokens
    embeddings.eval()
    
    print(f"  - Embedding layer: {type(embeddings)}")
    print(f"  - Vocab size: {embeddings.num_embeddings}")
    print(f"  - Embedding dim: {embeddings.embedding_dim}")
    
    # Create dummy input
    batch_size, seq_len = 1, 10
    vocab_size = config.text_config.vocab_size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)
    
    print(f"  - Input IDs shape: {dummy_input_ids.shape}")
    
    print("\n[2/2] Exporting to ONNX...")
    
    output_path = os.path.join(output_dir, "qwen3vl-embedding.onnx")
    
    try:
        torch.onnx.export(
            embeddings,
            dummy_input_ids,
            output_path,
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [OK] Embedding model exported: {output_path}")
        
    except Exception as e:
        print(f"  [FAIL] Embedding export failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_text(args, config, processor, model, output_dir, text_precision="int4"):
    """Export text model using onnxruntime_genai builder."""
    print("\n" + "=" * 80)
    print(f"Building Text Model ({text_precision.upper()})")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    print(f"\nUsing onnxruntime_genai builder for text model (precision={text_precision})...")
    
    model_name = None
    extra_options = {
        "exclude_embeds": "true",
        "filename": "model.onnx",  # Standard GenAI filename
    }
    
    # Only add INT4 options if using INT4
    if text_precision == "int4":
        extra_options["int4_algo_config"] = "rtn"
        if args.precision == torch.float32:
            extra_options["int4_accuracy_level"] = 4
    
    try:
        create_model(
            model_name, 
            args.input, 
            output_dir, 
            text_precision,  # Use the parameter instead of hardcoded "int4"
            args.execution_provider, 
            args.cache_dir, 
            **extra_options
        )
        print(f"\n[OK] Text model exported ({text_precision})")
    except Exception as e:
        print(f"\n[FAIL] Text model export failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_args():
    parser = argparse.ArgumentParser(description="Build Qwen3-VL ONNX models (simplified)")
    
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to folder containing the Hugging Face model files",
    )
    
    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Precision for vision/embedding models",
    )
    
    parser.add_argument(
        "-t",
        "--text_precision",
        required=False,
        default="int4",
        choices=["fp32", "int4"],
        help="Precision for text model (default: int4)",
    )
    
    parser.add_argument(
        "-e",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "dml"],
        help="Execution provider for ONNX models",
    )
    
    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        default=os.path.join('.', 'cache_dir'),
        help="Cache directory for temporary files",
    )
    
    # Export control arguments
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Export vision model only",
    )
    
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Export embedding model only",
    )
    
    parser.add_argument(
        "--text",
        action="store_true",
        help="Export text model only",
    )
    
    parser.add_argument(
        "--vision_grid",
        type=int,
        default=24,
        help="Spatial grid size for vision export (24 → 24×24=576 patches for 384×384; use 20 for 400 patches if processor outputs 20×20)",
    )
    
    args = parser.parse_args()
    
    # If no specific model is selected, export all
    if not args.vision and not args.embedding and not args.text:
        args.export_vision = True
        args.export_embedding = True
        args.export_text = True
    else:
        args.export_vision = args.vision
        args.export_embedding = args.embedding
        args.export_text = args.text
    
    # Convert precision string to torch dtype
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Set device based on execution provider
    if args.execution_provider == "cuda":
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    # Set output directories based on model type
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precision_str = "fp32" if args.precision == torch.float32 else "fp16"
    
    args.vision_output = os.path.join(script_dir, f"cpu-{precision_str}")
    args.embedding_output = args.vision_output  # Same directory
    args.text_output = os.path.join(script_dir, f"cpu-{args.text_precision}")
    
    # Create output directories
    os.makedirs(args.vision_output, exist_ok=True)
    os.makedirs(args.text_output, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    return args


def main():
    args = get_args()
    
    print("\n" + "=" * 80)
    print("Qwen3-VL ONNX Model Builder (Simplified)")
    print("=" * 80)
    print(f"\nInput directory: {args.input}")
    print(f"Vision/Embedding output: {args.vision_output}")
    print(f"Text output: {args.text_output}")
    print(f"Precision: {args.precision}")
    print(f"Text precision: {args.text_precision}")
    print(f"Execution provider: {args.execution_provider}")
    print(f"Device: {args.device}")
    
    # Show what will be exported
    export_list = []
    if args.export_vision:
        export_list.append("vision")
    if args.export_embedding:
        export_list.append("embedding")
    if args.export_text:
        export_list.append("text")
    print(f"\nExporting: {', '.join(export_list)}")
    
    print("\nSimplifications:")
    print("  - Fixed image size: 384×384 pixels")
    print("  - No post-processing (optimization/quantization for vision)")
    print("  - Direct embedding export (no vision merging)")
    print(f"  - Text model uses {args.text_precision.upper()} precision")
    
    # Copy pytorch_backup/modeling_qwen3_vl.py to site-packages before loading/exporting
    print("\nUsing repo source pytorch_backup/modeling_qwen3_vl.py (copying to site-packages before load)...")
    _use_pytorch_modeling_file(args)
    
    # Prepare model (only load processor if building text model)
    config, processor, model = prepare_model(args, load_processor=args.export_text)
    
    # Build selected components
    exported = []
    
    if args.export_vision:
        build_vision(args, config, processor, model, args.vision_output)
        exported.append(("Vision", args.vision_output, "qwen3vl-vision.onnx"))
    
    if args.export_embedding:
        build_embedding(args, config, processor, model, args.embedding_output)
        exported.append(("Embedding", args.embedding_output, "qwen3vl-embedding.onnx"))
    
    if args.export_text:
        build_text(args, config, processor, model, args.text_output, args.text_precision)
        exported.append(("Text", args.text_output, "model.onnx"))
    
    # Summary
    print("\n" + "=" * 80)
    print("[OK] ONNX model(s) exported successfully!")
    print("=" * 80)
    precision_str = "FP32" if args.precision == torch.float32 else "FP16"
    
    for name, output_dir, filename in exported:
        print(f"\n{name} model: {output_dir}")
        print(f"  - {filename}")
    
    print("\nNext steps:")
    print("  1. Test text+image: python qwen3-vl.py --image images/test_checkerboard.jpg --text 'Describe this.' --max_new_tokens 50")
    print("  2. If vision fails with Reshape (e.g. 576 vs 400), re-export with --vision_grid 20 to match 400 patches.")
    print("  3. Processor should output 384×384; grid (e.g. 24×24 or 20×20) must match --vision_grid.")
    print()


if __name__ == "__main__":
    main()
