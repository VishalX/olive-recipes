# DeepStack: pooler_output-only vision export

Qwen3-VL’s vision encoder returns `BaseModelOutputWithDeepstackFeatures`, which includes:

- **pooler_output**: merged visual features (one tensor).
- **deepstack_features**: a list of feature maps from vision layers 5, 11, 17, used for DeepStack injection into early decoder layers.

This Olive recipe exports the **vision** subgraph using **pooler_output only** as `image_features`. DeepStack injection (feeding `deepstack_features` into the decoder) is **not** implemented in the ONNX pipeline.

As a result, this export is “pooler_output-only” and may differ slightly in quality or behavior from the full Qwen3-VL model that uses DeepStack. For many use cases the difference is small. Full DeepStack support would require the vision model to expose multiple outputs and the decoder/embedding side to consume them.
