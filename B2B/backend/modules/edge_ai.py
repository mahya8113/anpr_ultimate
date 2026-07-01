"""
edge_ai.py - استقرار مدل روی دستگاه‌های لبه (ONNX، TensorRT، Raspberry Pi)
"""

import onnx
import onnxruntime
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class EdgeExporter:
    @staticmethod
    def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: str, opset: int = 12):
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True,
                          opset_version=opset, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"Model exported to ONNX: {onnx_path}")

    @staticmethod
    def export_to_tensorrt(onnx_path: str, trt_path: str, fp16: bool = False):
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.log(trt.Logger.ERROR, parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX")
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            engine = builder.build_engine(network, config)
            with open(trt_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info(f"TensorRT engine saved to {trt_path}")
        except ImportError:
            logger.error("TensorRT not installed")

    @staticmethod
    def run_onnx(onnx_path: str, input_np: np.ndarray) -> np.ndarray:
        sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: input_np})[0]
        return output

    @staticmethod
    def optimize_for_edge(onnx_path: str, output_path: str, quantization: bool = True):
        """ساده‌سازی مدل ONNX برای اجرا روی لبه"""
        from onnxsim import simplify
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, output_path)
            logger.info(f"Optimized model saved to {output_path}")
        else:
            logger.warning("Simplification failed")