"""
model_optimization.py - بهینه‌سازی مدل‌ها (کوانتیزاسیون، Pruning، تبدیل به ONNX/TensorRT)
"""

import torch
import torch.quantization as quant
import onnx
import onnxruntime
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelOptimizer:
    @staticmethod
    def quantize_dynamic(model: torch.nn.Module, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
        model.eval()
        model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.LSTM}, dtype=dtype)
        logger.info("Dynamic quantization completed")
        return model_quantized

    @staticmethod
    def quantize_static(model: torch.nn.Module, calibration_loader, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        for images, _ in calibration_loader:
            model(images)
        torch.quantization.convert(model, inplace=True)
        logger.info("Static quantization completed")
        return model

    @staticmethod
    def prune_model(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        logger.info(f"Pruned {amount*100}% of weights")
        return model

    @staticmethod
    def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: str, opset: int = 12) -> str:
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=opset,
                          input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}})
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"Model exported to ONNX: {onnx_path}")
        return onnx_path

    @staticmethod
    def optimize_onnx(onnx_path: str, output_path: str) -> str:
        from onnxruntime.transformers import optimizer
        opt_model = optimizer.optimize_model(onnx_path, model_type='bert', num_heads=12, hidden_size=768)
        opt_model.save_model_to_file(output_path)
        logger.info(f"Optimized ONNX saved: {output_path}")
        return output_path

    @staticmethod
    def convert_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True) -> str:
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
        config.max_workspace_size = 1 << 30
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        logger.info(f"TensorRT engine saved: {engine_path}")
        return engine_path

    @staticmethod
    def get_model_size(model_path: str) -> float:
        return Path(model_path).stat().st_size / (1024 * 1024)