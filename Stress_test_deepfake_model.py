import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import psutil
import os
import json



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class ImprovedCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(SEBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2



def quantize_model_dynamic(model):
    """Dynamic quantization (CPU only)."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

def load_models(device):
    print("\n" + "="*70)
    print(" LOADING MODELS & WEIGHTS ")
    print("="*70)

    models = {
        "ConvNeXt-Tiny": timm.create_model("convnext_tiny", pretrained=False, num_classes=2),
        "EfficientNet-B0": timm.create_model("efficientnet_b0", pretrained=False, num_classes=2),
        "Custom-ResNet-SE": ImprovedCustomCNN()
    }

    weight_paths = {
        "ConvNeXt-Tiny": Path("convnext_final.pth"),
        "EfficientNet-B0": Path("efficientnet_final.pth"),
        "Custom-ResNet-SE": Path("custom_cnn_final.pth")
    }

    for name, model in list(models.items()):
        path = weight_paths[name]
        if path.exists():
            print(f" Loaded weights for {name} from {path.name}")
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            print(f" Weights not found for {name} ({path.name}), skipping this model.")
            models.pop(name)
            continue
        model.to(device).eval()

    return models


def warmup_model(model, input_shape, device, num_iterations=10):
    dummy_input = torch.randn(input_shape).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

def benchmark_inference(model, batch_size, num_iterations, device, model_name):
    print(f"\n{'='*70}\nBenchmarking: {model_name}\n{'='*70}")
    input_shape = (batch_size, 3, 224, 224)  # standard image size
    warmup_model(model, input_shape, device)
    dummy_input = torch.randn(input_shape).to(device)

    times = []
    mem_before_cpu = get_cpu_memory()
    mem_before_gpu = get_gpu_memory() if device.type == "cuda" else 0

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc=model_name):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    mem_after_cpu = get_cpu_memory()
    mem_after_gpu = get_gpu_memory() if device.type == "cuda" else 0

    times = np.array(times)
    stats = {
        "model_name": model_name,
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "throughput_imgs_sec": (batch_size * 1000) / np.mean(times),
        "model_size_mb": get_model_size(model),
        "parameters": count_parameters(model),
        "cpu_memory_mb": mem_after_cpu - mem_before_cpu,
        "gpu_memory_mb": mem_after_gpu - mem_before_gpu,
        "times": times.tolist()
    }

    print(f"\nMean Time: {stats['mean_time_ms']:.2f} ms | "
          f"Throughput: {stats['throughput_imgs_sec']:.2f} img/s | "
          f"Size: {stats['model_size_mb']:.2f} MB")

    return stats


def print_summary_table(results_fp32, results_quantized):
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*100)
    print(f"\n{'Model':<20} | {'Precision':<10} | {'Time (ms)':<12} | {'Throughput':<15} | {'Size (MB)':<12}")
    print("-" * 100)

    models = list(set([r["model_name"].split(" (")[0] for r in results_fp32]))
    for model in models:
        fp32 = next(r for r in results_fp32 if r["model_name"].split(" (")[0] == model)
        quant = next((r for r in results_quantized if r["model_name"].split(" (")[0] == model), None)
        if not quant:
            continue

        speedup = fp32["mean_time_ms"] / quant["mean_time_ms"]
        reduction = (1 - quant["model_size_mb"] / fp32["model_size_mb"]) * 100

        print(f"{model:<20} | FP32       | {fp32['mean_time_ms']:>10.2f} | {fp32['throughput_imgs_sec']:>13.2f} | {fp32['model_size_mb']:>10.2f}")
        print(f"{'':<20} | Quantized  | {quant['mean_time_ms']:>10.2f} | {quant['throughput_imgs_sec']:>13.2f} | {quant['model_size_mb']:>10.2f}")
        print(f"{'':<20} | Improvement| {speedup:>9.2f}x | {((quant['throughput_imgs_sec']/fp32['throughput_imgs_sec'] - 1)*100):>11.1f}% | {reduction:>9.1f}%↓")
        print("-" * 100)



def main():
    print("\n" + "="*70)
    print(" DEEPFAKE DETECTION MODEL STRESS TEST ")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size, num_iterations = 64, 100
    models = load_models(device)

    # FP32 benchmark
    results_fp32 = [benchmark_inference(m, batch_size, num_iterations, device, n)
                    for n, m in models.items()]

    # Quantization benchmark (CPU only)
    print("\n" + "="*70)
    print(" QUANTIZATION PHASE ")
    print("="*70)

    quantized_models = {n: quantize_model_dynamic(m.cpu()) for n, m in models.items()}
    results_quantized = [benchmark_inference(m, batch_size, num_iterations, torch.device("cpu"), f"{n} (Quantized)")
                         for n, m in quantized_models.items()]

    print_summary_table(results_fp32, results_quantized)

    results = {
        "fp32": results_fp32,
        "quantized": results_quantized,
        "config": {
            "batch_size": batch_size,
            "iterations": num_iterations,
            "device": str(device)
        }
    }

    with open("stress_test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to stress_test_results.json\n")



if __name__ == "__main__":
    main()

