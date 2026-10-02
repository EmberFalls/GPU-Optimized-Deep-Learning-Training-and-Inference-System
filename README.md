# InferForge

**A GPU-aware deep learning training, optimization, and inference engineering system.**

InferForge is a planned, production-style engineering project for taking a real image-classification model from reproducible training to measured, optimized, concurrent inference. Its central rule is that every performance claim must be supported by a repeatable benchmark while model quality, hardware, runtime configuration, and failure behavior remain visible.

## Development status

**Repository definition only.** This README establishes the intended scope and engineering contract. No package, training pipeline, inference backend, API, benchmark result, or deployment artifact has been implemented yet. Commands and architecture below describe the planned interface and may evolve as implementation evidence accumulates.

## Engineering problem

A model that gives correct predictions in a notebook is not automatically a usable inference system. InferForge is intended to answer practical questions such as:

- How do precision, batch size, compilation, and runtime choice affect latency, throughput, memory, and quality?
- Where do preprocessing, data transfer, framework overhead, and GPU execution spend time?
- How should concurrent requests be queued, batched, timed out, and rejected under overload?
- Can PyTorch, ONNX Runtime, and TensorRT be compared fairly using the same checkpoint and workload?
- Can every reported improvement be reproduced from stored configuration and benchmark artifacts?

## Reference workload

The initial workload is planned as:

- **Task:** image classification
- **Model:** ConvNeXt-Tiny
- **Dataset:** Food-101
- **Primary quality metric:** top-1 validation accuracy
- **Input:** RGB images with model-appropriate preprocessing

ConvNeXt-Tiny is the default starting point because it is substantial enough to expose meaningful accelerator behavior while remaining practical to fine-tune and benchmark on a 24 GB GPU. A ViT-B/16 workload may be considered later if a concrete compatibility or experimental need justifies it.

The model serves as the reference workload for systems engineering; developing a novel neural architecture or achieving state-of-the-art classification accuracy is not a project objective. The initial implementation is expected to fine-tune a pretrained model so that engineering effort remains focused on training reproducibility, profiling, optimization, inference, and serving behavior.

## Initial architecture

```text
Food-101 dataset
       │
       ▼
Preprocessing ──► PyTorch model ──► Reproducible training/evaluation
                                            │
                                            ▼
                                  Trusted FP32 baseline
                                            │
                                            ▼
                                      GPU profiling
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
              Mixed precision        torch.compile          Quantization
                     └──────────────────────┼──────────────────────┘
                                            ▼
                          PyTorch / ONNX Runtime / TensorRT
                                            │
                                            ▼
                         Bounded queue ──► Dynamic batcher
                                            │
                                            ▼
                                      Inference worker
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                         FastAPI service            Redis cache
                              │
                              ▼
                    Load tests and benchmark JSON
                              │
                              ▼
                       Generated plots and reports
```

The first working inference path will be PyTorch eager execution. Optimized backends will be added without removing the reference implementation so correctness and performance remain directly comparable.

## Hardware strategy

### Local CPU and Intel XPU development

Normal development must not require CUDA merely to import or test the package. The local path is intended to support:

- configuration and unit tests;
- deterministic data-pipeline smoke tests;
- reduced training and evaluation runs;
- CPU inference and Intel XPU execution where available;
- FastAPI, queueing, batching, concurrency, and Redis integration;
- benchmark-harness development and ONNX checks where supported.

Optional accelerator dependencies will be loaded lazily. Requesting an unavailable device or backend must produce a clear error rather than silently using a different backend.

### NVIDIA benchmark environment

Final NVIDIA-specific experiments are planned for a rented RTX 4090 with 24 GB VRAM, such as a RunPod instance. That environment will be used for:

- CUDA mixed precision;
- CUDA-backed ONNX Runtime;
- TensorRT FP16 and justified INT8 experiments;
- PyTorch Profiler and question-driven NVIDIA Nsight captures;
- GPU memory, utilization, throughput, and concurrency measurements.

Expensive GPU time should be reserved for prepared, scripted experiments. Analysis and report editing should happen locally after artifacts have been copied to persistent storage and verified.

## Benchmark principles

InferForge will follow a baseline-first measurement loop:

```text
baseline → measure → profile → form a hypothesis → change one variable
         → validate correctness → repeat the same workload → compare
```

Comparisons must hold model weights, input dimensions, sample source, machine, concurrency, batch configuration, and warmup policy constant **except for the variable intentionally under study**. Benchmarks will separate cold start, warmup, and steady-state execution, and repeated measurements will be used instead of one-off timings.

An optimized backend will not be accepted on speed alone. It must first demonstrate compatible output shape, prediction agreement or documented numerical tolerance, and acceptable validation quality.

## Core metrics

Planned benchmark records include:

- end-to-end and model-execution latency: mean, minimum, maximum, P50, P95, and P99;
- requests per second and samples per second;
- accelerator-memory metrics appropriate to each backend, including PyTorch peak allocated/reserved memory and process- or device-level VRAM usage where measurable;
- average and peak GPU utilization when supported;
- top-1 validation accuracy and quality change relative to the reference model;
- queue wait time, batch fill time, and realized batch size;
- failures, timeouts, OOM events, invalid inputs, rejections, and queue saturation;
- cache hits and misses in cache-specific tests;
- git revision, checkpoint identity, hardware, driver/runtime versions, backend, precision, and full workload configuration.

Cached and uncached results will be measured separately. No cache hit will be presented as GPU inference throughput.

## Planned scope

The core project is expected to grow, in dependency order, to include:

- reproducible PyTorch training, validation, checkpointing, and resume;
- explicit CPU, Intel XPU, and CUDA device selection;
- FP32 and supported FP16/BF16 paths;
- PyTorch eager and `torch.compile` benchmarks;
- ONNX export and explicit ONNX Runtime provider selection;
- TensorRT FP16 benchmarking on NVIDIA hardware;
- INT8 only when calibration, correctness, performance, and quality are validated;
- profiling, warmup, static batching, and dynamic batching;
- a bounded request queue, timeouts, backpressure, and controlled GPU execution;
- FastAPI health, readiness, and prediction endpoints;
- Redis result caching where it demonstrates useful behavior;
- load testing, failure testing, Docker execution, and generated benchmark reports.

## Explicit non-goals

The core version will not include:

- Kubernetes, Kafka, Grafana, Prometheus, or a service mesh;
- large microservice architectures or distributed cloud orchestration;
- a frontend dashboard;
- complex authentication, enterprise RBAC, or multi-region deployment;
- a feature store or full model registry;
- speculative plugin systems, premature generic abstractions, or infrastructure without a measured need;
- distributed training before single-device behavior is correct and reproducible.

These exclusions keep the project centered on deep learning and inference engineering. Optional extensions will be considered only after the complete core path works and a measured requirement justifies them.

## Planned command interface

The following commands illustrate the intended workflow; **they do not exist yet**:

```text
uv sync
uv run inferforge train --config configs/train.yaml
uv run inferforge evaluate --checkpoint <checkpoint>
uv run inferforge benchmark --config configs/benchmark.yaml
uv run inferforge serve --config configs/inference.yaml
```

Exact command names and configuration fields will be finalized alongside their implementations. Large datasets, model weights, and runtime engines will be downloaded or built only through explicit commands—never as an import-time side effect.

## Definition of success

InferForge succeeds when it provides a trustworthy model-quality baseline, identifies real bottlenecks, validates optimized outputs, quantifies performance/quality tradeoffs, explains behavior under concurrent load and failure, and regenerates every published number from preserved benchmark artifacts.

No placeholder value will be presented as a result. Terms such as *production-style*, *load-tested*, *containerized*, and *benchmark-reproducible* will be used only after the corresponding evidence exists; the stronger claim *production-ready* is not assumed.
