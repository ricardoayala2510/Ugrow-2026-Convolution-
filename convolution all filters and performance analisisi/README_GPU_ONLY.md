# GPU-only CUDA Convolution Benchmark

This version measures only the CUDA/GPU runtime. There is no CPU sequential code.

The program uses the same input image repeatedly and tests these batch sizes:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

Each image runs 4 CUDA convolutions:

```text
Blur + Sharpen + Edge Detection + Emboss
```

So for 1024 images:

```text
1024 images x 4 filters = 4096 CUDA convolutions
```

The benchmark does not create 1024 output files. It saves only four sample outputs:

```text
data/output_blur.jpg
data/output_sharpen.jpg
data/output_edge_detection.jpg
data/output_emboss.jpg
```

The timing table is printed in the Slurm output and saved to:

```text
gpu_benchmark_results.csv
```

For your chart, use:

```text
images vs gpu_wall_ms
```

## Run on TACC

```bash
sbatch run.slurm
```

Then check:

```bash
cat gpu_convolution.<jobid>.out
```
