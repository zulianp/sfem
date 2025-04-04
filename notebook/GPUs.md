# P100

GPU Architecture NVIDIA Pascal
NVIDIA CUDA Cores 3584
FP64 		4.7  TFLOPS
FP32 		9.3  TFLOPS
FP16		18.7 TFLOPS

GPU Memory 16GB 
CoWoS  HBM2 at 732 GB/s or 12GB CoWoS HBM2 at 549 GB/s
System Interface PCIe Gen3

# V100 

Tensor Cores 640
Cuda  Cores 5120

			Tesla V100 PCle Tesla V100 SXM2

FP64 			7 TFLOPS 	7.8 TFLOPS
FP32		   14 TFLOPS   15.7 TFLOPS
TC ? 		  112 TFLOPS  125 	TFLOPS
Mem 		   32 GB       16    GB 
GPU Mem Bandwidth 900GB/sec

# A100

64 warps per SM
						A100 80GB PCIe	A100 80GB SXM
FP64					  9.7 	TFLOPS
FP64 Tensor Core		 19.5 	TFLOPS
FP32					 19.5 	TFLOPS
Tensor Float 32 (TF32)	156 	TFLOPS 		| 312 TFLOPS*
FP16 Tensor Core	    312 	TFLOPS 		| 624 TFLOPS*
INT8 Tensor Core		624 	TOPS   		| 1248 TOPS*

# H100 (Alps)

		H200 SXM1 	H200 NVL 
FP64 	34 TFLOPS 	30 TFLOPS
TC64 	67 TFLOPS 	60 TFLOPS
FP32 	67 TFLOPS 	60 TFLOPS
TC32   989 TFLOPS  835 TFLOPS
TC16  1979 TFLOPS 1671 TFLOPS (bfloat and float)
TC8   3958 TFLOPS 3341 TFLOPS (Int and float)
Mem     80 GB 		94 GB
GPU Mem Bandwidth 3.35TB/s 3.9TB/s

https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/

# H200

H200 	SXM1 		H200 NVL		
FP64 	34 TFLOPS 	30 TFLOPS
TC64 	67 TFLOPS 	60 TFLOPS
FP32 	67 TFLOPS 	60 TFLOPS
TC32   989 TFLOPS  835 TFLOPS
TC16  1979 TFLOPS 1671 TFLOPS (bfloat and float)
TC8   3958 TFLOPS 3341 TFLOPS (Int and float)
GPU Memory 141GB 141GB

GPU Memory Bandwidth 4.8TB/s 4.8TB/s

Rough expected speed-up for FP64
P100 -> V100  7.8/4.7 = 1.65x (reached ?)
P100 -> A100  9.7/4.7 = 2x 	  (reached 1.7-2x)
P100 -> H100  34/4.7  = 7.23x (reached 5.4x)

