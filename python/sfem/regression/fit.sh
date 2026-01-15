#!/usr/bin/env bash

set -e

python regression_prony.py --dataset qlv --fit-mr-first --alpha 0 --top-k 6 --init-from-dma --verbose  --numba --plot



# [sfem] MR fit first: C10=0.2458253554614771 MPa  C01=0.6565743299365601 MPa
# [sfem] QLV fit: b0=0.0795511  g_inf=1.31246  nnz=37/400
# [sfem] Top 6 g_k:
#     g[286]=0.271113784106657  tau=2942.2 s
#     g[287]=0.164222964906181  tau=3226.8 s
#     g[254]=-0.1468189590538714  tau=153.272 s
#     g[231]=-0.1280198992164453  tau=18.3298 s
#     g[255]=-0.10059242086923  tau=168.099 s
#     g[230]=-0.06396396628507869  tau=16.7131 s
# [sfem] Mode=uniax RMSE=0.0887171 MPa
# [sfem] Mode=equibiax RMSE=0.185017 MPa
# [sfem] Mode=pureshear RMSE=0.117953 MPa