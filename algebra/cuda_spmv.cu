// https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-api-reference

cusparseStatus_t
cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 csrRowOffsets,
                  void*                 csrColInd,
                  void*                 csrValues,
                  cusparseIndexType_t   csrRowOffsetsType,
                  cusparseIndexType_t   csrColIndType,
                  cusparseIndexBase_t   idxBase,
                  cudaDataType          valueType)

cusparseStatus_t
cusparseCreateConstCsr(cusparseSpMatDescr_t* spMatDescr, //const descriptor
                       int64_t               rows,
                       int64_t               cols,
                       int64_t               nnz,
                       void*                 csrRowOffsets,
                       void*                 csrColInd,
                       void*                 csrValues,
                       cusparseIndexType_t   csrRowOffsetsType,
                       cusparseIndexType_t   csrColIndType,
                       cusparseIndexBase_t   idxBase,
                       cudaDataType          valueType)



cusparseStatus_t
cusparseSpMV_bufferSize(cusparseHandle_t     handle,
                        cusparseOperation_t  opA,
                        const void*          alpha,
                        cusparseSpMatDescr_t matA, //const descriptor
                        cusparseDnVecDescr_t vecX, //const descriptor
                        const void*          beta,
                        cusparseDnVecDescr_t vecY,
                        cudaDataType         computeType,
                        cusparseSpMVAlg_t    alg,
                        size_t*              bufferSize)


cusparseStatus_t
cusparseSpMV(cusparseHandle_t     handle,
             cusparseOperation_t  opA,
             const void*          alpha,
             cusparseSpMatDescr_t matA, //const descriptor
             cusparseDnVecDescr_t vecX, //const descriptor
             const void*          beta,
             cusparseDnVecDescr_t vecY,
             cudaDataType         computeType,
             cusparseSpMVAlg_t    alg,
             void*                externalBuffer)


https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm