metal_kernel = """
            #include <metal_stdlib>;
            using namespace metal;
            kernel void add(const device float *A [[buffer(0)]],
                                const device float *B [[buffer(1)]],
                                device float *C [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                C[id] = A[id] + B[id]; 
            }
            kernel void mul(const device float *A [[buffer(0)]],
                                const device float *B [[buffer(1)]],
                                device float *C [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                C[id] = A[id] * B[id]; 
            }
            kernel void addInt(const device int *A [[buffer(0)]],
                                const device int *B [[buffer(1)]],
                                device int *C [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                C[id] = A[id] + B[id]; 
            }
            kernel void mulInt(const device int *A [[buffer(0)]],
                                const device int *B [[buffer(1)]],
                                device int *C [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                C[id] = A[id] * B[id]; 
            }


"""