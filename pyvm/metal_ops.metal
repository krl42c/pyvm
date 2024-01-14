kernel void addNumbers(const device float* inputA [[buffer(0)]],
                       const device float* inputB [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = inputA[id] + inputB[id];
}