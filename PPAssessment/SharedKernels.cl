// A double-buffered version of the Hillis-Steele inclusive scan
// Requires two additional input arguments which correspond to two local buffers
kernel void scanHillisSteeleBuffered(global const uint* input, global uint* output, local uint* temp1, local uint* temp2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Used for buffer swap.
	local uint* tempHolder;

	// Cache all N values from global memory to local memory.
	temp1[lid] = input[id];

	// Wait for all local threads to finish copying from global to local memory.
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
		{
			temp2[lid] = temp1[lid] + temp1[lid - i];
		}
		else
		{
			temp2[lid] = temp1[lid];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Buffer swap
		tempHolder = temp2;
		temp2 = temp1;
		temp1 = tempHolder;
	}

	// Copy the cache to output array.
	output[id] = temp1[lid];
}

// Calculates the block sums.
kernel void blockSum(global const uint* input, global uint* output, const uint localSize) {
	int id = get_global_id(0);
	output[id] = input[(id + 1) * localSize - 1];
}

// Adjust the values stored in partial scans by adding block sums to corresponding blocks.
kernel void scanAddAdjust(global uint* partialScanResult, global const uint* blockSums) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	partialScanResult[id] += blockSums[gid];
}

kernel void scanHillisSteele(global uint* inputHistogram) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int stride = 1; stride < N; stride *= 2) {
		if (id >= stride)
		{
			inputHistogram[id] += inputHistogram[id - stride];
		}

		// Sync the step.
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}