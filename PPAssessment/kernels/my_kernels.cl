kernel void histogramAtomic(global const ushort* inputImage, global uint* histogram, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;	
	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
}

kernel void scanHillisSteele(global uint* inputHistogram, global uint* outputHistogram) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global uint* temp;
	for (int stride = 1; stride < N; stride *= 2) {
		outputHistogram[id] = inputHistogram[id];
		if (id >= stride)
		{
			outputHistogram[id] += inputHistogram[id - stride];
		}

		// Sync the step.
		barrier(CLK_GLOBAL_MEM_FENCE);

		// Swap the input and output between steps.
		temp = inputHistogram;
		inputHistogram = outputHistogram;
		outputHistogram = temp;
	}
}

kernel void normaliseToLut(global const uint* inputHistogram, const uint maxValue, global uint* outputHistogram, const ushort maxPixelValue) {
	int id = get_global_id(0);

	// Calculate the normalised value between 0 and 1. We cast to a double to avoid integer rounding occurring.
	double normalised = (double)inputHistogram[id] / maxValue;
	// Scale the normalised value back up to the scale of unsigned char.
	uint scaled = normalised * maxPixelValue;
	outputHistogram[id] = scaled;
}

kernel void backprojection(global const ushort* inputImage, global const uint* inputHistogram, global ushort* outputImage, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;

	outputImage[id] = inputHistogram[binIndex];
}

kernel void scanBlelloch(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const uint* A, global uint* B, local uint* scratch_1, local uint* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local uint* scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}