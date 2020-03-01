kernel void histogramAtomic(global const uchar* inputImage, global uint* histogram, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	int binIndex = inputImage[id] / binSize;

	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
}

kernel void scanHillisSteele(global int* inputHistogram, global int* outputHistogram) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* temp;
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

kernel void normaliseToLut(global const int* inputHistogram, const int maxValue, global int* outputHistogram) {
	int id = get_global_id(0);

	// Calculate the normalised value between 0 and 1. We cast to a double to avoid integer rounding occurring.
	double normalised = (double)inputHistogram[id] / maxValue;
	// Scale the normalised value back up to the scale of unsigned char.
	int scaled = normalised * 255;
	outputHistogram[id] = scaled;
}

kernel void backprojection(global const uchar* inputImage, global const int* inputHistogram, global uchar* outputImage, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	int binIndex = inputImage[id] / binSize;

	outputImage[id] = inputHistogram[binIndex];
}