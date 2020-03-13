kernel void histogramAtomic(global const ushort* inputImage, global uint* histogram, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;
	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
}

kernel void normaliseToLut(global const uint* inputHistogram, const uint maxValue, global uint* outputHistogram, const ushort maxPixelValue) {
	int id = get_global_id(0);

	// Calculate the normalised value between 0 and 1. We cast to a double to avoid integer rounding occurring.
	double normalised = (double)inputHistogram[id] / maxValue;
	// Scale the normalised value back up to the scale of the image.
	uint scaled = normalised * maxPixelValue;
	outputHistogram[id] = scaled;
}


kernel void backprojection(global const ushort* inputImage, global const uint* inputHistogram, global ushort* outputImage, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;

	outputImage[id] = inputHistogram[binIndex];
}
