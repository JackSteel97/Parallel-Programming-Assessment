kernel void histogramAtomic(global const ushort* inputImage, global uint* histogram, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;	
	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
}

kernel void histogramAtomicHsl(global const float* inputImage, global uint* histogram, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, truncate towards zero.
	uint binIndex = (uint)trunc(inputImage[id] / binSize);

	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
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

kernel void normaliseToLut(global const uint* inputHistogram, const uint maxValue, global uint* outputHistogram, const ushort maxPixelValue) {
	int id = get_global_id(0);

	// Calculate the normalised value between 0 and 1. We cast to a double to avoid integer rounding occurring.
	double normalised = (double)inputHistogram[id] / maxValue;
	// Scale the normalised value back up to the scale of the image.
	uint scaled = normalised * maxPixelValue;
	outputHistogram[id] = scaled;
}

kernel void normaliseToLutHsl(global const uint* inputHistogram, const uint maxValue, global float* outputHistogram) {
	int id = get_global_id(0);

	// Calculate the normalised value between 0 and 1. We cast to a double to avoid integer rounding occurring.
	double normalised = (double)inputHistogram[id] / maxValue;
	// Scale the normalised value back up.
	double scaled = normalised * 100;
	outputHistogram[id] = scaled;
}

kernel void backprojection(global const ushort* inputImage, global const uint* inputHistogram, global ushort* outputImage, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	uint binIndex = inputImage[id] / binSize;

	outputImage[id] = inputHistogram[binIndex];
}

kernel void backprojectionHsl(global const float* inputImage, global const float* inputHistogram, global float* outputImage, const uint binSize) {
	int id = get_global_id(0);

	// Get the bin index, truncate towards zero.
	uint binIndex = (uint)trunc(inputImage[id] / binSize);

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

kernel void RgbToHsl(global const ushort* inputImage, global float* outputImage, const ushort maxPixelValue, const uint imageSize) {
	int id = get_global_id(0);

	// Get RGB values for this pixel.
	// Normalise to a fraction of 1 - cast to avoid rounding issues.
	float r = (float)inputImage[id] / maxPixelValue;
	float g = (float)inputImage[id + imageSize] / maxPixelValue;
	float b = (float)inputImage[id + (imageSize * 2)] / maxPixelValue;

	// Find the smallest value.
	float cMin = min(min(r, g), b);
	// Find the largest value.
	float cMax = max(max(r, g), b);

	float delta = cMax - cMin;

	float h = 0, s = 0, l = 0;

	if (delta == 0) {
		h = 0;
	}
	else if (cMax == r) {
		// Red is max.
		h = fmod(((g - b) / delta), 6);
	}
	else if (cMax == g) {
		// Green is max.
		h = (b - r) / delta + 2;
	}
	else {
		// Blue is max.
		h = (r - g) / delta + 4;
	}

	h = round(h * 60);

	// Make negative hues positive
	if (h < 0) {
		h += 360;
	}

	// Calculate lightness.
	l = (cMax + cMin) / 2;

	// Calculate saturation.
	if (delta == 0) {
		s = 0;
	}
	else {
		if (l < 0.5) {
			s = (cMax - cMin) / (cMax + cMin);
		}
		else {
			s = (cMax - cMin) / (2.0 - cMax - cMin);
		}
	}

	

	// Convert to percentage with one decimal place.
	//s = (round(s * 10) / 10)*100;
	//l = (round(l * 10) / 10)*100;
	s = s * 100;
	l = l * 100;
	// Set in corresponding channels out output.
	outputImage[id] = h;
	outputImage[id + imageSize] = s;
	outputImage[id + (imageSize * 2)] = l;
}

kernel void HslToRgb(global const float* inputImage, global ushort* outputImage, const ushort maxPixelValue, const uint imageSize) {
	int id = get_global_id(0);

	float h = inputImage[id];
	float s = inputImage[id + imageSize];
	float l = inputImage[id + (imageSize * 2)];

	// Normalise to between 0 and 1.
	s /= 100;
	l /= 100;

	float c = (1 - fabs((float)(2 * l - 1))) * s;
	float x = c * (1 - fabs((float)fmod(((float)(h / 60)), 2) - 1));
	float m = l - c / 2;

	float r = 0, g = 0, b = 0;

	if (0 <= h && h < 60) {
		r = c; g = x; b = 0;
	}
	else if (60 <= h && h < 120) {
		r = x; g = c; b = 0;
	}
	else if (120 <= h && h < 180) {
		r = 0; g = c; b = x;
	}
	else if (180 <= h && h < 240) {
		r = 0; g = x; b = c;
	}
	else if (240 <= h && h < 300) {
		r = x; g = 0; b = c;
	}
	else if (300 <= h && h < 360) {
		r = c; g = 0; b = x;
	}

	r = round((r + m) * maxPixelValue);
	g = round((g + m) * maxPixelValue);
	b = round((b + m) * maxPixelValue);

	outputImage[id] = (ushort)r;
	outputImage[id + imageSize] = (ushort)g;
	outputImage[id + (imageSize * 2)] = (ushort)b;
}