kernel void histogramAtomic(global const uchar* inputImage, global int* histogram, const int binSize) {
	int id = get_global_id(0);

	// Get the bin index, this integer division is always floored toward zero.
	int binIndex = inputImage[id] / binSize;

	// Atomically increment the value at this bin index.
	atomic_inc(&histogram[binIndex]);
}

kernel void scan_hs(global int* inputHistogram, global int* outputHistogram) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* temp;
	for (int stride = 1; stride < N; stride *= 2) {
		outputHistogram[id] = inputHistogram[id];
		if (id >= stride)
		{
			outputHistogram[id] += inputHistogram[id - stride];
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		temp = inputHistogram;
		inputHistogram = outputHistogram;
		outputHistogram = temp; //swap A & B between steps
	}
}

kernel void lut(global const int* inputHistogram, const int maxValue, global int* outputHistogram) {
	int id = get_global_id(0);
	//printf("Input Val: \%d, maxValue: \%d, result = \%d\n", inputHistogram[id], maxValue, (double)inputHistogram[id] / maxValue);
	double normalised = (double)inputHistogram[id] / maxValue;
	int scaled = normalised * 255;
	outputHistogram[id] = scaled;
}

kernel void backprojection(global const uchar* inputImage, global const int* inputHistogram, global uchar* outputImage) {
	int id = get_global_id(0);
	outputImage[id] = inputHistogram[inputImage[id]];
}



kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = 255 - A[id];
}

kernel void rgb2grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	if (colour_channel == 0) {
		uchar Y = 0.2126 * A[id] + 0.7152 * A[id + image_size] + 0.0722 * A[id + (image_size * 2)];
		B[id] = Y;
		B[id + image_size] = Y;
		B[id + (image_size * 2)] = Y;
	}

	// 46% slower.
	/*if (colour_channel == 0) {
		uchar Y = 0.2126 * A[id] + 0.7152 * A[id + image_size] + 0.0722 * A[id + (image_size * 2)];
		B[id] = Y;
	}
	else if (colour_channel == 1) {
		uchar Y = 0.2126 * A[id - image_size] + 0.7152 * A[id] + 0.0722 * A[id + image_size];
		B[id] = Y;
	}
	else if (colour_channel == 2) {
		uchar Y = 0.2126 * A[id - (image_size * 2)] + 0.7152 * A[id - image_size] + 0.0722 * A[id];
		B[id] = Y;
	}*/
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	uint result = 0;

	for (int i = x; i <= (x + 4); i++) {
		for (int j = y; j <= (y + 4); j++) {
			int curX = i;
			int curY = j;
			if (i >= width) {
				curX = width - 1;
			}
			if (j >= height) {
				curY = height - 1;
			}
			result += A[curX + curY * width + c * image_size];
		}
	}

	result /= 25;

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, global const float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	float result = 0;

	for (int i = x; i <= (x + 2); i++) {
		for (int j = y; j <= (y + 2); j++) {
			int curX = i;
			int curY = j;
			if (i >= width) {
				curX = width - 1;
			}
			if (j >= height) {
				curY = height - 1;
			}
			result += A[curX + curY * width + c * image_size] * mask[i - (x - 1) + j - (y - 1)];
		}
	}

	B[id] = (uchar)result;
}