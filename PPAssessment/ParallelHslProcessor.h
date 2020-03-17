#pragma once

class ParallelHslProcessor {
private:
	cl::Program& Program;
	cl::Context& Context;
	cl::CommandQueue& Queue;
	CImg<unsigned short>& InputImage;
	unsigned int& BinSize;
	double& TotalDurationMs;
	unsigned int& ImageSize;
	unsigned short& MaxPixelValue;
	int& DeviceId;


	vector<float> ConvertRgbToHsl() {
		const unsigned int sizeOfImage = InputImage.size() * sizeof(unsigned short);
		const unsigned int sizeOfOutput = InputImage.size() * sizeof(float);

		// Create buffers for the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImage);
		cl::Buffer outputImageBuffer(Context, CL_MEM_READ_WRITE, sizeOfOutput);

		// Copy image data to image buffer on the device and wait for it to finish before continuing.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImage, &InputImage.data()[0]);

		// Create the kernel to use.
		cl::Kernel conversionKernel = cl::Kernel(Program, "RgbToHsl");
		// Set kernel arguments.
		conversionKernel.setArg(0, inputImageBuffer);
		conversionKernel.setArg(1, outputImageBuffer);
		conversionKernel.setArg(2, MaxPixelValue);
		conversionKernel.setArg(3, ImageSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;
		// Queue the kernel for execution on the device.
		Queue.enqueueNDRangeKernel(conversionKernel, cl::NullRange, cl::NDRange(ImageSize), cl::NullRange, NULL, &perfEvent);

		vector<float> outputData(InputImage.size());
		// Copy the result from the device to the host.
		Queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeOfOutput, &outputData.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tConvert RGB to HSL: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return outputData;
	}

	vector<unsigned short> ConvertHslToRgb(const vector<float>& inputImage) {
		const unsigned int sizeOfImage = inputImage.size() * sizeof(float);
		const unsigned int sizeOfOutput = inputImage.size() * sizeof(unsigned short);

		// Create buffers for the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImage);
		cl::Buffer outputImageBuffer(Context, CL_MEM_READ_WRITE, sizeOfOutput);

		// Copy image data to image buffer on the device and wait for it to finish before continuing.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImage, &inputImage.data()[0]);

		// Create the kernel to use.
		cl::Kernel conversionKernel = cl::Kernel(Program, "HslToRgb");
		// Set kernel arguments.
		conversionKernel.setArg(0, inputImageBuffer);
		conversionKernel.setArg(1, outputImageBuffer);
		conversionKernel.setArg(2, MaxPixelValue);
		conversionKernel.setArg(3, ImageSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;
		// Queue the kernel for execution on the device.
		Queue.enqueueNDRangeKernel(conversionKernel, cl::NullRange, cl::NDRange(ImageSize), cl::NullRange, NULL, &perfEvent);

		vector<unsigned short> outputData(inputImage.size());
		// Copy the result from the device to the host.
		Queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeOfOutput, &outputData.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tConvert HSL to RGB: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return outputData;
	}

	vector<unsigned int> BuildImageHistogramHsl(const vector<float>& inputImage, size_t& sizeOfHistogram) {
		// Calculate the number of bins needed.
		const unsigned int numberOfBins = ceil(99 / BinSize) + 1;

		// Initialise a vector for the histogram with the appropriate bin size. Add one because this is capacity not maximum index.
		vector<unsigned int> hist(numberOfBins);

		// Calculate the size of the histogram in bytes - used for buffer allocation.
		sizeOfHistogram = hist.size() * sizeof(unsigned int);
		const unsigned int sizeOfImage = ImageSize * sizeof(float);

		// Create buffers for the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImage);
		cl::Buffer histogramBuffer(Context, CL_MEM_READ_WRITE, sizeOfHistogram);

		// Copy image data for the luminance channel to image buffer on the device and wait for it to finish before continuing.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImage, &inputImage.data()[(ImageSize * 2) - 1]);

		// Create the kernel to use.
		cl::Kernel histogramKernel = cl::Kernel(Program, "histogramAtomicHsl");
		// Set kernel arguments.
		histogramKernel.setArg(0, inputImageBuffer);
		histogramKernel.setArg(1, histogramBuffer);
		histogramKernel.setArg(2, BinSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;
		// Queue the kernel for execution on the device. Only run through one channel - the luminance channel.
		Queue.enqueueNDRangeKernel(histogramKernel, 0, cl::NDRange(ImageSize), cl::NullRange, NULL, &perfEvent);

		// Copy the result from the device to the host.
		Queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tBuild Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return hist;
	}

	vector<float> NormaliseToLookupTableHsl(const size_t& sizeOfHistogram, const vector<unsigned int>& histogram) {
		const size_t sizeOfOutput = histogram.size() * sizeof(float);

		// Create buffers for the histogram.
		cl::Buffer histogramInputBuffer(Context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer histogramOutputBuffer(Context, CL_MEM_READ_WRITE, sizeOfOutput);

		// Get the maximum value from the histogram. Because it is cumulative, it is just the last value.
		const unsigned int maxHistValue = histogram[histogram.size() - 1];

		// Copy histogram data to device buffer memory.
		Queue.enqueueWriteBuffer(histogramInputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

		// Create the kernel.
		cl::Kernel lutKernel = cl::Kernel(Program, "normaliseToLutHsl");

		// Set the kernel arguments.
		lutKernel.setArg(0, histogramInputBuffer);
		lutKernel.setArg(1, maxHistValue);
		lutKernel.setArg(2, histogramOutputBuffer);

		// Create  an event for performance tracking.
		cl::Event perfEvent;

		// Queue the kernel for execution on the device.
		Queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

		vector<float> outputLut(histogram.size());
		// Copy the result from the output buffer on the device to the host.
		Queue.enqueueReadBuffer(histogramOutputBuffer, CL_TRUE, 0, sizeOfOutput, &outputLut.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tNormalise to lookup: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return outputLut;
	}

	vector<float> BackprojectionHsl(const vector<float>& inputImage, const vector<float>& histogram, const unsigned int& binSize, const unsigned int& imageSize, double& totalDurationMs) {

		const unsigned int sizeOfHistogram = sizeof(float) * histogram.size();
		const unsigned int sizeOfImage = imageSize * sizeof(float);

		// Create buffers to store the data on the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImage);
		cl::Buffer inputHistBuffer(Context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer outputImageBuffer(Context, CL_MEM_READ_WRITE, sizeOfImage);

		// Write the data for the input image and histogram lookup table to the buffers.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImage, &inputImage.data()[(imageSize * 2) - 1]);
		Queue.enqueueWriteBuffer(inputHistBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

		// Create the kernel
		cl::Kernel backPropKernel = cl::Kernel(Program, "backprojectionHsl");

		// Set the kernel arguments.
		backPropKernel.setArg(0, inputImageBuffer);
		backPropKernel.setArg(1, inputHistBuffer);
		backPropKernel.setArg(2, outputImageBuffer);
		backPropKernel.setArg(3, binSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;

		// Execute the kernel on the device.
		Queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(imageSize), cl::NullRange, NULL, &perfEvent);

		// Create the vector to store the output data.
		vector<float> outputData(imageSize);

		// Copy the output from the device buffer to the output vector on the host.
		Queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeOfImage, &outputData.data()[0]);

		// Create an output image data with the hue and saturation channels from the input.
		vector<float>::const_iterator first = inputImage.begin();
		vector<float>::const_iterator last = inputImage.begin() + (imageSize * 2);
		vector<float> outputImageData(first, last);

		// Append new luminance channel.
		outputImageData.insert(outputImageData.end(), outputData.begin(), outputData.end());

		totalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tBackprojection: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return outputImageData;
	}

public:
	ParallelHslProcessor(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, CImg<unsigned short>& inputImage, unsigned int& binSize, double& totalDurationMs, unsigned int& imageSize, unsigned short& maxPixelValue, int& deviceId) :
		Program(program),
		Context(context),
		Queue(queue),
		InputImage(inputImage),
		BinSize(binSize),
		TotalDurationMs(totalDurationMs),
		ImageSize(imageSize),
		MaxPixelValue(maxPixelValue),
		DeviceId(deviceId) {}

	CImg<unsigned short> RunHistogramEqalisation() {
		cout << endl << "Running parallel Histogram Equalisation with colour preservation..." << endl;

		// Convert the input RGB image to HSL colour space.
		vector<float> hslImage = ConvertRgbToHsl();

		// Build a histogram on the luminance channel.
		size_t sizeOfHistogram;
		vector<unsigned int> hist = BuildImageHistogramHsl(hslImage, sizeOfHistogram);

		// Cumulative sum the histogram.
		hist = SharedParallel::CumulativeSumParallel(Program, Context, Queue, DeviceId, hist, TotalDurationMs);

		// Normalise and create a lookup table from the cumulative histogram.
		vector<float> hslHist = NormaliseToLookupTableHsl(sizeOfHistogram, hist);

		// Backproject with the lookup table histogram.
		vector<float> backProjection = BackprojectionHsl(hslImage, hslHist, BinSize, ImageSize, TotalDurationMs);

		// Convert back to RGB.
		vector<unsigned short> outputData = ConvertHslToRgb(backProjection);

		cout << endl << "Total HSL Kernel Duration: " << TotalDurationMs << "ms" << endl;

		// Create the image from the output data.
		CImg<unsigned short> outputImage(outputData.data(), InputImage.width(), InputImage.height(), InputImage.depth(), InputImage.spectrum());

		return outputImage;
	}


};