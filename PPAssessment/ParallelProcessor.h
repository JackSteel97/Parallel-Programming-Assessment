#pragma once

class ParallelProcessor {
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

	vector<unsigned int> BuildImageHistogram(const vector<unsigned short>& imageColourChannelData, const size_t& sizeOfImageChannel, const unsigned char& colourChannel, size_t& sizeOfHistogram) {

		// Calculate the number of bins needed.
		const unsigned int numberOfBins = ceil((MaxPixelValue+1) / BinSize);

		// Initialise a vector for the histogram with the appropriate bin size. Add one because this is capacity not maximum index.
		vector<unsigned int> hist(numberOfBins);

		// Calculate the size of the histogram in bytes - used for buffer allocation.
		sizeOfHistogram = hist.size() * sizeof(unsigned int);

		// Create buffers for the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImageChannel);
		cl::Buffer histogramBuffer(Context, CL_MEM_READ_WRITE, sizeOfHistogram);

		// Copy image data to image buffer on the device and wait for it to finish before continuing.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &imageColourChannelData.data()[0]);

		// Create the kernel to use.
		cl::Kernel histogramKernel = cl::Kernel(Program, "histogramAtomic");
		// Set kernel arguments.
		histogramKernel.setArg(0, inputImageBuffer);
		histogramKernel.setArg(1, histogramBuffer);
		histogramKernel.setArg(2, BinSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;
		// Queue the kernel for execution on the device.
		Queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(imageColourChannelData.size()), cl::NullRange, NULL, &perfEvent);

		// Copy the result from the device to the host.
		Queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tBuild Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return hist;
	}

	void NormaliseToLookupTable(const size_t& sizeOfHistogram, vector<unsigned int>& histogram) {

		// Create buffers for the histogram.
		cl::Buffer histogramInputBuffer(Context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer histogramOutputBuffer(Context, CL_MEM_READ_WRITE, sizeOfHistogram);

		// Get the maximum value from the histogram. Because it is cumulative, it is just the last value.
		const unsigned int maxHistValue = histogram[histogram.size() - 1];

		// Copy histogram data to device buffer memory.
		Queue.enqueueWriteBuffer(histogramInputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

		// Create the kernel.
		cl::Kernel lutKernel = cl::Kernel(Program, "normaliseToLut");

		// Set the kernel arguments.
		lutKernel.setArg(0, histogramInputBuffer);
		lutKernel.setArg(1, maxHistValue);
		lutKernel.setArg(2, histogramOutputBuffer);
		lutKernel.setArg(3, MaxPixelValue);

		// Create  an event for performance tracking.
		cl::Event perfEvent;

		// Queue the kernel for execution on the device.
		Queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

		// Copy the result from the output buffer on the device to the host.
		Queue.enqueueReadBuffer(histogramOutputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tNormalise to lookup: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
	}


	vector<unsigned short> Backprojection(const vector<unsigned short>& imageColourChannelData, const size_t sizeOfImageChannel, const vector<unsigned int>& histogram, const size_t& sizeOfHistogram, const unsigned char& colourChannel) {

		// Create buffers to store the data on the device.
		cl::Buffer inputImageBuffer(Context, CL_MEM_READ_ONLY, sizeOfImageChannel);
		cl::Buffer inputHistBuffer(Context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer outputImageBuffer(Context, CL_MEM_READ_WRITE, sizeOfImageChannel);

		// Write the data for the input image and histogram lookup table to the buffers.
		Queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &imageColourChannelData.data()[0]);
		Queue.enqueueWriteBuffer(inputHistBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

		// Create the kernel
		cl::Kernel backPropKernel = cl::Kernel(Program, "backprojection");

		// Set the kernel arguments.
		backPropKernel.setArg(0, inputImageBuffer);
		backPropKernel.setArg(1, inputHistBuffer);
		backPropKernel.setArg(2, outputImageBuffer);
		backPropKernel.setArg(3, BinSize);

		// Create  an event for performance tracking.
		cl::Event perfEvent;

		// Execute the kernel on the device.
		Queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(imageColourChannelData.size()), cl::NullRange, NULL, &perfEvent);

		// Create the vector to store the output data.-
		vector<unsigned short> outputData(imageColourChannelData.size());

		// Copy the output from the device buffer to the output vector on the host.
		Queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &outputData.data()[0]);

		TotalDurationMs += GetProfilingTotalTimeMs(perfEvent);

		// Print out the performance values.
		cout << "\tBackprojection: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

		return outputData;
	}

public:
	ParallelProcessor(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, CImg<unsigned short>& inputImage, unsigned int& binSize, double& totalDurationMs, unsigned int& imageSize, unsigned short& maxPixelValue, int& deviceId) :
		Program(program),
		Context(context),
		Queue(queue),
		InputImage(inputImage),
		BinSize(binSize),
		TotalDurationMs(totalDurationMs),
		ImageSize(imageSize),
		MaxPixelValue(maxPixelValue),
		DeviceId(deviceId) {}

	CImg<unsigned short> RunHistogramEqualisation() {
		cout << endl << "Running parallel Histogram Equalisation..." << endl;

		// Allocate a vector to store all channels of the output image.
		vector<unsigned short> outputImageData(InputImage.size());

		// Declare size of histogram (stored as number of bytes), the value is assigned by the build histogram method.
		size_t sizeOfHistogram, sizeOfImageChannel;

		for (unsigned char colourChannel = 0; colourChannel < InputImage.spectrum(); colourChannel++) {
			cout << endl << "Processing Colour Channel " << (int)colourChannel << endl;

			// Get the selected colour channel data out of the image.
			CImg<unsigned short>::const_iterator first = InputImage.begin() + (ImageSize * colourChannel);
			CImg<unsigned short>::const_iterator last = InputImage.begin() + (ImageSize * colourChannel) + ImageSize;
			vector<unsigned short> imageColourChannelData(first, last);

			sizeOfImageChannel = imageColourChannelData.size() * sizeof(unsigned short);

			// Build a histogram from the input image and get its size out.
			vector<unsigned int> hist = BuildImageHistogram(imageColourChannelData, sizeOfImageChannel, colourChannel, sizeOfHistogram);

			// Run cumulative sum on the histogram.
			hist = SharedParallel::CumulativeSumParallel(Program, Context, Queue, DeviceId, hist, TotalDurationMs);

			// Normalise and Create a lookup table from the cumulative histogram.
			NormaliseToLookupTable(sizeOfHistogram, hist);

			// BackProject with histogram lookup table.
			vector<unsigned short> outputData = Backprojection(imageColourChannelData, sizeOfImageChannel, hist, sizeOfHistogram, colourChannel);

			// Copy the channel to the output image data vector in the appropriate channel position.
			std::copy(outputData.begin(), outputData.end(), outputImageData.begin() + (ImageSize * colourChannel));
		}
		
		cout << endl << "Total Kernel Duration: " << TotalDurationMs << "ms" << endl;

		// Create the image from the output data.
		CImg<unsigned short> outputImage(outputImageData.data(), InputImage.width(), InputImage.height(), InputImage.depth(), InputImage.spectrum());

		return outputImage;
	}
};