#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

void print_help() {
	cout << "Application usage:" << endl;

	cout << "  -p : select platform " << endl;
	cout << "  -d : select device" << endl;
	cout << "  -l : list all platforms and devices" << endl;
	cout << "  -h : print this message" << endl;
}

vector<unsigned int> BuildImageHistogram(const cl::Program &program, const cl::Context &context, const cl::CommandQueue queue, const unsigned int &binSize, const CImg<unsigned char> &image, const unsigned char &colourChannel, size_t &sizeOfHistogram) {
	const unsigned int numberOfBins = ceil(255 / binSize)+1;

	// Initialise a vector for the histogram with the appropriate bin size. Add one because this is capacity not maximum index.
	vector<unsigned int> hist(numberOfBins);

	const unsigned int imageSize = image.width() * image.height();

	// Get the selected colour channel data out of the image.
	CImg<unsigned char>::const_iterator first = image.begin() + (imageSize*colourChannel);
	CImg<unsigned char>::const_iterator last = image.begin() + (imageSize * colourChannel) + imageSize;
	vector<unsigned char> imageColourChannelData(first, last);

	// Calculate the size of the histogram in bytes - used for buffer allocation.
	sizeOfHistogram = hist.size() * sizeof(unsigned int);

	// Create buffers for the device.
	cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, imageColourChannelData.size());
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy image data to image buffer on the device and wait for it to finish before continuing.
	queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, imageColourChannelData.size(), &imageColourChannelData.data()[0]);

	// Create the kernel to use.
	cl::Kernel histogramKernel = cl::Kernel(program, "histogramAtomic");
	// Set kernel arguments.
	histogramKernel.setArg(0, inputImageBuffer);
	histogramKernel.setArg(1, histogramBuffer);
	histogramKernel.setArg(2, binSize);

	// Create  an event for performance tracking.
	cl::Event perfEvent;
	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(imageColourChannelData.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result from the device to the host.
	queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

	// Print out the performance values.
	cout << "\tBuild Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

	return hist;
}

void AccumulateHistogramHillisSteele(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t& sizeOfHistogram, vector<unsigned int> &histogram) {

	// Create buffer for the histogram.
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);
	// Hillis-Steele needs a blank copy of the input for cache purposes.
	cl::Buffer histogramTempBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy the histogram data to the histogram buffer on the device. Wait for it to finish before continuing.
	queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Create the kernel for summing.
	cl::Kernel cumulitiveSumKernel = cl::Kernel(program, "scanHillisSteele");
	// Set kernel arguments.
	cumulitiveSumKernel.setArg(0, histogramBuffer);
	cumulitiveSumKernel.setArg(1, histogramTempBuffer);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(cumulitiveSumKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result back to the host from the device.
	queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Print out the performance values.
	cout << "\tAccumulate Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
}

void AccumulateHistogramBlelloch(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t& sizeOfHistogram, vector<unsigned int>& histogram) {

	// Create buffer for the histogram.
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy the histogram data to the histogram buffer on the device. Wait for it to finish before continuing.
	queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Create the kernel for summing.
	cl::Kernel cumulitiveSumKernel = cl::Kernel(program, "scanBlelloch");
	// Set kernel arguments.
	cumulitiveSumKernel.setArg(0, histogramBuffer);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(cumulitiveSumKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result back to the host from the device.
	queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Print out the performance values.
	cout << "\tAccumulate Histogram (Blelloch): " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
}

void NormaliseToLookupTable(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t &sizeOfHistogram, vector<unsigned int> &histogram) {
	
	// Create buffers for the histogram.
	cl::Buffer histogramInputBuffer(context, CL_MEM_READ_ONLY, sizeOfHistogram);
	cl::Buffer histogramOutputBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Get the maximum value from the histogram. Because it is cumulative, it is just the last value.
	const unsigned int maxHistValue = histogram[histogram.size() - 1];

	// Copy histogram data to device buffer memory.
	queue.enqueueWriteBuffer(histogramInputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Create the kernel.
	cl::Kernel lutKernel = cl::Kernel(program, "normaliseToLut");

	// Set the kernel arguments.
	lutKernel.setArg(0, histogramInputBuffer);
	lutKernel.setArg(1, maxHistValue);
	lutKernel.setArg(2, histogramOutputBuffer);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result from the output buffer on the device to the host.
	queue.enqueueReadBuffer(histogramOutputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Print out the performance values.
	cout << "\tNormalise to lookup: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
}

vector<unsigned char> Backprojection(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const CImg<unsigned char>& inputImage, const vector<unsigned int> &histogram, const size_t &sizeOfHistogram, const unsigned int &binSize, const unsigned char &colourChannel) {

	const unsigned int imageSize = inputImage.width() * inputImage.height();

	// Get the selected colour channel data out of the image.
	CImg<unsigned char>::const_iterator first = inputImage.begin() + (imageSize * colourChannel);
	CImg<unsigned char>::const_iterator last = inputImage.begin() + (imageSize * colourChannel) + imageSize;
	vector<unsigned char> imageColourChannelData(first, last);


	// Create buffers to store the data on the device.
	cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, imageSize);
	cl::Buffer inputHistBuffer(context, CL_MEM_READ_ONLY, sizeOfHistogram);
	cl::Buffer outputImageBuffer(context, CL_MEM_READ_WRITE, imageSize);

	// Write the data for the input image and histogram lookup table to the buffers.
	queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, imageSize, &imageColourChannelData.data()[0]);
	queue.enqueueWriteBuffer(inputHistBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	// Create the kernel
	cl::Kernel backPropKernel = cl::Kernel(program, "backprojection");

	// Set the kernel arguments.
	backPropKernel.setArg(0, inputImageBuffer);
	backPropKernel.setArg(1, inputHistBuffer);
	backPropKernel.setArg(2, outputImageBuffer);
	backPropKernel.setArg(3, binSize);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Execute the kernel on the device.
	queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(imageSize), cl::NullRange, NULL, &perfEvent);

	// Create the vector to store the output data.
	vector<unsigned char> outputData(imageSize);

	// Copy the output from the device buffer to the output vector on the host.
	queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, outputData.size(), &outputData.data()[0]);

	// Print out the performance values.
	cout << "\tBackprojection: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

	return outputData;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "E:/Dev/ParallelAssessment/Images/test_large.ppm";
	unsigned int binSize = 1;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	try {
		 
		// Get OpenCL context for the selected platform and device.
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device.
		cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code.
		cl::Program::Sources sources;

		// Load the kernels source.
		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error & err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Read image from file.
		CImg<unsigned char> inputImage(image_filename.c_str());

		// Display input image.
		CImgDisplay displayInput(inputImage, "input");

		size_t sizeOfHistogram;

		// Work out the image size of a single channel.
		const unsigned int imageSize = inputImage.width() * inputImage.height();

		// Allocate a vector to store all channels of the output image.
		vector<unsigned char> outputImageData(inputImage.size());

		for (unsigned char colourChannel = 0; colourChannel < inputImage.spectrum(); colourChannel++) {
			cout << endl << "Processing Colour Channel " << (int)colourChannel << endl << endl;

			// Build a histogram from the input image and get its size out.
			vector<unsigned int> hist = BuildImageHistogram(program, context, queue, binSize, inputImage, colourChannel, sizeOfHistogram);
		
			// Run cumulative sum on the histogram.
			AccumulateHistogramBlelloch(program, context, queue, sizeOfHistogram, hist);
		
			// Normalise and create a lookup table from the cumulative histogram.
			NormaliseToLookupTable(program, context, queue, sizeOfHistogram, hist);

			// Backproject with the cumulative histogram.
			vector<unsigned char> outputData = Backprojection(program, context, queue, inputImage, hist, sizeOfHistogram, binSize, colourChannel);

			// Copy the channel to the output image data vector in the appropriate channel position.
			std::copy(outputData.begin(), outputData.end(), outputImageData.begin() + (imageSize * colourChannel));
		}

		// Create the image from the output data.
		CImg<unsigned char> outputImage(outputImageData.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());

		
		CImgDisplay displayOutput(outputImage, "output");

		while (!displayInput.is_closed() && !displayOutput.is_closed()
			&& !displayInput.is_keyESC() && !displayOutput.is_keyESC()) {
			displayInput.wait(1);
			displayOutput.wait(1);
		}
	}
	catch (const cl::Error & err) {
		std::cout << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException & err) {
		std::cout << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}