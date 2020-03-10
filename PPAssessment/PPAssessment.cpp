#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"
#include <chrono>  // for high_resolution_clock

using namespace cimg_library;
using namespace std;
using namespace chrono;

void print_help() {
	cout << "Application usage:" << endl;

	cout << "  -p : select platform " << endl;
	cout << "  -d : select device" << endl;
	cout << "  -l : list all platforms and devices" << endl;
	cout << "  -h : print this message" << endl;
}

double GetProfilingTotalTimeMs(const cl::Event& evnt) {
	return (evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / static_cast<double>(ProfilingResolution::PROF_MS);
}

vector<unsigned int> BuildImageHistogram(const cl::Program &program, const cl::Context &context, const cl::CommandQueue queue, const unsigned int &binSize, const vector<unsigned short> &imageColourChannelData, const unsigned char &colourChannel, size_t &sizeOfHistogram, double &totalDurationMs, const unsigned short &maxPixelValue, const size_t &sizeOfImageChannel) {

	// Calculate the number of bins needed.
	const unsigned int numberOfBins = ceil(maxPixelValue / binSize)+1;

	// Initialise a vector for the histogram with the appropriate bin size. Add one because this is capacity not maximum index.
	vector<unsigned int> hist(numberOfBins);

	// Calculate the size of the histogram in bytes - used for buffer allocation.
	sizeOfHistogram = hist.size() * sizeof(unsigned int);

	// Create buffers for the device.
	cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, sizeOfImageChannel);
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy image data to image buffer on the device and wait for it to finish before continuing.
	queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &imageColourChannelData.data()[0]);

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

	totalDurationMs += GetProfilingTotalTimeMs(perfEvent);

	// Print out the performance values.
	cout << "\tBuild Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

	return hist;
}

vector<unsigned int> CumulativeSumParallel(const cl::Program& program, const cl::Context& context, const cl::CommandQueue& queue, const unsigned int deviceId, vector<unsigned int> input) {
	const size_t outputCount = input.size();
	const size_t outputSize = outputCount * sizeof(unsigned int);

	cl::Kernel phase1Kernel = cl::Kernel(program, "scanHillisSteeleBuffered");

	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceId];

	const size_t localSize = phase1Kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
	const size_t localSizeBytes = localSize * sizeof(unsigned int);

	const size_t paddingSize = input.size() % localSize;

	if (paddingSize > 0) {
		const vector<unsigned int> paddingVector(localSize - paddingSize, 0);
		input.insert(input.end(), paddingVector.begin(), paddingVector.end());
	}

	const size_t inputCount = input.size();
	const size_t inputSize = input.size() * sizeof(unsigned int);
	const size_t numberOfGroups = inputCount / localSize;
	const size_t numberOfGroupsBytes = numberOfGroups * sizeof(unsigned int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &input[0]);
	queue.enqueueFillBuffer(outputBuffer, 0, 0, outputSize);

	phase1Kernel.setArg(0, inputBuffer);
	phase1Kernel.setArg(1, outputBuffer);
	phase1Kernel.setArg(2, cl::Local(localSizeBytes));
	phase1Kernel.setArg(3, cl::Local(localSizeBytes));

	queue.enqueueNDRangeKernel(phase1Kernel, cl::NullRange, cl::NDRange(inputCount), cl::NDRange(localSize));

	cl::Kernel phase2Kernel = cl::Kernel(program, "blockSum");

	cl::Buffer phase2OutputBuffer(context, CL_MEM_READ_WRITE, numberOfGroupsBytes);

	phase2Kernel.setArg(0, outputBuffer);
	phase2Kernel.setArg(1, phase2OutputBuffer);
	phase2Kernel.setArg(2, static_cast<int>(localSize));

	queue.enqueueNDRangeKernel(phase2Kernel, cl::NullRange, cl::NDRange(numberOfGroups), cl::NullRange);

	cl::Kernel phase25Kernel = cl::Kernel(program, "scanHillisSteele");

	phase25Kernel.setArg(0, phase2OutputBuffer);
	queue.enqueueNDRangeKernel(phase25Kernel, cl::NullRange, cl::NDRange(numberOfGroups), cl::NullRange);

	cl::Kernel phase3Kernel = cl::Kernel(program, "scanAddAdjust");

	phase3Kernel.setArg(0, outputBuffer);
	phase3Kernel.setArg(1, phase2OutputBuffer);

	queue.enqueueNDRangeKernel(phase3Kernel, cl::NDRange(localSize), cl::NDRange(inputCount-localSize), cl::NDRange(localSize));

	vector<unsigned int> outputData(outputCount);
	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &outputData[0]);

	return outputData;
}


void AccumulateHistogramHillisSteele(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t &sizeOfHistogram, vector<unsigned int> &histogram, double &totalDurationMs) {

	// Create buffer for the histogram.
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);
	// Hillis-Steele needs a blank copy of the input for cache purposes.
	cl::Buffer histogramTempBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy the histogram data to the histogram buffer on the device. Wait for it to finish before continuing.
	queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);
	queue.enqueueFillBuffer(histogramTempBuffer, 0, 0, sizeOfHistogram);
	// Create the kernel for summing.
	cl::Kernel cumulativeSumKernel = cl::Kernel(program, "scanHillisSteele");
	// Set kernel arguments.
	cumulativeSumKernel.setArg(0, histogramBuffer);
	cumulativeSumKernel.setArg(1, histogramTempBuffer);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(cumulativeSumKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NDRange(256), NULL, &perfEvent);

	// Copy the result back to the host from the device.
	queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	totalDurationMs += GetProfilingTotalTimeMs(perfEvent);

	auto result = std::max_element(histogram.begin(), histogram.end());

	// Print out the performance values.
	cout << "\tAccumulate Histogram: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
}

void AccumulateHistogramBlelloch(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t &sizeOfHistogram, vector<unsigned int>& histogram) {

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

void NormaliseToLookupTable(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t &sizeOfHistogram, vector<unsigned int> &histogram, double &totalDurationMs, const unsigned short &maxPixelValue) {
	
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
	lutKernel.setArg(3, maxPixelValue);

	// Create  an event for performance tracking.
	cl::Event perfEvent;

	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result from the output buffer on the device to the host.
	queue.enqueueReadBuffer(histogramOutputBuffer, CL_TRUE, 0, sizeOfHistogram, &histogram.data()[0]);

	totalDurationMs += GetProfilingTotalTimeMs(perfEvent);

	// Print out the performance values.
	cout << "\tNormalise to lookup: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;
}

vector<unsigned short> Backprojection(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const vector<unsigned short>& imageColourChannelData, const vector<unsigned int> &histogram, const size_t &sizeOfHistogram, const unsigned int &binSize, const unsigned char &colourChannel, double &totalDurationMs, const size_t sizeOfImageChannel) {

	// Create buffers to store the data on the device.
	cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, sizeOfImageChannel);
	cl::Buffer inputHistBuffer(context, CL_MEM_READ_ONLY, sizeOfHistogram);
	cl::Buffer outputImageBuffer(context, CL_MEM_READ_WRITE, sizeOfImageChannel);

	// Write the data for the input image and histogram lookup table to the buffers.
	queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &imageColourChannelData.data()[0]);
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
	queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(imageColourChannelData.size()), cl::NullRange, NULL, &perfEvent);

	// Create the vector to store the output data.
	vector<unsigned short> outputData(imageColourChannelData.size());

	// Copy the output from the device buffer to the output vector on the host.
	queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, sizeOfImageChannel, &outputData.data()[0]);

	totalDurationMs += GetProfilingTotalTimeMs(perfEvent);

	// Print out the performance values.
	cout << "\tBackprojection: " << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

	return outputData;
}

CImg<unsigned short> ParallelImplementation(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const CImg<unsigned short> &inputImage, const unsigned int& binSize, double& totalDurationMs, const unsigned short &maxPixelValue) {
	cout << "Running parallel implementation..." << endl;

	// Work out the image size of a single channel.
	const unsigned int imageSize = inputImage.width() * inputImage.height();

	// Allocate a vector to store all channels of the output image.
	vector<unsigned short> outputImageData(inputImage.size());

	// Declare size of histogram (stored as number of bytes), the value is assigned by the build histogram method.
	size_t sizeOfHistogram, sizeOfImageChannel;

	totalDurationMs = 0;

	for (unsigned char colourChannel = 0; colourChannel < inputImage.spectrum(); colourChannel++) {
		cout << endl << "Processing Colour Channel " << (int)colourChannel << endl;

		// Get the selected colour channel data out of the image.
		CImg<unsigned short>::const_iterator first = inputImage.begin() + (imageSize * colourChannel);
		CImg<unsigned short>::const_iterator last = inputImage.begin() + (imageSize * colourChannel) + imageSize;
		vector<unsigned short> imageColourChannelData(first, last);

		sizeOfImageChannel = imageColourChannelData.size() * sizeof(unsigned short);

		// Build a histogram from the input image and get its size out.
		vector<unsigned int> hist = BuildImageHistogram(program, context, queue, binSize, imageColourChannelData, colourChannel, sizeOfHistogram, totalDurationMs, maxPixelValue, sizeOfImageChannel);

		// Run cumulative sum on the histogram.
		hist = CumulativeSumParallel(program, context, queue, 0, hist);

		//AccumulateHistogramHillisSteele(program, context, queue, sizeOfHistogram, hist, totalDurationMs);

		// Normalise and create a lookup table from the cumulative histogram.
		NormaliseToLookupTable(program, context, queue, sizeOfHistogram, hist, totalDurationMs, maxPixelValue);

		// Backproject with the cumulative histogram.
		vector<unsigned short> outputData = Backprojection(program, context, queue, imageColourChannelData, hist, sizeOfHistogram, binSize, colourChannel, totalDurationMs, sizeOfImageChannel);

		// Copy the channel to the output image data vector in the appropriate channel position.
		std::copy(outputData.begin(), outputData.end(), outputImageData.begin() + (imageSize * colourChannel));
	}

	cout << endl << "Total Kernel Duration: " << totalDurationMs << "ms" << endl;

	// Create the image from the output data.
	CImg<unsigned short> outputImage(outputImageData.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());

	return outputImage;
}

CImg<unsigned short> SerialImplementation(const CImg<unsigned short>& inputImage, const unsigned int& binSize, double &totalDurationMs, const unsigned short &maxPixelValue) {
	
	const unsigned int imageSize = inputImage.width() * inputImage.height();

	vector<unsigned short> outputImageData(inputImage.size());

	const unsigned int numberOfBins = ceil(maxPixelValue / binSize) + 1;

	totalDurationMs = 0;

	time_point<high_resolution_clock> start, end;

	for (unsigned char colourChannel = 0; colourChannel < inputImage.spectrum(); colourChannel++) {
		vector<unsigned int> hist(numberOfBins);

		// Get the selected colour channel data out of the image.
		CImg<unsigned short>::const_iterator first = inputImage.begin() + (imageSize * colourChannel);
		CImg<unsigned short>::const_iterator last = inputImage.begin() + (imageSize * colourChannel) + imageSize;
		vector<unsigned short> imageColourChannelData(first, last);

		// Step one, build histogram.
		start = high_resolution_clock::now();
		for (unsigned int i = 0; i < imageColourChannelData.size(); i++) {
			unsigned int binIndex = imageColourChannelData[i] / binSize;
			hist[binIndex]++;
		}
		end = high_resolution_clock::now();
		totalDurationMs += duration_cast<milliseconds>(end - start).count();

		// Step two, cumulative sum histogram.
		start = high_resolution_clock::now();
		for (unsigned int i = 1; i < hist.size(); i++) {
			hist[i] += hist[i - 1];
		}
		end = high_resolution_clock::now();
		totalDurationMs += duration_cast<milliseconds>(end - start).count();

		// Step three, convert to normalised lookup table.
		start = high_resolution_clock::now();

		// Get max value (it's just the last one), cast to float so we avoid integer truncation later when dividing.
		const float maxHistValue = static_cast<float>(hist[hist.size() - 1]);
		for (unsigned int i = 0; i < hist.size(); i++) {
			hist[i] = (hist[i] / maxHistValue) * maxPixelValue;
		}
		end = high_resolution_clock::now();
		totalDurationMs += duration_cast<milliseconds>(end - start).count();

		// Step four, backproject.
		start = high_resolution_clock::now();
		for (unsigned int i = 0; i < imageColourChannelData.size(); i++) {
			const unsigned int binIndex = imageColourChannelData[i] / binSize;
			outputImageData[i + (imageSize * colourChannel)] = hist[binIndex];
		}
		end = high_resolution_clock::now();
		totalDurationMs += duration_cast<milliseconds>(end - start).count();
	}

	cout << endl << "Total Serial Algorithm Duration: " << totalDurationMs << "ms" << endl;

	// Create the image from the output data.
	CImg<unsigned short> outputImageSerial(outputImageData.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());

	return outputImageSerial;
}


int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "E:/Dev/Parallel-Programming-Assessment/Images/test_colour_16_small.ppm";
	unsigned int binSize = 1;
	unsigned short maxPixelValue = 65535;

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

		//CumulativeSumParallel(program, context, queue, device_id);
		
		double totalDurationSerial, totalDurationParallel;

		// Read image from file.
		CImg<unsigned short> inputImage(image_filename.c_str());

		// Display input image.
		CImgDisplay displayInput(inputImage, "input");

		CImg<unsigned short> outputImageSerial = SerialImplementation(inputImage, binSize, totalDurationSerial, maxPixelValue);
		CImg<unsigned short> outputImageParallel = ParallelImplementation(program, context, queue, inputImage, binSize, totalDurationParallel, maxPixelValue);

		cout << endl << "Parallel Implementation is " << static_cast<int>(totalDurationSerial / totalDurationParallel) << " times faster than the serial equivalent on this image." << endl;
		
		CImgDisplay displayOutputSerial(outputImageSerial, "serial_output");
		CImgDisplay displayOutputParallel(outputImageParallel, "parallel_output");

		while (!displayInput.is_closed() && !displayOutputSerial.is_closed() && !displayOutputParallel.is_closed()){
			displayInput.wait(1);
			displayOutputSerial.wait(1);
			displayOutputParallel.wait(1);
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