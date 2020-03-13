#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"
#include <chrono>  // for high_resolution_clock

#include "ParallelHslProcessor.h";
#include "ParallelProcessor.h";
#include "SharedParallel.h";
#include "SerialProcessor.h";

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





int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_colour_16_small.ppm";
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
		AddSources(sources, "kernels/RgbKernels.cl");
		AddSources(sources, "kernels/HslKernels.cl");
		AddSources(sources, "kernels/SharedKernels.cl");

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
		
		double totalDurationSerial, totalDurationParallel;

		// Read image from file.
		CImg<unsigned short> inputImage(image_filename.c_str());

		// Display input image.
		CImgDisplay displayInput(inputImage, "input");

		// Get the size of a single channel of the image. i.e. the actual number of pixels.
		const unsigned int imageSize = inputImage.height() * inputImage.width();

		// Check if it's 8-bit or 16-bit.
		maxPixelValue = inputImage.max();
		if (maxPixelValue > 255) {
			maxPixelValue = 65535;
		}
		else {
			maxPixelValue = 255;
		}

		
		//vector<unsigned short> outputRgbImg = ConvertHslToRgb(program, context, queue, hslImg, totalDurationParallel, imageSize, maxPixelValue);

		CImg<unsigned short> outputImageSerial = SerialImplementation(inputImage, binSize, totalDurationSerial, maxPixelValue);
		//CImg<unsigned short> outputImageParallel = ParallelImplementation(program, context, queue, inputImage, binSize, totalDurationParallel, maxPixelValue, device_id);
		CImg<unsigned short> outputImageParallel = ParallelCorrectColours(program, context, queue, inputImage, binSize, totalDurationParallel, imageSize, maxPixelValue, device_id);

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