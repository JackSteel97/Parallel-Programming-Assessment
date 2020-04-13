#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"
#include <chrono>  // for high_resolution_clock

using namespace cimg_library;
using namespace std;
using namespace chrono;

#include "SharedParallel.h";
#include "ParallelHslProcessor.h";
#include "ParallelProcessor.h";
#include "SerialProcessor.h";

void print_help() {
	cout << "Application usage:" << endl;

	cout << "  -p : select platform " << endl;
	cout << "  -d : select device" << endl;
	cout << "  -l : list all platforms and devices" << endl;
	cout << "  -h : print this message" << endl;
}

int printMenu() {
	cout << endl << "Main Menu" << endl;

	cout << "[1] Run Histogram Equalisation in Serial." << endl;
	cout << "[2] Run Histogram Equalisation in Parallel." << endl;
	cout << "[3] Run Histogram Equalisation in Parallel with Colour Preservation." << endl;
	cout << "[4] Run Comparison Between Serial and Parallel Performance." << endl;

	int selection = 0;
	// Go until we get a valid selection.
	do {
		cout << "Select a numbered option: ";
		cin >> selection;
		if (cin.fail()) {
			cout << endl << "Invalid entry, please enter an available number." << endl;
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			selection = -1;
		}
	} while (selection < 0);

	return selection;
}

CImg<unsigned short> getCustomImage() {
	cout << "Enter the absolute file path to the custom image: ";
	string customFilePath;
	cin >> customFilePath;

	// Read image from file.
	CImg<unsigned short> inputImage(customFilePath.c_str());

	return inputImage;
}

unsigned int printBinSizeMenu(const unsigned short &maxPixelValue) {
	unsigned int selection = 0;
	// Go until we get a valid selection.
	do {
		cout << endl << "Enter a bin size to use (1-" << maxPixelValue << "): ";
		cin >> selection;
		if (cin.fail() || selection < 1 || selection > maxPixelValue) {
			cout << endl << "Invalid entry, please enter an available number." << endl;
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			selection = 0;
		}
	} while (selection < 1);

	return selection;
}

CImg<unsigned short> printImageLoadMenu() {
	cout << endl << "Image Loader" << endl;

	cout << "[1] Small Greyscale (test.ppm)." << endl;
	cout << "[2] Large Greyscale (test_large.ppm)." << endl;
	cout << "[3] 8-Bit Colour (test_colour.ppm)." << endl;
	cout << "[4] 16-Bit Colour (test_colour_16.ppm)." << endl;
	cout << "[5] Custom Image." << endl;

	int selection = 0;
	// Go until we get a valid selection.
	do {
		cout << "Select a numbered option: ";
		cin >> selection;
		if (cin.fail()) {
			cout << endl << "Invalid entry, please enter an available number." << endl;
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			selection = -1;
		}
	} while (selection < 0);

	string imageFilePathPrefix = "E:/Dev/Parallel-Programming-Assessment/Images/";

	string imageFile = "test.ppm";
	switch (selection) {
	case 1:
		imageFile = "test.ppm";
		break;
	case 2:
		imageFile = "test_large.ppm";
		break;
	case 3:
		imageFile = "test_colour.ppm";
		break;
	case 4:
		imageFile = "test_colour_16.ppm";
		break;
	case 5:
		return getCustomImage();
	default:
		cout << "Invalid Menu Selection." << endl;
		return printImageLoadMenu();
	}

	string imageFileName = imageFilePathPrefix + imageFile;

	// Read image from file.
	CImg<unsigned short> inputImage(imageFileName.c_str());

	return inputImage;
}

void waitForImageClosure(CImgDisplay& input, CImgDisplay& output) {
	while (!input.is_closed() && !output.is_closed()) {
		input.wait(1);
		output.wait(1);
	}
}


void AccumulateHistogramHillisSteele(const cl::Program& program, const cl::Context& context, const cl::CommandQueue queue, const size_t& sizeOfHistogram, vector<unsigned int>& histogram, double& totalDurationMs) {

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





int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platformId = 0;
	int deviceId = 0;
	unsigned int binSize = 1;
	unsigned short maxPixelValue = 65535;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformId = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceId = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		//else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	try {

		// Get OpenCL context for the selected platform and device.
		cl::Context context = GetContext(platformId, deviceId);

		// Display the selected device.
		cout << "Running on " << GetPlatformName(platformId) << ", " << GetDeviceName(platformId, deviceId) << endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code.
		cl::Program::Sources sources;

		// Load the kernels source.
		AddSources(sources, "RgbKernels.cl");
		AddSources(sources, "HslKernels.cl");
		AddSources(sources, "SharedKernels.cl");

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

		// Get an image loaded by the user's choice.
		CImg<unsigned short> inputImage = printImageLoadMenu();

		// Get the size of a single channel of the image. i.e. the actual number of pixels.
		unsigned int imageSize = inputImage.height() * inputImage.width();

		// Check if it's 8-bit or 16-bit.
		maxPixelValue = inputImage.max();
		if (maxPixelValue > 255) {
			maxPixelValue = 65535;
		}
		else {
			maxPixelValue = 255;
		}

		int selection = printMenu();

		binSize = printBinSizeMenu(maxPixelValue);

		double totalDuration = 0;
		CImg<unsigned short> outputImage;
		switch (selection) {
		case 1: {
			SerialProcessor serialProc(inputImage, binSize, totalDuration, maxPixelValue, imageSize);
			outputImage = serialProc.RunHistogramEqualisation();
			break;
		}
		case 2: {
			ParallelProcessor parallelProc(program, context, queue, inputImage, binSize, totalDuration, imageSize, maxPixelValue, deviceId);
			outputImage = parallelProc.RunHistogramEqualisation();
			break;
		}
		case 3: {
			ParallelHslProcessor parallelHslProc(program, context, queue, inputImage, binSize, totalDuration, imageSize, maxPixelValue, deviceId);
			outputImage = parallelHslProc.RunHistogramEqalisation();
			break;
		}
		case 4: {
			double totalParallelDuration = 0;
			SerialProcessor serialProc(inputImage, binSize, totalDuration, maxPixelValue, imageSize);
			CImg<unsigned short> serialOutput = serialProc.RunHistogramEqualisation();

			ParallelProcessor parallelProc(program, context, queue, inputImage, binSize, totalParallelDuration, imageSize, maxPixelValue, deviceId);
			outputImage = parallelProc.RunHistogramEqualisation();

			cout << endl << "------------------------------------------------------------------------------------------------------" << endl;
			cout << "\tSerial duration: " << totalDuration << "ms" << endl;
			cout << "\tParallel duration: " << totalParallelDuration << "ms" << endl;
			cout << "\tThe parallel implementation is " << static_cast<int>(totalDuration / totalParallelDuration) << " times faster than the serial equivalent on this image." << endl;
			cout << "------------------------------------------------------------------------------------------------------" << endl;
			break;
		}
		default:
			cout << "Invalid menu selection." << endl;
			selection = printMenu();
		};


		if (maxPixelValue == 255) {
			// 8-Bit image, convert the CImgs to use chars.
			CImg<unsigned char> input8Bit = inputImage;
			CImg<unsigned char> output8Bit = outputImage;

			// Display input image.
			CImgDisplay displayInput(input8Bit, "input");
			CImgDisplay displayOutput(output8Bit, "output");
			waitForImageClosure(displayInput, displayOutput);
		}
		else {
			// Just display the 16 bit ones.
			CImgDisplay displayInput(inputImage, "input");
			CImgDisplay displayOutput(outputImage, "output");
			waitForImageClosure(displayInput, displayOutput);
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