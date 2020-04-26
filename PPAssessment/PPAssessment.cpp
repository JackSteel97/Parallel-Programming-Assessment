#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"
#include <chrono>  // for high_resolution_clock.

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

void clearInput() {
	cin.clear();
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
			clearInput();
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

unsigned int printBinSizeMenu(const unsigned int& maxBinSize) {
	unsigned int selection = 0;
	// Go until we get a valid selection.
	do {
		cout << endl << "Enter a bin size to use (1-" << maxBinSize << "): ";
		cin >> selection;
		if (cin.fail() || selection < 1 || selection > maxBinSize) {
			cout << endl << "Invalid entry, please enter an available number." << endl;
			clearInput();
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

	// Read image from file.
	CImg<unsigned short> inputImage(imageFile.c_str());

	return inputImage;
}

// Wait for the CImgDisplays to be closed.
void waitForImageClosure(CImgDisplay& input, CImgDisplay& output) {
	while (!input.is_closed() && !output.is_closed()) {
		input.wait(1);
		output.wait(1);
	}
}

/*
Report:

<summary>
	This solution has been developed to work on an AMD Ryzen based system. As such to run on your machine,
	you will likely need to change the Additional Include Directories and the Additional Library Directories
	in the C/C++ and Linker project property pages, to the specific locations on your machine.

	Command-line arguments are not required to run this solution – all option choices are handled at runtime
	by simple text menus. The first menu allows selection of an image on which to operate. The system supports
	the two provided greyscale images, in addition to 8-bit and 16-bit colour images provided. You can also
	specify an absolute file path to a custom image to load. The second menu provides a choice of an algorithm;
	the system can perform histogram equalisation in serial, parallel, and parallel
	with Hue Saturation Luminance (HSL) conversion – preserving colours better.
	A comparison between serial and parallel performance can also be automatically executed. Next, you can
	specify a custom bin size. Detailed performance metrics are reported for each algorithm.

	The parallel algorithm uses a double-buffered Hillis-Steele cumulative sum to ensure optimised performance
	for any number of bins. This cumulative sum utilises on-device buffers to reduce copy operations between host and device.
	The code is split into classes that handle each algorithm’s implementation.
	The colour preservation option converts the image to from RGB format to HSL for processing and then back to
	RGB for display resulting in more accurate colours in the output (Waldman, 2013).
	These conversions are also done in parallel using the map pattern.

	The entirety of the code is heavily optimised, utilising constants, and specific data types to reduce memory
	footprint and runtime. Notable original additions include variable bin size, colour image support,
	16-bit image support, HSL implementation, and performance comparison.
</summary>

<references>
	Waldman, N. (2013) Math behind colorspace conversions, RGB-HSL – Niwa Available from http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/ [accessed 10 April 2020].
</references>
*/
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

		while (true) {


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

			if (selection == 3) {
				// Hsl processing - 100% is max HSL value.
				binSize = printBinSizeMenu(100);
			}
			else {
				binSize = printBinSizeMenu(maxPixelValue+1);
			}
			
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
			}

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
			clearInput();
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