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

vector<int> BuildImageHistogram(const cl::Program &program, const cl::Context &context, const cl::CommandQueue queue, const unsigned int &binSize, const CImg<unsigned char> &image) {
	const int numberOfBins = ceil(255 / binSize);

	// Initialise a vector for the histogram with the appropriate bin size. Add one because this is capacity not maximum index.
	vector<int> hist(numberOfBins+1);

	size_t sizeOfHistogram = hist.size() * sizeof(int);

	// Create buffers for the device.
	cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, image.size());
	cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

	// Copy image data to image buffer on the device and wait for it to finish before continuing.
	queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, image.size(), &image.data()[0]);

	// Create the kernel to use.
	cl::Kernel histogramKernel = cl::Kernel(program, "histogramAtomic");
	// Set kernel arguments.
	histogramKernel.setArg(0, inputImageBuffer);
	histogramKernel.setArg(1, histogramBuffer);
	histogramKernel.setArg(2, binSize);

	// Create  an event for performance tracking.
	cl::Event perfEvent;
	// Queue the kernel for execution on the device.
	queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image.size()), cl::NullRange, NULL, &perfEvent);

	// Copy the result from the device to the host.
	queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

	// Print out the performance values.
	cout << GetFullProfilingInfo(perfEvent, ProfilingResolution::PROF_US) << endl;

	return hist;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "E:/Dev/ParallelAssessment/Images/test.ppm";

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
		cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

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
		CImg<unsigned char> image_input(image_filename.c_str());

		// Display input image.
		CImgDisplay disp_input(image_input, "input");

		vector<int> hist = BuildImageHistogram(program, context, queue, 10, image_input);
		size_t sizeOfHistogram = hist.size() * sizeof(int);

		//Part 4 - device operations
		//device - buffers
		cl::Buffer histogramCumulitiveInputBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);
		cl::Buffer histogramOutputBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(histogramCumulitiveInputBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);
		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel cumulitiveSumKernel = cl::Kernel(program, "scan_hs");
		cumulitiveSumKernel.setArg(0, histogramCumulitiveInputBuffer);
		cumulitiveSumKernel.setArg(1, histogramOutputBuffer);

		queue.enqueueNDRangeKernel(cumulitiveSumKernel, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(histogramCumulitiveInputBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);




		//Part 4 - device operations
		//device - buffers
		cl::Buffer histogramLutInputBuffer(context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer histogramLutOutputBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);

		

		int maxHistValue = hist[hist.size() - 1];

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(histogramLutInputBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);
		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel lutKernel = cl::Kernel(program, "lut");
		lutKernel.setArg(0, histogramLutInputBuffer);
		lutKernel.setArg(1, maxHistValue);
		lutKernel.setArg(2, histogramLutOutputBuffer);

		queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(histogramLutOutputBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

		//Part 4 - device operations
		//device - buffers
		cl::Buffer backPropInputImage(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer backPropInputHist(context, CL_MEM_READ_ONLY, sizeOfHistogram);
		cl::Buffer backPropOutputImage(context, CL_MEM_READ_WRITE, image_input.size());

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(backPropInputImage, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(backPropInputHist, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel backPropKernel = cl::Kernel(program, "backprojection");
		backPropKernel.setArg(0, backPropInputImage);
		backPropKernel.setArg(1, backPropInputHist);
		backPropKernel.setArg(2, backPropOutputImage);

		queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(backPropOutputImage, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
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