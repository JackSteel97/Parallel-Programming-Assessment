#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
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

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());

		vector<int> hist(256);

		CImgDisplay disp_input(image_input, "input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error & err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		size_t sizeOfHistogram = hist.size() * sizeof(int);

		//Part 4 - device operations
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer histogramInputBuffer(context, CL_MEM_READ_WRITE, sizeOfHistogram);
		//cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel histogramKernel = cl::Kernel(program, "histogramAtomic");
		histogramKernel.setArg(0, dev_image_input);
		histogramKernel.setArg(1, histogramInputBuffer);

		cl::Event prof_event;
		queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(histogramInputBuffer, CL_TRUE, 0, sizeOfHistogram, &hist.data()[0]);


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

		queue.enqueueNDRangeKernel(cumulitiveSumKernel, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event);

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

		queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event);

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

		queue.enqueueNDRangeKernel(backPropKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(backPropOutputImage, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		std::cout << "Kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() <<
			std::endl;

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