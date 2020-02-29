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
	string image_filename = "test.ppm";

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


		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		std::cout << "Read input" << std::endl;
		std::vector<unsigned char> outputImage(image_input.size());
		std::vector<int> Hist(256);

		try {
			std::cout << "Created output" << std::endl;
			for (int i = 0; i < image_input.size(); i++) {
				int y = i / image_input.width();
				int x = i % image_input.width();
				Hist[image_input(x, y)]++;
			}
			std::cout << "Done hist summing" << std::endl;
			for (int i = 1; i < Hist.size(); i++) {
				Hist[i] += Hist[i - 1];
			}
			std::cout << "Done hist cumulation" << std::endl;
			int maxHist = Hist[Hist.size() - 1];
			std::cout << "Done max hist cumulation" << std::endl;

			for (int i = 0; i < Hist.size(); i++) {
				Hist[i] = Hist[i] * 255 / maxHist;
			}
			std::cout << "Done LUT" << std::endl;
			for (int i = 0; i < image_input.size(); i++) {
				int y = i / image_input.width();
				int x = i % image_input.width();
				outputImage[i] = Hist[image_input(x, y)];
			}
			std::cout << "Done back-projection" << std::endl;
		}
		catch (const std::exception & e) {
			std::cout << "Exception: " << e.what() << std::endl;
		}

		CImg<unsigned char> output_image_serial(outputImage.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image_serial, "output");
		CImgDisplay disp_input(image_input, "input");
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

		return 0;

		////Part 3 - host operations
		////3.1 Select computing devices
		//cl::Context context = GetContext(platform_id, device_id);

		////display the selected device
		//std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		////create a queue to which we will push commands for the device
		//cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		////3.2 Load & build the device code
		//cl::Program::Sources sources;

		//AddSources(sources, "kernels/my_kernels.cl");

		//cl::Program program(context, sources);

		////build and debug the kernel code
		//try {
		//	program.build();
		//}
		//catch (const cl::Error & err) {
		//	std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		//	std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		//	std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		//	throw err;
		//}

		////Part 4 - device operations

		////device - buffers
		//cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		//cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		////4.1 Copy images to device memory
		//queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		////4.2 Setup and execute the kernel (i.e. device code)
		//cl::Kernel kernel = cl::Kernel(program, "identity");
		//kernel.setArg(0, dev_image_input);
		//kernel.setArg(1, dev_image_output);

		//cl::Event prof_event;
		//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);

		//vector<unsigned char> output_buffer(image_input.size());
		////4.3 Copy the result from device to host
		//queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		//CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		////CImgDisplay disp_output(output_image, "output");

		//std::cout << "Kernel execution time [ns]:" <<
		//	prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() <<
		//	std::endl;

		//while (!disp_input.is_closed() && !disp_output.is_closed()
		//	&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		//	disp_input.wait(1);
		//	disp_output.wait(1);
		//}
	}
	catch (const cl::Error & err) {
		std::cout << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException & err) {
		std::cout << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}