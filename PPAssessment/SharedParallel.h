#pragma once

class SharedParallel {
public:
	static vector<unsigned int> CumulativeSumParallel(const cl::Program& program, const cl::Context& context, const cl::CommandQueue& queue, const int deviceId, vector<unsigned int> input, double& totalDurationMs) {
		// Save the size and count of the input for use with the output later. We need to do this first before any padding is added to the input.
		const size_t outputCount = input.size();
		const size_t outputSize = outputCount * sizeof(unsigned int);

		cout << "\tTwo-Stage Scan for Large Arrays:" << endl;

		// Create the kernel for the double buffered scan.
		cl::Kernel phase1Kernel = cl::Kernel(program, "scanHillisSteeleBuffered");

		// Get the device so we can extract info about it.
		const cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceId];

		// Get the preferred local size.
		const size_t localSize = phase1Kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
		// Calculate how many bytes this is.
		const size_t localSizeBytes = localSize * sizeof(unsigned int);

		// Work out if we need any padding.
		const size_t paddingSize = input.size() % localSize;

		// Do we need padding?
		if (paddingSize > 0) {
			// Yes, add the appropriate number of zeros as padding. Zeros are used because they don't affect the sum.
			const vector<unsigned int> paddingVector(localSize - paddingSize, 0);
			input.insert(input.end(), paddingVector.begin(), paddingVector.end());
		}

		// How many elements do we have now?
		const size_t inputCount = input.size();

		// What's the memory footprint of the input?
		const size_t inputSize = input.size() * sizeof(unsigned int);

		// How many work groups need to run?
		const size_t numberOfGroups = inputCount / localSize;

		// What's the memory footprint of this many work groups?
		const size_t numberOfGroupsBytes = numberOfGroups * sizeof(unsigned int);

		// Create a buffer to store the input data.
		const cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);

		// Create a buffer to store the output data.
		const cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);

		// Write the input data to the device.
		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &input[0]);
		// Fill the output buffer with zeros on the device.
		queue.enqueueFillBuffer(outputBuffer, 0, 0, outputSize);

		// Set arguments for the scan kernel.
		phase1Kernel.setArg(0, inputBuffer);
		phase1Kernel.setArg(1, outputBuffer);
		phase1Kernel.setArg(2, cl::Local(localSizeBytes));
		phase1Kernel.setArg(3, cl::Local(localSizeBytes));

		// Create  an event for performance tracking.
		cl::Event perfEventPhase1;

		// Run the double buffered scan kernel. Keep the result on device for now so we can re-use the buffer.
		queue.enqueueNDRangeKernel(phase1Kernel, cl::NullRange, cl::NDRange(inputCount), cl::NDRange(localSize), NULL, &perfEventPhase1);

		// Create the block sum kernel for phase 2.
		cl::Kernel phase2Kernel = cl::Kernel(program, "blockSum");

		// Create a separate output buffer that will store the block sum result.
		// This buffer needs to be the same size as the number of groups.
		const cl::Buffer phase2OutputBuffer(context, CL_MEM_READ_WRITE, numberOfGroupsBytes);

		// Set the arguments for the block sum kernel.
		phase2Kernel.setArg(0, outputBuffer);
		phase2Kernel.setArg(1, phase2OutputBuffer);
		// The kernel needs to know the local size of the previous scan kernel - cast it to an int because size_t is not compatible with the kernel code.
		phase2Kernel.setArg(2, static_cast<int>(localSize));

		// Create  an event for performance tracking.
		cl::Event perfEventPhase2;

		// Run the block sum kernel.
		queue.enqueueNDRangeKernel(phase2Kernel, cl::NullRange, cl::NDRange(numberOfGroups), cl::NullRange, NULL, &perfEventPhase2);

		// Create the kernel for the scan on the block sums.
		// We use standard Hillis-Steele here because it is approximately twice as fast as a serial scan with atomics (5us vs 10us).
		// Although Hillis-Steele is an inclusive scan we can immitate an exclusive scan by skipping the first and last items later when we run the scanAddAdjust kernel.
		cl::Kernel phase2ScanKernel = cl::Kernel(program, "scanHillisSteele");

		// Set the arguments for the block scan.
		phase2ScanKernel.setArg(0, phase2OutputBuffer);

		// Create  an event for performance tracking.
		cl::Event perfEventPhase2Scan;

		// Run the block scan kernel.
		queue.enqueueNDRangeKernel(phase2ScanKernel, cl::NullRange, cl::NDRange(numberOfGroups), cl::NullRange, NULL, &perfEventPhase2Scan);

		// Create a kernel for adjusting the original partial scan with the new block scan result.
		cl::Kernel phase3Kernel = cl::Kernel(program, "scanAddAdjust");

		// Input the original partial scan result.
		phase3Kernel.setArg(0, outputBuffer);
		// Input the block scan result.
		phase3Kernel.setArg(1, phase2OutputBuffer);

		// Create  an event for performance tracking.
		cl::Event perfEventPhase3;

		size_t phase3GlobalSize = inputCount - localSize;

		// Correction for when number of bins is less than local size.
		if (phase3GlobalSize <= 0) {
			phase3GlobalSize = inputCount;
		}

		// Run the scan add kernel.
		queue.enqueueNDRangeKernel(phase3Kernel, cl::NDRange(localSize), cl::NDRange(phase3GlobalSize), cl::NDRange(localSize), NULL, &perfEventPhase3);

		// Create an output vector of the right size.
		vector<unsigned int> outputData(outputCount);

		// Copy the contents of the output buffer on the device to the vector on the host.
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &outputData[0]);

		// Print out the performance values.
		cout << "\t\tDouble Buffered Hillis-Steele Scan: " << GetFullProfilingInfo(perfEventPhase1, ProfilingResolution::PROF_US) << endl;
		cout << "\t\tBlock Sum: " << GetFullProfilingInfo(perfEventPhase2, ProfilingResolution::PROF_US) << endl;
		cout << "\t\tBlock Scan: " << GetFullProfilingInfo(perfEventPhase2Scan, ProfilingResolution::PROF_US) << endl;
		cout << "\t\tScan Add: " << GetFullProfilingInfo(perfEventPhase3, ProfilingResolution::PROF_US) << endl;
		totalDurationMs += GetProfilingTotalTimeMs(perfEventPhase1);
		totalDurationMs += GetProfilingTotalTimeMs(perfEventPhase2);
		totalDurationMs += GetProfilingTotalTimeMs(perfEventPhase2Scan);
		totalDurationMs += GetProfilingTotalTimeMs(perfEventPhase3);

		return outputData;
	}
};
