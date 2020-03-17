#pragma once

class SerialProcessor {
private:
	CImg<unsigned short>& InputImage;
	unsigned int& BinSize;
	double& TotalDurationMs;
	unsigned short& MaxPixelValue;
	unsigned int ImageSize;

	vector<unsigned int> BuildHistogram(const vector<unsigned short>& imageColourChannelData) {
		const unsigned int numberOfBins = ceil(MaxPixelValue / BinSize) + 1;

		vector<unsigned int> hist(numberOfBins);

		for (unsigned int i = 0; i < imageColourChannelData.size(); i++) {
			unsigned int binIndex = imageColourChannelData[i] / BinSize;
			hist[binIndex]++;
		}

		return hist;
	}

	void CumulativeSumHistogram(vector<unsigned int>& histogram) {
		for (unsigned int i = 1; i < histogram.size(); i++) {
			histogram[i] += histogram[i - 1];
		}
	}

	void NormaliseToLut(vector<unsigned int>& histogram) {
		// Get max value (it's just the last one), cast to float so we avoid integer truncation later when dividing.
		const float maxHistValue = static_cast<float>(histogram[histogram.size() - 1]);

		for (unsigned int i = 0; i < histogram.size(); i++) {
			histogram[i] = (histogram[i] / maxHistValue) * MaxPixelValue;
		}
	}

	void BackProject(const vector<unsigned short>& imageColourChannelData, vector<unsigned short>& outputImageData, const unsigned char& colourChannel, const vector<unsigned int>& hist) {
		for (unsigned int i = 0; i < imageColourChannelData.size(); i++) {
			const unsigned int binIndex = imageColourChannelData[i] / BinSize;
			outputImageData[i + (ImageSize * colourChannel)] = hist[binIndex];
		}
	}
public:
	SerialProcessor(CImg<unsigned short>& inputImage, unsigned int& binSize, double& totalDurationMs, unsigned short& maxPixelValue, unsigned int& imageSize) :
		InputImage(inputImage),
		BinSize(binSize),
		TotalDurationMs(totalDurationMs),
		MaxPixelValue(maxPixelValue),
		ImageSize(imageSize) {}


	CImg<unsigned short> RunHistogramEqualisation() {
		cout << endl << "Running serial Histogram Equalisation..." << endl;

		// Allocate a vector to store the output pixels.
		vector<unsigned short> outputImageData(InputImage.size());

		time_point<high_resolution_clock> start, end;
		double currentDuration = 0;
		for (unsigned char colourChannel = 0; colourChannel < InputImage.spectrum(); colourChannel++) {
			cout << "Running on colour channel " << static_cast<int>(colourChannel) << ":" << endl;

			// Get this colour channel data out of the image.
			CImg<unsigned short>::const_iterator first = InputImage.begin() + (ImageSize * colourChannel);
			CImg<unsigned short>::const_iterator last = InputImage.begin() + (ImageSize * colourChannel) + ImageSize;
			vector<unsigned short> imageColourChannelData(first, last);

			// Step one, build histogram.
			start = high_resolution_clock::now();
			vector<unsigned int> hist = BuildHistogram(imageColourChannelData);
			end = high_resolution_clock::now();
			currentDuration = duration_cast<milliseconds>(end - start).count();
			TotalDurationMs += currentDuration;
			cout << "\tBuild histogram duration: " << currentDuration << "ms" << endl;

			// Step two, cumulative sum histogram.
			start = high_resolution_clock::now();
			CumulativeSumHistogram(hist);
			end = high_resolution_clock::now();
			currentDuration = duration_cast<milliseconds>(end - start).count();
			TotalDurationMs += currentDuration;
			cout << "\tAccumulate histogram duration: " << currentDuration << "ms" << endl;

			// Step three, convert to normalised lookup table.
			start = high_resolution_clock::now();
			NormaliseToLut(hist);
			end = high_resolution_clock::now();
			currentDuration = duration_cast<milliseconds>(end - start).count();
			TotalDurationMs += currentDuration;
			cout << "\tNormalise to Lookup table duration: " << currentDuration << "ms" << endl;

			// Step four, backproject.
			start = high_resolution_clock::now();
			BackProject(imageColourChannelData, outputImageData, colourChannel, hist);
			end = high_resolution_clock::now();
			currentDuration = duration_cast<milliseconds>(end - start).count();
			TotalDurationMs += currentDuration;
			cout << "\tBackprojection duration: " << currentDuration << "ms" << endl;
		}

		cout << endl << "Total Serial Algorithm Duration: " << TotalDurationMs << "ms" << endl;

		// Create the image from the output data.
		CImg<unsigned short> outputImageSerial(outputImageData.data(), InputImage.width(), InputImage.height(), InputImage.depth(), InputImage.spectrum());

		return outputImageSerial;
	}
};