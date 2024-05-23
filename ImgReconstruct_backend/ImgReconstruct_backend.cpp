#include <iostream>
#include "sparseImageEncryption.h"

int main()
{

	cv::Mat img = cv::imread("C:/images/256.jpg", cv::IMREAD_COLOR);
	uint32_t TILE_SIZE = 128;

	//int N = 4;
	//int M = 4;

	if (img.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	encryptionImage img_encrypted = encryptImage(img, TILE_SIZE, "5v48v5832v5924");
	cv::Mat out = decryptImage(img_encrypted, "5v48v5832v5024");
	imshow("Display window", out);
	int y = cv::waitKey(0);

	return 1;
}