#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
 
#include <string>
#include <iostream>
#include <time.h>
 
using namespace std;
using namespace cv;
using namespace dnn;
 
int main()
{
	//Net cvNet = readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v1_coco.pbtxt");
	Net net = readNetFromCaffe("halmet.prototxt", "halmet.caffemodel");
 
	const char* classNames[] = { "background", "w","b","r","y", "head" };
 
	float detect_thresh = 0.24;
 
	VideoCapture cap(-1);
        cv::Mat image;
	while(1)
	{
		cap >>image;
 
		clock_t start_t = clock();
		net.setInput(blobFromImage(image, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false));
		Mat cvOut = net.forward();
		cout << "Cost time: " << clock() - start_t << endl;
 
		Mat detectionMat(cvOut.size[2], cvOut.size[3], CV_32F, cvOut.ptr<float>());
		for (int i = 0; i < detectionMat.rows; i++)
		{
			int obj_class = detectionMat.at<float>(i, 1);
			float confidence = detectionMat.at<float>(i, 2);
 
			if (confidence > detect_thresh)
			{
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
 
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
 
				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
 
				rectangle(image, object, Scalar(0, 0, 255), 2);
				putText(image, classNames[obj_class], Point(xLeftBottom, yLeftBottom - 10), 3, 0.5, Scalar(0, 0, 255), 2);
			}
		}
 
		imshow("test", image);
		cv::waitKey(1);
	}
	
	return 0;
}
