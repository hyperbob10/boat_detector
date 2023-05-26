#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

#include "boatDetector.h"


using namespace std;
using namespace cv;


int main(int argc , char** argv)
{
	string imgPath = (string)argv[1];//"../venice";
	string type = (string)argv[2];//"png";

	vector<cv::String> filenames;
	cv::utils::fs::glob(cv::String(imgPath), cv::String("*." + type), filenames);

	//create a new boatDetector object
	string pathClassifier = "../cascade.xml";
	boatDetector* classifier = new boatDetector(pathClassifier);

	string groundTruthPath = imgPath + "/ground_truth";
	vector<vector<Rect>> groundTruth = classifier->loadGroundTruth(groundTruthPath);
	char i = 0;

	for (const auto& fn : filenames)
	{
		vector<Rect> gt = groundTruth[i];

		Mat src = imread(fn);
		std::cout << "Loading image..." << endl;

		//load the image to be processed
		bool isEmpty = classifier->loadImage(src);


		//If the image is empty then return
		if (isEmpty)
		{
			cout << "Error: Empty Image" << endl;
			return -1;
		}

		imshow("Starting image", src); 
		waitKey();

		




		vector<Rect> boxes;
		vector<double> scores;
		vector<int> levels;

		//Detect now the boats in the image
		bool found = classifier->detectBoat(boxes , scores , levels);
		vector<Rect> newBoxes; //create an array of new bounding boxes

		vector<double> newScores;
		vector<double> IoUs;

		//If some boats are found then perform the post processing
		if (found)
		{
			classifier->showBoxes(boxes, scores, src, "Before Post Processing");

			classifier->clusterBoxes(boxes, scores, newBoxes , newScores);

			classifier->showBoxes(newBoxes, newScores, src, "After Post Processing");
			
			IoUs = classifier->getAccuracy(newBoxes, gt);			

			classifier->drawGTvsBoxes(newBoxes, gt, IoUs, src);
 		}


		else
			cout << "No boats found" << endl;

		destroyAllWindows();
		i++;		
	}
	return 0;
}





