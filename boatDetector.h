#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;

class boatDetector
{
public:

	boatDetector(string path);

	//Load image and return src.empty()
	bool loadImage(Mat& src);

	//detect boats in the loaded image and return true if at least one boat is found
	bool detectBoat(vector<Rect>& boxes, vector<double>& scores, vector<int>& levels);

	//clusters the found boxes and return the number of clusters
	int clusterBoxes(vector<Rect>& boxes, vector<double>& scores, vector<Rect>& newBoxes, vector<double>& newScores);

	//draw the boxes around the boats and show the resulting image
	void showBoxes(vector<Rect>& boxes, vector<double>& scores,Mat& img, string title);

	//get the IoU score
	vector<double> getAccuracy(vector<Rect>& boxes, vector<Rect>& groundTruth);

	//return the colored image
	Mat getColorImage();

	//return the gray image
	Mat getGrayImage();

	//return the classifier
	CascadeClassifier getClassifier();

	//load the ground truth boxes for every image. path should contain the directory where the ground truth files are stored
	vector<vector<Rect>> loadGroundTruth(cv::String path);

	//draw the detected boats and the ground truth ones
	void drawGTvsBoxes(vector<Rect>& boxes, vector<Rect>& groundTruth, vector<double>& IoU , Mat& src);

	//delete bounding boxes which don't contain boats
	void pruning(vector<Rect>& boxes , vector<double>& scores);

	//return true if two rectangles overlap. False otherwise
	bool overlap(Rect& lhs, Rect& rhs);

	//return true if the ratio between the areas of the rectangle is below a certain threshold
	bool areComparable(Rect& lhs, Rect& rhs, double threshold);

protected:

	CascadeClassifier classifier;
	Mat colorImage;
	Mat grayImage;

};

