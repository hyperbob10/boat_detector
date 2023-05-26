#include "boatDetector.h"


	boatDetector::boatDetector(string path)
	{
		classifier.load(path);
	}



	//Load image and return true if it's empty. Otherwise convert it in Gray scale
	bool boatDetector::loadImage(Mat& src)
	{
		if (src.empty())
			return true;

		// copy the loaded image
		colorImage = src.clone(); 
		//convert it in gray scale
		cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);

		return false;
	}





	//detect boats in the loaded image and return true if at least one boat is found
	bool boatDetector::detectBoat(vector<Rect>& boxes, vector<double>& scores , vector<int>& levels)
	{
		float scaleFactor = 1.1;
		int minNeighbours = 6;
		int minSize = 24;

		classifier.detectMultiScale(grayImage, boxes, levels, scores, scaleFactor, minNeighbours, 0, Size(minSize , minSize), Size(), true);
		
		vector<double> newScores;

		//transform the scores in probabilities by using the sigmoid function
		for (const auto& sc : scores)
			newScores.push_back(1 / (1 + exp(-sc)));

		scores = newScores;


		return boxes.size() > 0;
	}



	//clusters the found boxes and return the number of clusters
	int boatDetector::clusterBoxes(vector<Rect>& boxes , vector<double>& scores , vector<Rect>& newBoxes , vector<double>& newScores)
	{
		//if there aren't detected boxes return doing nothing
		if (boxes.size() < 1) return 0;

		vector<int> labels;
		struct predicate
		{
			bool operator()(const Rect& lhs, const Rect& rhs)
			{
				//compute the coordinates of the centers of the boxes
				double x_lhs = lhs.x + lhs.width / 2;
				double y_lhs = lhs.y + lhs.height / 2;

				double x_rhs = rhs.x + rhs.width / 2;
				double y_rhs = rhs.y + rhs.height / 2;

				Point cen_rhs = Point(x_rhs, y_rhs);
				Point cen_lhs = Point(x_lhs, y_lhs);

				//return true if the boxes overlap or if they are distant within a certain threshold
				return ((lhs&rhs).area() > 0 || cv::norm(Mat(cen_rhs), Mat(cen_lhs)) < 80);
			}
		};	

		//label the objects belonging to the same cluster
		partition(boxes, labels, predicate());


		//cluster the boxes
		int max = 0;
		for (const auto& lb : labels)
			if (lb > max) 
				max = lb;


		vector < vector<Rect> > groups(max + 1);
		vector < vector<double> > groupScores(max + 1);
		for (int j = 0; j < labels.size(); j++)
		{
			groups[labels[j]].push_back(boxes[j]); 
			groupScores[labels[j]].push_back(scores[j]);
		}


		//merge now the boxes belonging to the same cluster
		//the scores of the resulting boxes are the averages of the groups scores
		newBoxes.resize(groups.size());
		newScores.resize(groupScores.size());
		for (int i = 0; i < groups.size(); i++)
		{
			if (!groups.empty())
			{
				newBoxes[i] = groups[i][0];
				newScores[i] = 0;
				for(int j = 0; j < groups[i].size(); j++)
				{
					newBoxes[i] |= groups[i][j];
					newScores[i] = newScores[i] + (groupScores[i][j]);
				}

				newScores[i] = newScores[i] / groups[i].size();
			}
		}

		//prune the resulting boxes which don't contain any boat
		pruning(newBoxes, newScores);

		return newBoxes.size();
	}






	//draw the boxes around the boats and show the resulting image
	void boatDetector::showBoxes(vector<Rect>& boxes, vector<double>& scores , Mat& img , String title)
	{
		Mat image = img.clone();

		for (int i = 0; i < boxes.size(); i++)
			rectangle(image, boxes[i], Scalar(0, 0, 255), 2, 8); ///Draws rectangle around the plate in the image
		

		imshow(title, image);
		waitKey();
	}




	//get a vector containing the IoU scores for every detected box
	vector<double> boatDetector::getAccuracy(vector<Rect>& boxes, vector<Rect>& groundTruth)
	{
		vector<double> IoUs;

		for (auto& bb : boxes)
		{
			double iou = 0; // start from IoU equal to zero so that if no intersection you have 0

			for (auto& gt : groundTruth)
				if (overlap(bb , gt ))
				{
					double i = static_cast<double>((bb & gt).area());
					double u = static_cast<double>((bb | gt).area());
					
					if(i/u > 0 && areComparable(bb , gt , 0.15))
						iou = i/u;
				}
			IoUs.push_back(iou);
		}
		return IoUs;
	}






	bool boatDetector::overlap(Rect& lhs, Rect& rhs)
	{
		return (lhs & rhs).area() > 0;
	}


	//return true if the ratio between the areas of the rectangle is below a certain threshold
	bool boatDetector::areComparable(Rect& lhs, Rect& rhs , double threshold)
	{
		double ratio = (double)(rhs.area()) / (double)(lhs.area());

		if (ratio > 1)
			ratio = 1 / ratio;

		return ratio > threshold;
	}

	//get the colored image
	Mat boatDetector::getColorImage()
	{
		return colorImage;
	}

	//get the gray image
	Mat boatDetector::getGrayImage()
	{
		return grayImage;
	}

	CascadeClassifier boatDetector::getClassifier()
	{
		return classifier;
	}



	//load the ground truth boxes for every image. path should contain the directory where the ground truth files are stored
	vector<vector<Rect>> boatDetector::loadGroundTruth(cv::String path) 
	{
		cv::Point tl, br;	//rectangles top left and bottom right corners
		std::vector<cv::String> filenames;
		int index, file_index = 0;
		cv::utils::fs::glob(path , cv::String("*.txt"), filenames);
		std::vector<std::vector<cv::Rect>> file_array(filenames.size());
		std::cout << "Number of images:" << file_array.size() << std::endl;

		for (const auto& fn : filenames)
		{
			std::fstream newfile;
			newfile.open(fn, std::ios::in);
			if (newfile.is_open()) 
			{
				std::cout << "Opened: " << fn << std::endl;
				std::string tp, temp;
				while (getline(newfile, tp))  //read data from file object and put it into string.
				{
					index = tp.find(':');
					temp = tp.substr(index + 1, tp.size() - 1);
					// top letf corner X
					tp = temp;
					index = tp.find(';');
					temp = tp.substr(0, index);
					tl.x = std::stoi(temp);
					// bottom right corner X
					tp = tp.substr(index + 1, tp.size() - 1);
					index = tp.find(';');
					temp = tp.substr(0, index);  
					br.x = std::stoi(temp);
					// top letf corner Y
					tp = tp.substr(index + 1, tp.size() - 1);
					index = tp.find(';');
					temp = tp.substr(0, index);  
					tl.y = std::stoi(temp);
					// bottom right corner Y
					tp = tp.substr(index + 1, tp.size() - 1);
					index = tp.find(';');
					temp = tp.substr(0, index);     
					br.y = std::stoi(temp);
					cv::Rect boat_rect(tl, br);
					file_array[file_index].push_back(boat_rect);
				}
			}
			newfile.close();
			file_index++;
		}
		return file_array;
	}





	void boatDetector::drawGTvsBoxes(vector<Rect>& boxes, vector<Rect>& groundTruth, vector<double>& IoU , Mat& src)
	{
		Mat img = src.clone();

		for (const auto& gt : groundTruth)
			rectangle(img, gt, Scalar(0, 255, 0), 2);

		for (int i = 0; i < boxes.size(); i++)
			{
				
				string text = "IoU: " + std::to_string(IoU[i] * 100) + " %";
				
				//draw bounding box
				cv::rectangle(img, boxes[i], Scalar(0, 0, 255), 2);

				//write the IoU inside a black rectangle
				int y_st = boxes[i].y + 15;
				cv::rectangle(img, Point(boxes[i].x - 2, y_st ), Point(boxes[i].x, y_st) + Point(150, -13), CV_RGB(0, 0, 0), FILLED);
				putText(img, text, Point(boxes[i].x, y_st - 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.8, FILLED);
			}


		imshow("Image with Ground Truth, Found boxes and IoU scores", img);
		waitKey();
	}




	void boatDetector::pruning(vector<Rect>& boxes , vector<double>& scores)
	{
		Mat img = getGrayImage();
		
		//Perform an average smoothing to delete some details
		blur(img, img, Size(5, 5));

		//Detect the edges by using the Canny algorithm
		Mat edges;
		Canny(img, edges, 200, 100);

		//Sharpen the image
		img = img - edges;

		//If inside a boxes there aren't boats, then erase it
		for (int i = 0; i < boxes.size(); i++)
		{
			vector<Rect> subBoxes;
			vector<double> weights;
			vector<int> levels;
			Mat cropped = img(boxes[i]);
			
			classifier.detectMultiScale(cropped, subBoxes, 1.1, 3, 0 | CASCADE_SCALE_IMAGE);
			if (subBoxes.size() < 1)
			{
				boxes.erase(boxes.begin() + i);
				scores.erase(scores.begin() + i);
			}
				
		}
	}