#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include "Car.cpp"

using namespace std;
using namespace cv;

bool isCorectMatch(vector<Point> contour,Rect boundingRect) {
	double ratio = (double)boundingRect.width / (double)boundingRect.height;
	return(boundingRect.area() > 400 &&
		   boundingRect.width  > 30  &&
		   boundingRect.height > 30  &&
		   ratio > 0.2 && ratio < 4.0 &&
		   contourArea(contour) / (double)boundingRect.area() > 0.5
		);
}

vector<Rect>drawCarsContours(Mat& frame, vector<vector<Point>> contours) {
	vector<vector<Point>> hull(contours.size());
	Mat hullsFrame(frame.size(),CV_8UC1,Scalar(0,0,0));
	
	for (int i = 0;i < contours.size();i++) {
		convexHull(contours[i], hull[i]);
		Rect rect = boundingRect(contours[i]);
		if (isCorectMatch(hull[i], rect)) {
			drawContours(hullsFrame, hull, i, Scalar(255, 255, 255), -1);
			
		}
	}

	vector<vector<Point>> goodMatchsContours;
	vector<Rect> goodMatchRects;
	findContours(hullsFrame, goodMatchsContours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0;i < goodMatchsContours.size();i++) {
		Rect rect = boundingRect(goodMatchsContours[i]);
		goodMatchRects.push_back(rect);
		rectangle(frame, rect, Scalar(69, 0, 225));

	}
	//imshow("hull", hullsFrame);
	return goodMatchRects;
}

inline void createCountingLine(Mat& frame) {
	
	line(frame, Point(0, frame.rows / 5*3),Point(frame.cols,frame.rows/5*3),Scalar(0,0,0),4);
}

vector<vector<Point>> contoursOfDetectedMove(Mat& frame, Mat& nextFrame) {
	Mat grayFrame;
	Mat nextGrayFrame;
	Mat diffResult;

	Mat structuringElement = getStructuringElement(MORPH_RECT, Size(5, 5));
	
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
	cvtColor(nextFrame, nextGrayFrame, COLOR_BGR2GRAY);

	GaussianBlur(grayFrame, grayFrame, Size(5, 5), 0, 0);
	GaussianBlur(nextGrayFrame, nextGrayFrame, Size(5, 5), 0, 0);

	absdiff(grayFrame, nextGrayFrame, diffResult);
	threshold(diffResult, diffResult, 30, 255, CV_THRESH_BINARY);
	
	for (int i = 0;i < 2;i++) {
		dilate(diffResult, diffResult, structuringElement);
		dilate(diffResult, diffResult, structuringElement);
		erode(diffResult, diffResult, structuringElement);
	}

//	imshow("diff", diffResult);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(diffResult, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	return contours;
}

inline double distanceBetweenPoints(Point a, Point b) {
	return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}


bool matchToExistingCars(vector<Car> cars,Rect currentCar) {
	cout << cars.size() << endl;
	for (Car car : cars) {
		if (distanceBetweenPoints(car.getPosition(),Point(currentCar.x,currentCar.y)) < 45.0) {
			cout << car.getPosition() << " " << Point(currentCar.x, currentCar.y)<<endl;
			car.updatePosition(Point(currentCar.x, currentCar.y));
			return true;
			}
		}
	return false;
}

inline void showAmountOfCars(Mat& frame, int amountOfCars) {
	putText(frame, to_string(amountOfCars), Point(frame.cols/2-20, frame.rows / 5 * 3 - 20), FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0),3);
}

int main() {
	Mat frame;
	Mat nextFrame;
	
	vector<Car> cars;
	vector<Car> previousCars;
	VideoCapture cap;

	cap.open("cars.mp4");
	cap.read(frame);
	
	int counter = 0;

	while (1) {

		int key = waitKey(15);
		cap.read(nextFrame);
		if (nextFrame.empty() || key == 'q')
			break;
		
		vector<Rect> carsRect = drawCarsContours(frame, contoursOfDetectedMove(frame, nextFrame));
		createCountingLine(frame);
		

		for (Rect car : carsRect) {
			if (!matchToExistingCars(cars, car))
				cars.push_back(Car(car));
			else {
				previousCars.push_back(Car(car));
			}
		}

		for (Car car : previousCars) {
			if (car.onTheLine(Point(0, frame.rows / 5 * 3), Point(frame.cols, frame.rows / 5 * 3)) && !car.wasCounted()) {
				counter++;
				car.counted = true;
			}
			car.frameOffset++;
		}

		showAmountOfCars(frame, counter);
		cout << previousCars.size() << endl;
		previousCars.clear();

		imshow("window", frame);
		frame = nextFrame.clone();
	}
	destroyAllWindows();
}
