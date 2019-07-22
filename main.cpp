#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat findConvexPoly(Mat &src)
{
    Mat blurImage, edges;
    blur(src, blurImage, Size(3, 3));
    Canny(blurImage, edges, 50, 100);

    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), [&contours](vector<Point> lhs, vector<Point> rhs) { return lhs.size() > rhs.size(); });

    vector<Point> hull;
    convexHull(Mat(contours[0]), hull, false);

    vector<vector<Point>> cnt(1);
    approxPolyDP(hull, cnt[0], 20, true);

    Mat poly = Mat::zeros(edges.size(), CV_8UC1);
    drawContours(poly, cnt, 0, Scalar(255));
    
    return poly;
}

vector<Point2f> detectCornors(Mat &src)
{
    vector<Vec2f> lines;
    HoughLines(src, lines, 1, CV_PI/180, 50, 0, 0);

    Mat labels, centers;
    vector<Point2f> data;
    for (size_t i=0; i<lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        float x = rho*cos(theta), y = rho*sin(theta);
        data.push_back(Point2f(x,y));
    }

    kmeans(data, 4, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 5, KMEANS_PP_CENTERS, centers);
    
    vector<Point2f> fourPoints, xyPoints;
    for (size_t i=0; i<4; i++)
    {
        float x = centers.at<float>(i,0);
        float y = centers.at<float>(i,1);
        float rho = sqrt(x*x+y*y);
        float theta = atan2(y, x);

        xyPoints.push_back(Point2f(x,y));
        fourPoints.push_back(Point2f(rho,theta));
    }

    sort(xyPoints.begin(), xyPoints.end(), [](Point2f &lhs, Point2f &rhs) { return abs(lhs.y/lhs.x) < abs(rhs.y/rhs.x); } );

    vector<Point2f> ans;
    for (size_t i=0; i<2; i++)
    {
        float x0 = xyPoints[i].x;
        float y0 = xyPoints[i].y;

        for (size_t j=2; j<4; j++)
        {
            float x1 = xyPoints[j].x;
            float y1 = xyPoints[j].y;
            float x = (y0*(x1*x1+y1*y1)-y1*(x0*x0+y0*y0))/(y0*x1-x0*y1);
            float y = (x0*(x1*x1+y1*y1)-x1*(x0*x0+y0*y0))/(y1*x0-x1*y0);
            ans.push_back(Point2f(x,y));
        }
    }

    // order of points (top-left, bottom-left, top-right, bottom-right)
    sort(ans.begin(), ans.end(), [](Point2f lhs, Point2f rhs) { return lhs.x<rhs.x; } );
    sort(ans.begin(), ans.begin()+2, [](Point2f lhs, Point2f rhs) { return lhs.y<rhs.y; });
    sort(ans.begin()+2, ans.end(), [](Point2f lhs, Point2f rhs) { return lhs.y<rhs.y; });

    return ans;
}

Mat warpCard(Mat src)
{
    auto convexPoly = findConvexPoly(src);
    auto cornors = detectCornors(convexPoly);
    Mat warpedCard(340, 540, CV_8UC1);
    Mat homography = findHomography(cornors, std::vector<Point>{Point(0,0), Point(0, warpedCard.rows), Point(warpedCard.cols, 0), Point(warpedCard.cols, warpedCard.rows)});
    warpPerspective(src, warpedCard, homography, Size(warpedCard.cols, warpedCard.rows));
    return warpedCard;
}

vector<Mat> getRefOCR()
{
    Mat ref = imread("font.png");
    cvtColor(ref, ref, CV_BGR2GRAY);
    
    vector<vector<Point>> contours;
    findContours(ref, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    sort(contours.begin(), contours.end(), [&contours](vector<Point> lhs, vector<Point> rhs) { return lhs[0].x < rhs[0].x; } );

    vector<Mat> roi(contours.size());
    for (size_t i=0; i<contours.size(); i++)
    {
        Rect boundRect = boundingRect(contours[i]);
        resize(ref(boundRect), roi[i], Size(54,84));
    }
    return roi;
}

vector<Rect> findBoundingRect(Mat src)
{
    Mat tophat;
    Mat ele = getStructuringElement(MORPH_RECT, Size(15, 5));
    morphologyEx(src, tophat, MORPH_TOPHAT, ele);

    Mat sobel;
    Mat gradX, gradY;
    Mat absGradX, absGradY;
    Sobel(tophat, gradX, CV_32F, 1, 0);
    convertScaleAbs(gradX, absGradX);
    Sobel(tophat, gradY, CV_32F, 0, 1);
    convertScaleAbs(gradY, absGradY);
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobel);

    morphologyEx(sobel, sobel, MORPH_CLOSE, ele);
    threshold(sobel, sobel, 0, 255, THRESH_BINARY | THRESH_OTSU);
    morphologyEx(sobel, sobel, MORPH_CLOSE, ele);

    vector<vector<Point>> contours;
    findContours(sobel, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> rects;
    for (auto c:contours)
    {
        Rect r = boundingRect(c);
        if (r.width*1.0/r.height > 2 && r.width*1.0/r.height < 4 && r.width > 80 && r.width < 100 && r.height > 25 && r.height < 45)
        {
            rects.push_back(r);
        }
    }

    sort(rects.begin(), rects.end(), [](Rect lhs, Rect rhs) { return lhs.x < rhs.x; });
    return rects;
}

vector<int> detectDigits(Mat gray, vector<Mat> digitROI)
{
    
    auto rects = findBoundingRect(gray);
    
    vector<Mat> roi(4);
    vector<int> ans;
    for (size_t i=0; i<rects.size(); i++)
    {
        roi[i] = gray(Rect(rects[i].x-5, rects[i].y-5, rects[i].width+10, rects[i].height+10));
        blur(roi[i], roi[i], Size(3, 3));
        threshold(roi[i], roi[i], 0, 255, THRESH_BINARY | THRESH_OTSU);
        vector<vector<Point>> digitCnt;
        findContours(roi[i].clone(), digitCnt, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        sort(digitCnt.begin(), digitCnt.end(), [](vector<Point> lhs, vector<Point> rhs) { return lhs[0].x < rhs[0].x; });

        for (size_t j=0; j<digitCnt.size(); j++)
        {
            Rect r = boundingRect(digitCnt[j]);
            Mat d = roi[i](r);
            resize(d,d,Size(54,84));

            int score=INT_MIN;
            int idx=-1;
            for (size_t k=0; k<digitROI.size(); k++)
            {
                Mat res;
                matchTemplate(d, digitROI[k], res, CV_TM_CCOEFF);
                double minVal, maxVal;
                minMaxLoc(res, &minVal, &maxVal);
                if (maxVal>score)
                {
                    score = maxVal;
                    idx=k;
                }
            }
            ans.push_back(idx);
        }
    }
    return ans;
}


int main(int argc, char *argv[])
{
    if (argc !=2)
        throw invalid_argument("No filename provided");

    Mat src = imread(argv[1]);
    resize(src, src, Size(540, 540.0*src.rows/src.cols));
    cvtColor(src, src, CV_BGR2GRAY);

    auto warpedCard = warpCard(src);
    auto ref = getRefOCR();
    auto ans = detectDigits(warpedCard, ref);

    cout<<"Account Number: ";
    for (int i=0; i<16; i++)
    {
        cout<<ans[i];
        if ((i+1)%4==0 && i!=15)
            cout<<"-";
    }
    cout<<endl;
    return 0;
}
