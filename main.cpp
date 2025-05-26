#include <opencv2/opencv.hpp>
#include <raspicam/raspicam_cv.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <csignal>
#include <vector>
#include <chrono>


static std::vector<cv::Point> rdp(const std::vector<cv::Point>& pts, double eps) {
    if (pts.size() < 3)
        return pts;

    cv::Point A = pts.front();
    cv::Point B = pts.back();
    double dx = B.x - A.x;
    double dy = B.y - A.y;
    double norm = std::hypot(dx, dy);

    if (norm < 1e-6)
        return {A, B};

    double maxDist = 0.0;
    int idx = 0;
    for (int i = 1; i + 1 < static_cast<int>(pts.size()); ++i) {
        const cv::Point& P = pts[i];
        double dist = std::abs(dy * P.x - dx * P.y + B.x * A.y - B.y * A.x) / norm;
        if (dist > maxDist) {
            maxDist = dist;
            idx = i;
        }
    }

    if (maxDist > eps) {
        std::vector<cv::Point> left(pts.begin(), pts.begin() + idx + 1);
        std::vector<cv::Point> right(pts.begin() + idx, pts.end());

        auto leftRes = rdp(left, eps);
        auto rightRes = rdp(right, eps);

        leftRes.insert(leftRes.end(), rightRes.begin() + 1, rightRes.end());
        return leftRes;
    }

    return {A, B};
}


class LowPassFilter {
private:
    double alpha;
    double prevValue;
    bool initialized;

public:
    LowPassFilter(double a = 0.1) : alpha(a), prevValue(0.0), initialized(false) {}

    double filter(double value) {
        if (!initialized) {
            prevValue = value;
            initialized = true;
            return value;
        }
        double newValue = alpha * value + (1.0 - alpha) * prevValue;
        prevValue = newValue;
        return newValue;
    }

    void reset() {
        initialized = false;
    }
};

static std::vector<cv::Point> findLineContour(const cv::Mat& edges, const cv::Mat& frame) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    if (contours.empty())
        return {};
    

    size_t longestIdx = 0;
    size_t maxLength = 0;
    
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() > maxLength) {
            maxLength = contours[i].size();
            longestIdx = i;
        }
    }
    

    if (maxLength < 50)
        return {};
    
    return contours[longestIdx];
}


static std::pair<double, double> calculateDeviation(const std::vector<cv::Point>& trajectory, 
                                                    int cx, int cy, int height) {
    if (trajectory.size() < 2)
        return {0.0, 0.0};

    cv::Point farthest = trajectory[0];
    for (const auto& pt : trajectory) {
        if (pt.y < farthest.y) {
            farthest = pt;
        }
    }
    
    double minDist = std::numeric_limits<double>::max();
    cv::Point closestPt;
    int closestSegment = -1;
    
    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        cv::Point A = trajectory[i];
        cv::Point B = trajectory[i + 1];
        

        double dx = B.x - A.x;
        double dy = B.y - A.y;
        double t = std::max(0.0, std::min(1.0, ((cx - A.x) * dx + (cy - A.y) * dy) / (dx * dx + dy * dy + 1e-6)));
        
        cv::Point proj(A.x + t * dx, A.y + t * dy);
        double dist = std::hypot(cx - proj.x, cy - proj.y);
        
        if (dist < minDist) {
            minDist = dist;
            closestPt = proj;
            closestSegment = i;
        }
    }
    

    double lateral = closestPt.x - cx;
    

    double dx = farthest.x - cx;
    double dy = cy - farthest.y;
    double angular = std::atan2(dx, dy) * 180.0 / M_PI;
    
    return {lateral, angular};
}


static cv::Mat getROI(const cv::Mat& frame) {


    int height = frame.rows;
    int width = frame.cols;
    

    return frame;
}

volatile bool running = true;

void sigHandler(int sig) {
    running = false;
}

int main() {
    signal(SIGINT, sigHandler);
    

    raspicam::RaspiCam_Cv camera;
    
    camera.set(cv::CAP_PROP_FORMAT, CV_8UC3);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera.set(cv::CAP_PROP_FPS, 30);
    
    if (!camera.open()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }
    

    double alpha = 0.15;        
    double rdpEpsilon = 2.5;      
    int sobelThreshold = 40;     
    int lineThickness = 10;      
    

    LowPassFilter lateralFilter(alpha);
    LowPassFilter angularFilter(alpha);
    

    std::cout << "Waiting for camera to stabilize..." << std::endl;
    cv::waitKey(2000);
    

    cv::namedWindow("Line Follower - Downward Camera", cv::WINDOW_AUTOSIZE);
    
    std::cout << "Starting line follower (downward-facing camera). Press Ctrl+C to exit." << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    
    auto lastTime = std::chrono::steady_clock::now();
    int frameCount = 0;
    
    while (running) {
        cv::Mat frame;
        camera.grab();
        camera.retrieve(frame);
        
        if (frame.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            continue;
        }
        

        int cx = frame.cols / 2;
        int cy = frame.rows / 2;
        

        cv::Mat roi = getROI(frame);
        

        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        

        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
        


        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        

        std::vector<cv::Point> lineContour = findLineContour(binary, frame);
        

        cv::Mat display = frame.clone();
        

        cv::circle(display, cv::Point(cx, cy), 8, cv::Scalar(255, 0, 255), -1);
        cv::circle(display, cv::Point(cx, cy), 15, cv::Scalar(255, 0, 255), 2);
        

        cv::arrowedLine(display, cv::Point(cx, cy), cv::Point(cx, cy - 50), 
                       cv::Scalar(255, 255, 0), 2, cv::LINE_AA, 0, 0.3);
        
        if (!lineContour.empty()) {

            std::vector<cv::Point> simplified = rdp(lineContour, rdpEpsilon);
            
            if (simplified.size() >= 2) {

                cv::polylines(display, simplified, false, cv::Scalar(0, 255, 0), 3);
                

                for (const auto& pt : simplified) {
                    cv::circle(display, pt, 4, cv::Scalar(0, 255, 255), -1);
                }
                

                auto [lateral, angular] = calculateDeviation(simplified, cx, cy, frame.rows);
                

                lateral = lateralFilter.filter(lateral);
                angular = angularFilter.filter(angular);
                

                cv::Point farthest = simplified[0];
                for (const auto& pt : simplified) {
                    if (pt.y < farthest.y) {
                        farthest = pt;
                    }
                }
                

                cv::line(display, cv::Point(cx, cy), farthest, cv::Scalar(0, 0, 255), 2);
                

                cv::circle(display, farthest, 8, cv::Scalar(0, 0, 255), -1);
                

                int lateralPx = static_cast<int>(lateral);
                cv::line(display, cv::Point(cx, cy - 5), cv::Point(cx + lateralPx, cy - 5), 
                        cv::Scalar(255, 0, 0), 3);
                

                std::string lateralText = "Lateral: " + std::to_string(static_cast<int>(lateral)) + " px";
                std::string angularText = "Angular: " + std::to_string(static_cast<int>(angular)) + " deg";
                

                cv::rectangle(display, cv::Point(5, 5), cv::Point(250, 90), 
                             cv::Scalar(0, 0, 0), -1);
                
                cv::putText(display, lateralText, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                cv::putText(display, angularText, cv::Point(10, 60), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                

                std::string direction = (lateral > 5) ? "Turn LEFT" : 
                                      (lateral < -5) ? "Turn RIGHT" : "STRAIGHT";
                cv::putText(display, direction, cv::Point(10, 85), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                

                std::cout << "\rLateral: " << std::setw(6) << lateral 
                         << " px \t Angular: " << std::setw(6) << angular 
                         << " deg \t " << std::setw(10) << direction << std::flush;
            }
        } else {

            cv::putText(display, "NO LINE DETECTED", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            std::cout << "\rNo line detected...                              " << std::flush;
        }
        

        for (int i = 0; i < display.cols; i += 40) {
            cv::line(display, cv::Point(i, 0), cv::Point(i, display.rows), 
                    cv::Scalar(50, 50, 50), 1);
        }
        for (int i = 0; i < display.rows; i += 40) {
            cv::line(display, cv::Point(0, i), cv::Point(display.cols, i), 
                    cv::Scalar(50, 50, 50), 1);
        }
        

        frameCount++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count();
        if (elapsed >= 1) {
            double fps = frameCount / static_cast<double>(elapsed);
            cv::putText(display, "FPS: " + std::to_string(static_cast<int>(fps)), 
                       cv::Point(540, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            frameCount = 0;
            lastTime = currentTime;
        }
        

        cv::imshow("Line Follower - Downward Camera", display);
        

        cv::Mat binaryDisplay;
        cv::cvtColor(binary, binaryDisplay, cv::COLOR_GRAY2BGR);
        cv::resize(binaryDisplay, binaryDisplay, cv::Size(320, 240));
        cv::imshow("Binary Detection", binaryDisplay);
        

        if (cv::waitKey(1) == 27) {
            running = false;
        }
    }
    
    std::cout << std::endl << "Shutting down..." << std::endl;
    
    cv::destroyAllWindows();
    camera.release();
    
    return 0;
}
