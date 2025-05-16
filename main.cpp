#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <iomanip>
#include <csignal>

static std::vector<cv::Point> rdp(const std::vector<cv::Point>& pts, double eps) {
    if (pts.size() < 3)
        return pts;

    cv::Point A = pts.front();
    cv::Point B = pts.back();
    double dx = B.x - A.x;
    double dy = B.y - A.y;
    double norm = std::hypot(dx, dy);

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

    return { A, B };
}

static cv::Vec4i createSingleTrajectory(const std::vector<cv::Vec4i>& lines,
                                         double maxDistance,
                                         int imageWidth, int imageHeight) {
    if (lines.empty()) return {0, 0, 0, 0};

    std::vector<cv::Point> allPts;
    allPts.reserve(lines.size() * 2);
    for (const auto& ln : lines) {
        allPts.emplace_back(ln[0], ln[1]);
        allPts.emplace_back(ln[2], ln[3]);
    }

    cv::Vec4f fl;
    cv::fitLine(allPts, fl, cv::DIST_L2, 0, 0.01, 0.01);

    float vx = fl[0];
    float vy = fl[1];
    float x0 = fl[2];
    float y0 = fl[3];

    if (std::abs(vy) < 0.1f) {
        double tL = -x0 / vx;
        double tR = (imageWidth - x0) / vx;
        int yL = static_cast<int>(std::round(y0 + tL * vy));
        int yR = static_cast<int>(std::round(y0 + tR * vy));
        return {0, yL, imageWidth, yR};
    }

    double tTop = -y0 / vy;
    double tBot = (imageHeight - y0) / vy;
    int xTop = static_cast<int>(std::clamp(std::round(x0 + tTop * vx), 0.0, static_cast<double>(imageWidth)));
    int xBot = static_cast<int>(std::clamp(std::round(x0 + tBot * vx), 0.0, static_cast<double>(imageWidth)));

    if (vy > 0)
        return {xTop, 0, xBot, imageHeight};
    else
        return {xBot, imageHeight, xTop, 0};
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
    
    double alpha = 0.2;
    int maxDist = 15;
    
    LowPassFilter devFilter(alpha), angFilter(alpha);
    cv::Vec4i prevLine{0, 0, 0, 0};
    bool locked = false;
    
    sleep(3);

    std::cout << "Starting trajectory detection. Press Ctrl+C to exit." << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    
    while (running) {
        cv::Mat color;
        camera.grab();
        camera.retrieve(color);
        
        if (color.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            continue;
        }
        
        int cx = color.cols / 2;
        int cy = color.rows / 2;

        cv::Vec4i traj{0, 0, 0, 0};
        if (!locked) {
            cv::Mat gray, sx, sy, edges;
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, sx, CV_16S, 1, 0);
            cv::Sobel(gray, sy, CV_16S, 0, 1);
            cv::convertScaleAbs(sx, sx);
            cv::convertScaleAbs(sy, sy);
            cv::bitwise_or(sx, sy, edges);
            cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);

            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);
            if (!lines.empty()) {
                traj = createSingleTrajectory(lines, maxDist, color.cols, color.rows);
                if (traj != cv::Vec4i(0, 0, 0, 0)) {
                    prevLine = traj;
                    locked = true;
                }
            }
        } else {
            traj = prevLine;
        }

        if (traj != cv::Vec4i(0, 0, 0, 0)) {
            cv::Point A(traj[0], traj[1]);
            cv::Point B(traj[2], traj[3]);

            cv::Point farPt = (A.y < B.y ? A : B);
            if (A.y < cy && B.y < cy)
                farPt = (A.y < B.y ? A : B);

            double num = std::abs((B.y - A.y) * cx - (B.x - A.x) * cy + B.x * A.y - B.y * A.x);
            double den = std::hypot(B.y - A.y, B.x - A.x);
            double lat = devFilter.filter(num / den);

            double dx = farPt.x - cx;
            double dy = farPt.y - cy;
            double targetAng = std::atan2(dy, dx) * 180.0 / M_PI;
            double angErr = angFilter.filter(targetAng);

            std::cout << "\rLateral: " << std::setw(6) << lat << " px \t Angular: " << std::setw(6) << angErr << " deg" << std::flush;
        } else {
            std::cout << "\rDetecting..." << std::flush;
        }
        
        usleep(33333);  // ~30fps
    }
    
    std::cout << std::endl << "Exiting..." << std::endl;
    camera.release();
    return 0;
}