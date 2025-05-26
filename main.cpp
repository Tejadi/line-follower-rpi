#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <csignal>

static std::vector<cv::Point> rdp(const std::vector<cv::Point>& pts,
                                  double eps)
{
    if (pts.size() < 3) return pts;

    cv::Point A = pts.front(), B = pts.back();
    double dx = B.x - A.x, dy = B.y - A.y;
    double norm = std::hypot(dx, dy);
    if (norm < 1e-6) return {A, B};

    double maxDist = 0.0; int idx = 0;
    for (int i = 1; i + 1 < (int)pts.size(); ++i) {
        const auto& P = pts[i];
        double dist = std::abs(dy * P.x - dx * P.y + B.x * A.y - B.y * A.x) / norm;
        if (dist > maxDist) { maxDist = dist; idx = i; }
    }
    if (maxDist > eps) {
        std::vector<cv::Point> left(pts.begin(), pts.begin()+idx+1),
                               right(pts.begin()+idx, pts.end());
        auto l = rdp(left, eps), r = rdp(right, eps);
        l.insert(l.end(), r.begin()+1, r.end());  return l;
    }
    return {A,B};
}

class LowPassFilter {
public:
    explicit LowPassFilter(double a=0.15): alpha(a), init(false), prev(0.0) {}
    double filter(double v){
        if(!init){prev=v; init=true; return v;}
        prev = alpha*v + (1-alpha)*prev; return prev;
    }
private:
    double alpha, prev; bool init;
};

static std::vector<cv::Point> longestContour(const cv::Mat& bin)
{
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if(cs.empty()) return {};
    size_t best=0, maxLen=0;
    for(size_t i=0;i<cs.size();++i)
        if(cs[i].size()>maxLen){maxLen=cs[i].size(); best=i;}
    return (maxLen<50)? std::vector<cv::Point>{} : cs[best];
}

static std::pair<double,double> deviation(const std::vector<cv::Point>& traj,
                                          int cx,int cy,int h)
{
    if(traj.size()<2) return {0,0};
    cv::Point far = *std::min_element(traj.begin(), traj.end(),
                                      [](auto&a,auto&b){return a.y<b.y;});
    double minDist=1e9; cv::Point proj;
    for(size_t i=0;i+1<traj.size();++i){
        auto A=traj[i], B=traj[i+1];
        double dx=B.x-A.x, dy=B.y-A.y;
        double t = ((cx-A.x)*dx + (cy-A.y)*dy)/(dx*dx+dy*dy+1e-6);
        t = std::clamp(t,0.0,1.0);
        cv::Point P(A.x+t*dx, A.y+t*dy);
        double d = std::hypot(cx-P.x, cy-P.y);
        if(d<minDist){minDist=d; proj=P;}
    }
    double lateral = proj.x - cx;
    double angle   = std::atan2(far.x - cx, cy - far.y)*180.0/M_PI;
    return {lateral, angle};
}

volatile bool run=true;
void sigint(int){run=false;}

int main()
{
    std::signal(SIGINT, sigint);
    cv::VideoCapture cam(0, cv::CAP_V4L2);
    if(!cam.isOpened()){
        std::cerr<<"Failed to open /dev/video0\n";
        return -1;
    }
    cam.set(cv::CAP_PROP_FRAME_WIDTH , 640);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cam.set(cv::CAP_PROP_FPS, 30);
    cam.set(cv::CAP_PROP_BUFFERSIZE, 1);

    LowPassFilter latF(0.15), angF(0.15);
    const double eps = 2.5;

    auto t0 = std::chrono::steady_clock::now();
    int frameCnt=0;

    std::cout<<std::fixed<<std::setprecision(1);
    std::cout<<"Running head-less line follower (Ctrl-C to quit)\n";

    while(run){
        cv::Mat frame;  if(!cam.read(frame)||frame.empty()) continue;
        int cx=frame.cols/2, cy=frame.rows/2;

        cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, {5,5}, 1.0);
        cv::Mat bin;  cv::threshold(gray, bin, 0,255,
                                    cv::THRESH_BINARY_INV|cv::THRESH_OTSU);
        auto contour = longestContour(bin);
        if(!contour.empty()){
            auto simp = rdp(contour, eps);
            auto [lat,ang]=deviation(simp,cx,cy,frame.rows);
            lat=latF.filter(lat); ang=angF.filter(ang);

            std::string dir = (lat>5) ? "LEFT" : (lat<-5) ? "RIGHT" : "STRAIGHT";
            std::cout<<"\rLat "<<std::setw(6)<<lat<<" px  "
                     <<"Ang "<<std::setw(6)<<ang<<" deg  "
                     <<dir<<std::flush;
        }else{
            std::cout<<"\rNo line..."<<std::flush;
        }


        ++frameCnt;
        auto now = std::chrono::steady_clock::now();
        if(std::chrono::duration_cast<std::chrono::seconds>(now-t0).count()>=1){
            std::cout<<"  FPS "<<frameCnt<<"\n";
            frameCnt=0; t0=now;
        }
    }
    std::cout<<"\nExiting.\n";
    return 0;
}
