#include <iostream>
#include <map>
#include <unordered_map>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

template< typename T >
T rad2deg( const T& r )
{
  return 180*r/M_PI;
}

struct hash_point {
    size_t operator()(const cv::Point& p ) const
    {
        return std::hash<cv::Point::value_type>()(p.x)^std::hash<cv::Point::value_type>()(p.y);
    }
};

void draw_cross(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color );
void draw_cross_45(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color );
void gradient_L1_norm(const cv::Mat& img, cv::Mat& norm);
void gradient_L2_norm(const cv::Mat& img, cv::Mat& norm);
void gradient_phase(const cv::Mat& img, cv::Mat& phase, bool is_degree );

typedef std::multimap<int,cv::Point> R_table_t;
void compute_R_table(const cv::Mat& gradient_norm, const cv::Mat& gradient_phase_radians, R_table_t& rt, cv::Point& centroid, std::vector<cv::Point>& mask);
void compute_R_table(const cv::Mat& img, R_table_t& rt);
void draw_R_table_sample(cv::Mat& img, const cv::Mat& gradient_phase_radians, const std::vector<cv::Point>& mask, size_t period, const cv::Point& centroid);
std::string R_table_to_string(const R_table_t& rt);