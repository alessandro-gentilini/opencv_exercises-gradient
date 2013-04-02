#include <iostream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>

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

class Augmented_point : public cv::Point
{
public:
	Augmented_point(int x, int y):cv::Point(x,y){cnt_=0;}
	Augmented_point(const cv::Point& p):cv::Point(p){cnt_=0;}
	int cnt_;
	bool operator==(const Augmented_point& ap)const{return ap==*this;}
};



void draw_cross(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color );
void draw_cross_45(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color );
void gradient_L1_norm(const cv::Mat& img, cv::Mat& norm);
void gradient_L2_norm(const cv::Mat& img, cv::Mat& norm);
void gradient_phase(const cv::Mat& img, cv::Mat& phase, bool is_degree );

//typedef std::multimap<int,Augmented_point> R_table_t;
typedef std::multimap<int,cv::Point> R_table_t;



std::ostream& operator<<(std::ostream& os, const R_table_t& rt);

typedef std::vector< R_table_t > Model_Tables_t;
typedef std::vector< int > Model_Angles_t;

void compute_model(const cv::Mat& model_img, const Model_Angles_t& angles, Model_Tables_t& rts, cv::Point& centroid );

void compute_R_table(const std::vector< cv::Point > &rotated_corners, const cv::Mat& gradient_norm, const cv::Mat& gradient_phase_radians, R_table_t& rt, cv::Point& centroid, std::vector<cv::Point>& pixel_on_edge);
void compute_R_table(const cv::Mat& img, const std::vector< cv::Point > &rotated_corners, R_table_t& rt, cv::Point& centroid);
void draw_R_table_sample(cv::Mat& img, const cv::Mat& gradient_phase_radians, const std::vector<cv::Point>& mask, size_t period, const cv::Point& centroid);


void locate(int scene_rows,int scene_cols,const std::vector<cv::Point>& pixel_on_edge,const cv::Mat& gradient_phase_radians,const R_table_t& rt,cv::Point& location,size_t& nvotes);

typedef int angle_t;
typedef std::map<angle_t,cv::Point> Locations_t;

typedef std::vector< size_t > Votes_t;
void search(const cv::Mat& scene, const Model_Tables_t& rts, const Model_Angles_t& angles, Locations_t& locations, Votes_t& votes);
void rotate( const cv::Mat &img, const cv::Point &center, int angle, cv::Mat &rotated, std::vector< cv::Point > &rotated_corners );