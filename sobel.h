#ifdef _WIN64
#define _USE_MATH_DEFINES
#include <ppl.h>
#elif _WIN32
//define something for Windows (32-bit)
#elif __linux
// linux
#elif __unix // all unices not caught above
// Unix
#elif __posix
// POSIX
#endif

#include <iostream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <list>
#include <tuple>
#include <set>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

template< typename T >
T rad2deg( const T &r )
{
    return 180 * r / M_PI;
}

typedef int angle_t;
typedef size_t vote_t;

typedef std::map<angle_t, cv::Point> Locations_t;
typedef std::vector< vote_t > Votes_t;
typedef std::multimap<angle_t, cv::Point> R_table_t;
typedef std::vector< R_table_t > Model_Tables_t;
typedef std::vector< angle_t > Model_Angles_t;
typedef void (*locate_fun_t)(int scene_rows, int scene_cols, const std::vector<cv::Point> &pixel_on_edge, const cv::Mat &gradient_phase_radians, const R_table_t &rt, cv::Point &location, vote_t &nvotes);

void draw_cross(cv::Mat &img, const cv::Point center, float arm_length, const cv::Scalar &color );
void draw_cross_45(cv::Mat &img, const cv::Point center, float arm_length, const cv::Scalar &color );
void gradient_L1_norm(const cv::Mat &img, cv::Mat &norm);
void gradient_L2_norm(const cv::Mat &img, cv::Mat &norm);
void gradient_phase(const cv::Mat &img, cv::Mat &phase, bool is_degree );

std::ostream &operator<<(std::ostream &os, const R_table_t &rt);

void compute_model(const cv::Mat &model_img, const Model_Angles_t &angles, Model_Tables_t &rts, cv::Point &centroid );
void compute_R_table(const std::vector< cv::Point > &rotated_corners, const cv::Mat &gradient_norm, const cv::Mat &gradient_phase_radians, R_table_t &rt, cv::Point &centroid, std::vector<cv::Point> &pixel_on_edge);
void compute_R_table(const cv::Mat &img, const std::vector< cv::Point > &rotated_corners, R_table_t &rt, cv::Point &centroid);
void draw_R_table_sample(cv::Mat &img, const cv::Mat &gradient_phase_radians, const std::vector<cv::Point> &mask, size_t period, const cv::Point &centroid);

void locate(int scene_rows, int scene_cols, const std::vector<cv::Point> &pixel_on_edge, const cv::Mat &gradient_phase_radians, const R_table_t &rt, cv::Point &location, vote_t &nvotes);
void search(locate_fun_t locate_fun, const cv::Mat &scene, const Model_Tables_t &rts, const Model_Angles_t &angles, Locations_t &locations, Votes_t &votes);
void rotate( const cv::Mat &img, const cv::Point &center, angle_t angle, cv::Mat &rotated, std::vector< cv::Point > &rotated_corners );

void save_first_table(const char *filename, const R_table_t &t, const char *model, const char *scene);
void save_result(const char* filename,const char* model,const char* scene,const Locations_t& locations,const Votes_t& votes, cv::Mat& scene_img, cv::Point& best_location);
void save_model_stats(const Model_Tables_t& rts);

#ifdef _WIN64
void parallel_locate(int scene_rows, int scene_cols, const std::vector<cv::Point> &pixel_on_edge, const cv::Mat &gradient_phase_radians, const R_table_t &rt, cv::Point &location, vote_t &nvotes);
#endif