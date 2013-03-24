// Alessandro Gentilini

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <map>

using namespace cv;

void draw_cross(cv::Mat& img, const cv::Point center, float arm_length )
{
  cv::Point N(center-cv::Point(0,arm_length));
  cv::Point S(center+cv::Point(0,arm_length));
  cv::Point E(center+cv::Point(arm_length,0));
  cv::Point W(center-cv::Point(arm_length,0));  
  cv::line(img,N,S,CV_RGB(255,255,0));
  cv::line(img,E,W,CV_RGB(255,255,0));
}

template< typename T >
T rad2deg( const T& r )
{
  return 180*r/M_PI;
}

int main( int argc, char** argv )
{

  Mat model, model_gray;
  Mat L1_gradient_magnitude;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Load an image
  model = imread( argv[1] );

  if( !model.data )
    { return -1; }

  //GaussianBlur( model, model, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( model, model_gray, CV_RGB2GRAY );

  
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( model_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( model_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( model_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( model_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, L1_gradient_magnitude );


  imshow( "L1 gradient magnitude", L1_gradient_magnitude );

  
  

  cv::Mat grad_x_f, grad_y_f;
  grad_x.convertTo(grad_x_f,CV_32F);
  grad_y.convertTo(grad_y_f,CV_32F);


  cv::Mat mag;
  cv::magnitude(grad_x_f,grad_y_f,mag);

  cv::convertScaleAbs(mag,mag);
  imshow("L2 gradient magnitude",mag);

  cv::Mat mag_bin;
  cv:threshold(mag,mag_bin,128,255,THRESH_BINARY);
  imshow("binarized L2 gradient magnitude",mag_bin);

  cv::Mat phi_degree;
  cv::phase(grad_x_f,grad_y_f,phi_degree,true);

  cv::Mat phi_radian;
  cv::phase(grad_x_f,grad_y_f,phi_radian,false);

  cv::convertScaleAbs(phi_degree,phi_degree);
  imshow("phase [degree]",phi_degree);  

  cv::Mat phi_on_edge;
  cv::bitwise_and(phi_degree,mag_bin,phi_on_edge);

  cvtColor( phi_on_edge, phi_on_edge, CV_GRAY2RGB );

  std::vector< cv::Point > mymask;

  cv::Point centroid(0,0);
  for ( int x = 0; x < phi_on_edge.cols; x+=15 ){
    for ( int y = 0; y < phi_on_edge.rows; y+=15 ){
      if(mag_bin.at<unsigned char>(y,x)==255){
        cv::Point p(x,y);
        mymask.push_back(p);
        centroid += p;
      }
    }
  }
  centroid.x /= mymask.size();
  centroid.y /= mymask.size();

  std::cout << "\nPoint on boundary index,angles in degree (clock wise)\n";
  for ( size_t i = 0; i < mymask.size(); i++ ){
    float angle = rad2deg(phi_radian.at<float>(mymask[i].y,mymask[i].x));
    std::cout << i << "\t,\t" << angle << "\n";
  }  

  typedef std::multimap<int,cv::Point> R_table_t;
  R_table_t R_table;

  for ( size_t i = 0; i < mymask.size(); i++ ){
    float angle = rad2deg(phi_radian.at<float>(mymask[i].y,mymask[i].x));
    cv::line(phi_on_edge,centroid,mymask[i],CV_RGB(255,0,255));
    R_table.insert(std::make_pair(angle,centroid-mymask[i]));
  }

  std::cout << "\nR table:\n";
  for ( R_table_t::const_iterator it = R_table.begin(); it != R_table.end(); ++it ) {
    std::pair <R_table_t::iterator, R_table_t::iterator> ret(R_table.equal_range(it->first));
    std::cout << it->first << "\n";
    for (R_table_t::iterator it1=ret.first; it1!=ret.second; ++it1){
      std::cout << "\t" << it1->second << "\n";
    }
  }

  for ( size_t i = 0; i < mymask.size(); i++ ) {
    float angle = phi_radian.at<float>(mymask[i].y,mymask[i].x);
    float arm_length = 50;
    cv::Point tip(mymask[i].x+arm_length*cos(angle),mymask[i].y+arm_length*sin(angle));
    cv::line(phi_on_edge,mymask[i],tip,CV_RGB(255,0,0));
    cv::circle(phi_on_edge,tip,2,CV_RGB(0,255,0));    
    std::ostringstream oss;
    oss << i;
    cv::putText(phi_on_edge,oss.str(),tip,FONT_HERSHEY_SIMPLEX,1,CV_RGB(255,0,0));
  }
  cv::circle(phi_on_edge,centroid,2,CV_RGB(255,255,0)); 
  draw_cross(phi_on_edge,centroid,100);

  
  for ( R_table_t::const_iterator it = R_table.begin(); it != R_table.end(); ++it ) {
    cv::line(phi_on_edge,centroid-it->second,centroid,CV_RGB(0,255,255));
  }
  



  imshow("phase for pixels belonging to the edge",phi_on_edge);

  cv::convertScaleAbs(grad_x,grad_x);
  cv::convertScaleAbs(grad_y,grad_y);






  waitKey(0);

  return 0;
}

