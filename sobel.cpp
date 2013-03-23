// Alessandro Gentilini

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <iostream>

using namespace cv;

int main( int argc, char** argv )
{

  Mat src, src_gray;
  Mat L1_gradient_magnitude;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
    { return -1; }

  //GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src, src_gray, CV_RGB2GRAY );

  
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
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

  for ( int x = 0; x < phi_on_edge.cols; x+=2 ){
    for ( int y = 0; y < phi_on_edge.rows; y+=2 ){
      if(mag_bin.at<unsigned char>(y,x)==255){
        float angle = phi_radian.at<float>(y,x);
        cv::Point tip(x+10*cos(angle),y+10*sin(angle));
        cv::line(phi_on_edge,cv::Point(x,y),tip,CV_RGB(255,0,0));
        cv::circle(phi_on_edge,tip,2,CV_RGB(0,255,0));
      }
    }
  }

  imshow("phase for pixels belonging to the edge",phi_on_edge);

  cv::convertScaleAbs(grad_x,grad_x);
  cv::convertScaleAbs(grad_y,grad_y);


  waitKey(0);

  return 0;
}

