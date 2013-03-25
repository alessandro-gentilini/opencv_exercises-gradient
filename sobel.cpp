// Alessandro Gentilini

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <map>
#include <unordered_map>

void draw_cross(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color )
{
  cv::Point N(center-cv::Point(0,arm_length));
  cv::Point S(center+cv::Point(0,arm_length));
  cv::Point E(center+cv::Point(arm_length,0));
  cv::Point W(center-cv::Point(arm_length,0));  
  cv::line(img,N,S,color);
  cv::line(img,E,W,color);
}

void draw_cross_45(cv::Mat& img, const cv::Point center, float arm_length, const cv::Scalar& color )
{
  cv::Point NE(center+cv::Point(arm_length,arm_length));
  cv::Point SW(center+cv::Point(-arm_length,-arm_length));
  cv::Point SE(center+cv::Point(arm_length,-arm_length));
  cv::Point NW(center+cv::Point(-arm_length,arm_length));  
  cv::line(img,NE,SW,color);
  cv::line(img,SE,NW,color);
}

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

int main( int argc, char** argv )
{

  cv::Mat model, model_gray;
  cv::Mat L1_gradient_magnitude;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Load an image
  model = cv::imread( argv[1] );

  cv::imshow("model",model);

  if( !model.data )
    { return -1; }

  //GaussianBlur( model, model, Size(3,3), 0, 0, cv::BORDER_DEFAULT );

  /// Convert it to gray
  cv::cvtColor( model, model_gray, CV_RGB2GRAY );

  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( model_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  Sobel( model_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( model_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  Sobel( model_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, L1_gradient_magnitude );


  cv::imshow( "L1 gradient magnitude", L1_gradient_magnitude );

  
  

  cv::Mat grad_x_f, grad_y_f;
  grad_x.convertTo(grad_x_f,CV_32F);
  grad_y.convertTo(grad_y_f,CV_32F);


  cv::Mat mag;
  cv::magnitude(grad_x_f,grad_y_f,mag);

  cv::convertScaleAbs(mag,mag);
  cv::imshow("L2 gradient magnitude",mag);

  cv::Mat mag_bin;
  cv:threshold(mag,mag_bin,128,255,cv::THRESH_BINARY);
  cv::imshow("binarized L2 gradient magnitude",mag_bin);

  cv::Mat phi_degree;
  cv::phase(grad_x_f,grad_y_f,phi_degree,true);

  cv::Mat phi_radian;
  cv::phase(grad_x_f,grad_y_f,phi_radian,false);

  cv::convertScaleAbs(phi_degree,phi_degree);
  cv::imshow("phase [degree]",phi_degree);  

  cv::Mat phi_on_edge;
  cv::bitwise_and(phi_degree,mag_bin,phi_on_edge);

  cvtColor( phi_on_edge, phi_on_edge, CV_GRAY2RGB );

  std::vector< cv::Point > mymask;

  cv::Point centroid(0,0);
  for ( int x = 0; x < phi_on_edge.cols; x+=1 ){//15
    for ( int y = 0; y < phi_on_edge.rows; y+=1 ){
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
    if ( i%200 == 0 ) {
      cv::line(phi_on_edge,centroid,mymask[i],CV_RGB(0,255,255));
    }
    R_table.insert(std::make_pair(static_cast<int>(angle),centroid-mymask[i]));
  }

  std::cout << "\nR table:\n";
  for ( R_table_t::const_iterator it = R_table.begin(); it != R_table.end(); ++it ) {
    std::pair <R_table_t::iterator, R_table_t::iterator> ret(R_table.equal_range(it->first));
    std::cout << it->first << "\n";
    for (R_table_t::iterator it1=ret.first; it1!=ret.second; ++it1){
      std::cout << "\t" << it1->second << "\n";
    }
  }

  for ( size_t i = 0; i < mymask.size(); i+=1 ) {
    if ( i%200==0 ) {
      float angle = phi_radian.at<float>(mymask[i].y,mymask[i].x);
      float arm_length = 50;
      cv::Point tip(mymask[i].x+arm_length*cos(angle),mymask[i].y+arm_length*sin(angle));
      cv::line(phi_on_edge,mymask[i],tip,CV_RGB(255,0,0));
      cv::circle(phi_on_edge,tip,2,CV_RGB(0,255,0));    
      std::ostringstream oss;
      oss << i;
      cv::putText(phi_on_edge,oss.str(),tip,cv::FONT_HERSHEY_SIMPLEX,1,CV_RGB(255,0,0));
    }
  }
  cv::circle(phi_on_edge,centroid,2,CV_RGB(255,255,0)); 
  draw_cross(phi_on_edge,centroid,100,CV_RGB(0,0,255));

  size_t idx = 0;
  for ( R_table_t::const_iterator it = R_table.begin(); it != R_table.end(); ++it ) {
    if ( (idx++)%200==0 ) {
      //cv::line(phi_on_edge,centroid-it->second,centroid,CV_RGB(255,0,255));
    }
  }
  
  



  cv::imshow("phase for pixels belonging to the edge",phi_on_edge);

  cv::convertScaleAbs(grad_x,grad_x);
  cv::convertScaleAbs(grad_y,grad_y);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cv::Mat scene(cv::imread( argv[2] ));

  if( !scene.data )
    { return -1; }

  cv::Mat scene_gray;
  cv::Mat scene_L1_gradient_magnitude;

  /// Convert it to gray
  cv::cvtColor( scene, scene_gray, CV_RGB2GRAY );
  /// Generate grad_x and grad_y
  cv::Mat scene_grad_x, scene_grad_y;
  cv::Mat scene_abs_grad_x, scene_abs_grad_y;
  /// Gradient X
  cv::Sobel( scene_gray, scene_grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( scene_grad_x, scene_abs_grad_x );
  /// Gradient Y
  cv::Sobel( scene_gray, scene_grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( scene_grad_y, scene_abs_grad_y );
  /// Total Gradient (approximate)
  cv::addWeighted( scene_abs_grad_x, 1, scene_abs_grad_y, 1, 0, scene_L1_gradient_magnitude );


  cv::imshow( "scene L1 gradient magnitude", scene_L1_gradient_magnitude );

  cv::Mat scene_grad_x_f, scene_grad_y_f;
  scene_grad_x.convertTo(scene_grad_x_f,CV_32F);
  scene_grad_y.convertTo(scene_grad_y_f,CV_32F);


  cv::Mat scene_mag;
  cv::magnitude(scene_grad_x_f,scene_grad_y_f,scene_mag);

  cv::convertScaleAbs(scene_mag,scene_mag);
  cv::imshow("scene L2 gradient magnitude",scene_mag);

  cv::Mat scene_mag_bin;
  cv::threshold(scene_mag,scene_mag_bin,128,255,cv::THRESH_BINARY);
  cv::imshow("scene binarized L2 gradient magnitude",scene_mag_bin);

  cv::Mat scene_phi_degree;
  cv::phase(scene_grad_x_f,scene_grad_y_f,scene_phi_degree,true);

  cv::Mat scene_phi_radian;
  cv::phase(scene_grad_x_f,scene_grad_y_f,scene_phi_radian,false);

  cv::convertScaleAbs(scene_phi_degree,scene_phi_degree);
  cv::imshow("scene phase [degree]",scene_phi_degree);  

  cv::Mat scene_phi_on_edge;
  cv::bitwise_and(scene_phi_degree,scene_mag_bin,scene_phi_on_edge);

  cv::cvtColor( scene_phi_on_edge, scene_phi_on_edge, CV_GRAY2RGB );

  std::vector< cv::Point > scene_mymask;
  for ( int x = 0; x < scene_phi_on_edge.cols; x+=1 ){
    for ( int y = 0; y < scene_phi_on_edge.rows; y+=1 ){
      if(scene_mag_bin.at<unsigned char>(y,x)==255){
        cv::Point p(x,y);
        scene_mymask.push_back(p);
      }
    }
  }

  cv::Point location;
  int cnt = 0;
  std::unordered_map<cv::Point,int,hash_point> votes;
  

  cv::Mat accumulator = cv::Mat::zeros(scene_gray.rows,scene_gray.cols,cv::DataType<int>::type);
  std::cout << "\nSearch:\n";
  for ( size_t i = 0; i < scene_mymask.size(); i++ ) {
    float angle = rad2deg(scene_phi_radian.at<float>(scene_mymask[i].y,scene_mymask[i].x));
    std::pair <R_table_t::iterator, R_table_t::iterator> ret(R_table.equal_range(static_cast<int>(angle)));
    if (ret.first != ret.second){
      std::cout << angle << "\n";
      for (R_table_t::iterator it1=ret.first; it1!=ret.second; ++it1){
        std::cout << "\t" << it1->second << "\n";
        cv::Point candidate = it1->second + scene_mymask[i];
        if (candidate.y<accumulator.rows && candidate.x<accumulator.cols) {
          accumulator.at<int>(candidate.y,candidate.x)++;
        }
        //draw_cross(scene,scene_mymask[i],3);
        //cv::line(scene,scene_mymask[i],candidate,CV_RGB(0,0,255));
        
        if ( votes.count(candidate) ) {
          votes[candidate]++;
        } else {
          votes.insert(std::make_pair(candidate,1));
        }
        if ( votes[candidate] > cnt ){
          location = candidate;
          cnt = votes[candidate];
        }
        
      }    
    }
  }

  double a_min,a_max;
  cv::Point p_min,p_max;
  cv::minMaxLoc(accumulator,&a_min,&a_max,&p_min,&p_max);
  std::cerr << "min=" << a_min << " @ " << p_min << "\n";
  std::cerr << "max=" << a_max << " @ " << p_max << "\n";
  std::cerr << "cnt=" << cnt << " @ " << location << "\n";

  cv::Mat acc_to_show;
  cv::convertScaleAbs( accumulator, acc_to_show );
  cv::imshow("accumulator",acc_to_show);

  
  
  draw_cross(scene,location,100,CV_RGB(0,0,255));
  draw_cross_45(scene,p_max,100,CV_RGB(255,0,0));
  cv::imshow("Found",scene);




  cv::waitKey(0);

  return 0;
}

