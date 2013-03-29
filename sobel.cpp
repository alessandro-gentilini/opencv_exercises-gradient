// Alessandro Gentilini, 2013

#include "sobel.h"

int main( int argc, char** argv )
{
  // begin of model creation

  // read the model image
  cv::Mat model_img = cv::imread( argv[1] );
  if( !model_img.data ) { 
    std::cerr << "Error loading model image: " << argv[1] << "\n";
    return 1; 
  }

  // compute the model
  std::vector< int > angles;
  std::vector< R_table_t > rts;
  compute_model(model_img,rts,angles);

  // save the 0 degree R table on file (for regression test)
  std::ofstream rt_file("rt_0.csv"); 
  rt_file << "# " << argv[1] << " " << argv[2] << "\n";
  rt_file << rts[0];

  // end of model creation


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  // begin of model search inside the scene image

  // read the scene image
  cv::Mat scene(cv::imread( argv[2] ));
  if( !scene.data ) { 
    std::cerr << "Error loading scene image: " << argv[2] << "\n";
    return 1; 
  }

  // search the model inside the scene
  Locations_t locations;
  Votes_t votes;
  search( scene, rts, angles, locations, votes );

  // save the results (for regression test)
  std::ofstream result_file("result.csv");
  result_file << "# " << argv[1] << " " << argv[2] << "\n";
  result_file << "angle,votes,location_x,location_y\n";  
  size_t i = 0;
  for ( auto it = locations.begin(); it != locations.end(); ++it ) {
    result_file << it->first << "," << votes[i] << "," << it->second.x << "," << it->second.y << "\n";
    i++;
  }  

  // end of search

  return 0;
}

void search(const cv::Mat& scene, const Model_Tables_t& rts, const Model_Angles_t& angles, Locations_t& locations, Votes_t& votes)
{
  cv::Mat scene_mag;
  gradient_L2_norm(scene,scene_mag);
  cv::convertScaleAbs(scene_mag,scene_mag);

  cv::Mat scene_mag_bin;
  cv::threshold(scene_mag,scene_mag_bin,128,255,cv::THRESH_BINARY);

  cv::Mat scene_phi_radian;
  gradient_phase(scene,scene_phi_radian,false);

  std::vector< cv::Point > scene_mymask;
  for ( int x = 0; x < scene_mag_bin.cols; x+=1 ){
    for ( int y = 0; y < scene_mag_bin.rows; y+=1 ){
      if(scene_mag_bin.at<unsigned char>(y,x)==255){
        scene_mymask.push_back(cv::Point(x,y));
      }
    }
  }

  locations.clear();
  votes.resize(rts.size());
  for ( size_t i = 0; i < rts.size(); i++ ) {
    cv::Point loc;
    locate(scene.rows,scene.cols,scene_mymask,scene_phi_radian,rts[i],loc,votes[i]);
    locations.insert(std::make_pair(angles[i],loc));
  }
}

void locate(int scene_rows,int scene_cols,const std::vector<cv::Point>& mask,const cv::Mat& gradient_phase_radians,const R_table_t& rt,cv::Point& location,size_t& nvotes)
{
  cv::Mat accumulator = cv::Mat::zeros(scene_rows,scene_cols,cv::DataType<int>::type);
  for ( size_t i = 0; i < mask.size(); i++ ) {
    float angle = rad2deg(gradient_phase_radians.at<float>(mask[i].y,mask[i].x));
    auto ret(rt.equal_range(static_cast<int>(angle)));
    for (R_table_t::const_iterator it1=ret.first; it1!=ret.second; ++it1){
      cv::Point candidate = it1->second + mask[i];
      if (candidate.y >= 0 && candidate.y<accumulator.rows && candidate.x >=0 && candidate.x<accumulator.cols) {
        accumulator.at<int>(candidate.y,candidate.x)++;
      }
    }    
  }

  double a_min,a_max;
  cv::Point p_min,p_max;
  cv::minMaxLoc(accumulator,&a_min,&a_max,&p_min,&p_max);
  location = p_max;
  nvotes = a_max;
}

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

void gradient_L1_norm(const cv::Mat& img, cv::Mat& norm)
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  cv::Mat gray;
  cv::cvtColor( img, gray, CV_RGB2GRAY );
  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( model_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( model_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, norm );
}

void gradient_L2_norm(const cv::Mat& img, cv::Mat& norm)
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  cv::Mat gray;
  cv::cvtColor( img, gray, CV_RGB2GRAY );
  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;

  /// Gradient X
  //Scharr( model_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

  /// Gradient Y
  //Scharr( model_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

  cv::Mat grad_x_f, grad_y_f;
  grad_x.convertTo(grad_x_f,CV_32F);
  grad_y.convertTo(grad_y_f,CV_32F);

  cv::magnitude(grad_x_f,grad_y_f,norm);
}

void gradient_phase(const cv::Mat& img, cv::Mat& phase, bool is_degree )
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  cv::Mat gray;
  cv::cvtColor( img, gray, CV_RGB2GRAY );
  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;

  /// Gradient X
  //Scharr( model_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );

  /// Gradient Y
  //Scharr( model_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

  cv::Mat grad_x_f, grad_y_f;
  grad_x.convertTo(grad_x_f,CV_32F);
  grad_y.convertTo(grad_y_f,CV_32F);

  cv::phase(grad_x_f,grad_y_f,phase,is_degree);
}

void compute_R_table(const cv::Mat& img, R_table_t& rt, cv::Point& centroid)
{
  cv::Mat L2_gradient_magnitude;
  gradient_L2_norm(img,L2_gradient_magnitude);
  cv::convertScaleAbs(L2_gradient_magnitude,L2_gradient_magnitude);

  cv::Mat phi_radian;
  gradient_phase(img,phi_radian,false);

  std::vector< cv::Point > mymask;
  compute_R_table(L2_gradient_magnitude,phi_radian,rt,centroid,mymask);  
}

void compute_R_table(const cv::Mat& gradient_norm, const cv::Mat& gradient_phase_radians, R_table_t& rt, cv::Point& centroid, std::vector<cv::Point>& mask)
{
  rt.clear();

  cv::Mat mag_bin;
  cv:threshold(gradient_norm,mag_bin,128,255,cv::THRESH_BINARY);

  std::vector< cv::Point > mymask;
  centroid = cv::Point(0,0);
  for ( int x = 0; x < mag_bin.cols; x+=1 ){//15
    for ( int y = 0; y < mag_bin.rows; y+=1 ){
      if(mag_bin.at<unsigned char>(y,x)==255){
        cv::Point p(x,y);
        mymask.push_back(p);
        centroid += p;
      }
    }
  }

  if ( mymask.empty() ) return;

  centroid.x /= mymask.size();
  centroid.y /= mymask.size();  

  for ( size_t i = 0; i < mymask.size(); i++ ){
    float angle = rad2deg(gradient_phase_radians.at<float>(mymask[i].y,mymask[i].x));
    cv::Point r = centroid-mymask[i];
    rt.insert(std::make_pair(static_cast<int>(angle),r));
  }

  mask = mymask;
}

void draw_R_table_sample(cv::Mat& img, const cv::Mat& gradient_phase_radians, const std::vector<cv::Point>& mask, size_t period, const cv::Point& centroid)
{
  for ( size_t i = 0; i < mask.size(); i++ ) {
    if ( i%period==0 ) {
      float angle = gradient_phase_radians.at<float>(mask[i].y,mask[i].x);
      float arm_length = 50;
      cv::Point tip(mask[i].x+arm_length*cos(angle),mask[i].y+arm_length*sin(angle));
      cv::line(img,mask[i],tip,CV_RGB(255,0,0));
      cv::circle(img,tip,2,CV_RGB(0,255,0));    
      std::ostringstream oss;
      oss << i;
      cv::putText(img,oss.str(),tip,cv::FONT_HERSHEY_SIMPLEX,1,CV_RGB(255,0,0));
      cv::line(img,centroid,mask[i],CV_RGB(0,255,255));
    }
  }
  cv::circle(img,centroid,2,CV_RGB(255,255,0)); 
  draw_cross(img,centroid,100,CV_RGB(0,0,255));
}

std::ostream& operator<<(std::ostream& os, const R_table_t& rt)
{
  os << "gradient_phase,r_x,r_y\n";
  for ( auto it = rt.begin(); it != rt.end(); ++it ) {
    auto range(rt.equal_range(it->first));
    for ( auto jt=range.first; jt!=range.second; ++jt ){
      os << it->first << "," << jt->second.x << "," << jt->second.y << "\n";
    }
  }
  return os;
}

void compute_model(const cv::Mat& model_img,std::vector< R_table_t >& rts, std::vector< int >& angles)
{
  rts.clear();
  angles.clear();

  // compute the R table for angle 0, to get an initial value for the centroid
  R_table_t rt0;
  cv::Point centroid;
  compute_R_table(model_img,rt0,centroid);

  // compute the R tables for various model orientation
  size_t jdx = 0;
  for ( int angle = 0; angle < 360; angle+=45 ) {
    angles.push_back( angle );

    cv::Mat rot( cv::getRotationMatrix2D( centroid, angle, 1 ) );
    cv::Mat rotated;
    cv::warpAffine(model_img,rotated,rot,model_img.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,CV_RGB(255,255,255));

    std::ostringstream name;
    name << "rotated" << angle << ".bmp";
    cv::imwrite(name.str(),rotated);

    R_table_t rt;
    cv::Point dont_care;
    compute_R_table(rotated,rt,dont_care);
    rts.push_back(rt);
  }
}