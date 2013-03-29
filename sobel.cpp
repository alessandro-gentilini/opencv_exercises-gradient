// Alessandro Gentilini

#include "sobel.h"

int main( int argc, char** argv )
{
  cv::Mat model = cv::imread( argv[1] );
  if( !model.data ) { 
    std::cerr << "Error loading model image\n";
    return 1; 
  }

  cv::Mat model_gray;
  cv::cvtColor( model, model_gray, CV_RGB2GRAY );

  cv::Mat L1_gradient_magnitude;
  gradient_L1_norm(model,L1_gradient_magnitude);
  cv::imshow( "L1 gradient magnitude", L1_gradient_magnitude );

  cv::Mat L2_gradient_magnitude;
  gradient_L2_norm(model,L2_gradient_magnitude);
  cv::convertScaleAbs(L2_gradient_magnitude,L2_gradient_magnitude);
  cv::imshow("L2 gradient magnitude",L2_gradient_magnitude);

  cv::Mat mag_bin;
  cv:threshold(L2_gradient_magnitude,mag_bin,128,255,cv::THRESH_BINARY);
  cv::imshow("binarized L2 gradient magnitude",mag_bin);

  cv::Mat phi_degree;
  gradient_phase(model,phi_degree,true);

  cv::Mat phi_radian;
  gradient_phase(model,phi_radian,false);

  cv::convertScaleAbs(phi_degree,phi_degree);
  cv::imshow("phase [degree]",phi_degree);  

  R_table_t R_table;
  cv::Point centroid;
  std::vector< cv::Point > mymask;
  compute_R_table(L2_gradient_magnitude,phi_radian,R_table,centroid,mymask);

  R_table_t R_table_2;
  compute_R_table(model,R_table_2);

  if ( R_table != R_table_2 ) {
    std::cerr << "doh!\n";
    return 1;
  }

  std::ofstream rt_file("rt_0.csv"); 
  rt_file << "# " << argv[1] << " " << argv[2] << "\n";
  rt_file << R_table;

  cv::Mat phi_on_edge;
  cv::bitwise_and(phi_degree,mag_bin,phi_on_edge);
  cvtColor( phi_on_edge, phi_on_edge, CV_GRAY2RGB );


  

  draw_R_table_sample(phi_on_edge,phi_radian,mymask,200,centroid);
  cv::imshow("phase for pixels belonging to the edge",phi_on_edge);

  size_t idx = 0;
  for ( R_table_t::const_iterator it = R_table.begin(); it != R_table.end(); ++it ) {
    if ( (idx++)%200==0 ) {
      //cv::line(phi_on_edge,centroid-it->second,centroid,CV_RGB(255,0,255));
    }
  }


  std::vector< int > angles;
  std::vector< R_table_t > rts;
  size_t jdx = 0;
  for ( int angle = 0; angle < 360; angle+=45 ) {
    angles.push_back( angle );

    cv::Mat rot( cv::getRotationMatrix2D( centroid, angle, 1 ) );
    cv::Mat rotated;
    cv::warpAffine(model,rotated,rot,model.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,CV_RGB(255,255,255));
    std::ostringstream oss;
    oss << jdx++ << ": rotated " << angle << "°";
    cv::imshow(oss.str(),rotated);
    std::ostringstream oss1;
    oss1 << "rotated" << angle << ".bmp";
    cv::imwrite(oss1.str(),rotated);

    R_table_t rt;
    compute_R_table(rotated,rt);
    rts.push_back(rt);
  }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cv::Mat scene(cv::imread( argv[2] ));

  if( !scene.data )
    { 
      std::cerr << "Error";
      return -1; 
    }

  cv::Mat scene_L1_gradient_magnitude;
  gradient_L1_norm(scene,scene_L1_gradient_magnitude);
  cv::imshow( "scene L1 gradient magnitude", scene_L1_gradient_magnitude );

  cv::Mat scene_mag;
  gradient_L2_norm(scene,scene_mag);
  cv::convertScaleAbs(scene_mag,scene_mag);
  cv::imshow("scene L2 gradient magnitude",scene_mag);

  cv::Mat scene_mag_bin;
  cv::threshold(scene_mag,scene_mag_bin,128,255,cv::THRESH_BINARY);
  cv::imshow("scene binarized L2 gradient magnitude",scene_mag_bin);

  cv::Mat scene_phi_degree;
  gradient_phase(scene,scene_phi_degree,true);

  cv::Mat scene_phi_radian;
  gradient_phase(scene,scene_phi_radian,false);

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
  cv::Mat accumulator = cv::Mat::zeros(scene.rows,scene.cols,cv::DataType<int>::type);
  for ( size_t i = 0; i < scene_mymask.size(); i++ ) {
    float angle = rad2deg(scene_phi_radian.at<float>(scene_mymask[i].y,scene_mymask[i].x));
    auto ret(R_table.equal_range(static_cast<int>(angle)));
    if (ret.first != ret.second){
      //std::cout << angle << "\n";
      for (R_table_t::iterator it1=ret.first; it1!=ret.second; ++it1){
        //std::cout << "\t" << it1->second << "\n";
        cv::Point candidate = it1->second + scene_mymask[i];
        if (candidate.y >= 0 && candidate.y<accumulator.rows && candidate.x >= 0 && candidate.x<accumulator.cols) {
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
  /*
  std::cerr << "min=" << a_min << " @ " << p_min << "\n";
  std::cerr << "max=" << a_max << " @ " << p_max << "\n";
  std::cerr << "cnt=" << cnt << " @ " << location << "\n";
  */

  size_t nvotes;
  cv::Point newlocation;
  locate(scene.rows,scene.cols,scene_mymask,scene_phi_radian,R_table,newlocation,nvotes);
  //std::cerr << "cnt=" << nvotes << " @ " << newlocation << "\n";

  size_t max_votes = 0;
  cv::Point good_location;
  size_t angle_index;
  std::cerr << "\n";
 
  std::ofstream result_file("result.csv");
  result_file << "# " << argv[1] << " " << argv[2] << "\n";
  result_file << "angle,votes,location_x,location_y\n";

  for ( size_t i = 0; i < rts.size(); i++ ) {
    size_t nvotes;
    cv::Point newlocation;
    locate(scene.rows,scene.cols,scene_mymask,scene_phi_radian,rts[i],newlocation,nvotes);
    result_file << angles[i] << "," << nvotes << "," << newlocation.x << "," << newlocation.y << "\n";
    if ( nvotes > max_votes ) {
      angle_index = i;
      max_votes = nvotes;
      good_location = newlocation;
    } 
  }

  cv::Mat acc_to_show;
  cv::convertScaleAbs( accumulator, acc_to_show );
  cv::imshow("accumulator",acc_to_show);
  
  draw_cross(scene,good_location,100,CV_RGB(0,0,255));
  draw_cross_45(scene,good_location,100,CV_RGB(255,0,0));
  cv::imshow("Found",scene);

  //cv::waitKey(0);

  return 0;
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

void compute_R_table(const cv::Mat& img, R_table_t& rt)
{
  cv::Mat L2_gradient_magnitude;
  gradient_L2_norm(img,L2_gradient_magnitude);
  cv::convertScaleAbs(L2_gradient_magnitude,L2_gradient_magnitude);

  cv::Mat phi_radian;
  gradient_phase(img,phi_radian,false);

  cv::Point centroid;
  std::vector< cv::Point > mymask;
  compute_R_table(L2_gradient_magnitude,phi_radian,rt,centroid,mymask);  
}

void compute_R_table(const cv::Mat& gradient_norm, const cv::Mat& gradient_phase_radians, R_table_t& rt, cv::Point& centroid, std::vector<cv::Point>& mask)
{
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


