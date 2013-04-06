// Alessandro Gentilini, 2013

#include "sobel.h"


struct opencv_mat_plus
{
    cv::Mat operator()(const cv::Mat &m1, const cv::Mat &m2) const
    {
        return m1 + m2;
    }
};

bool show_dbg_img;
bool save_regression_data;

int main( int argc, char **argv )
{
    show_dbg_img = argc >= 4 && argv[3] == std::string("1");
    save_regression_data = argc >= 5 && argv[4] == std::string("1");

    // begin of model creation

    // read the model image
    cv::Mat model_img = cv::imread( argv[1] );
    if ( !model_img.data )
    {
        std::cerr << "Error loading model image: " << argv[1] << "\n";
        return 1;
    }

    // compute the angles for the model
    Model_Angles_t angles;
    for ( angle_t a = 0; a < 360; a += 45 )
    {
        angles.push_back(a);
    }

    // compute the model
    Model_Tables_t rts;
    cv::Point centroid;
    compute_model(model_img, angles, rts, centroid);

    if ( save_regression_data )
    {
        // save the 0 degree R table on file (for regression test)
        save_first_table("rt_0.csv", rts[0], argv[1], argv[2]);

        save_model_stats(rts[0]);
    }

    // end of model creation


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // begin of model search inside the scene image

    // read the scene image
    cv::Mat scene(cv::imread( argv[2] ));
    if ( !scene.data )
    {
        std::cerr << "Error loading scene image: " << argv[2] << "\n";
        return 1;
    }

    // search the model inside the scene
    Locations_t locations;
    Votes_t votes;

#ifdef _WIN64
    search( parallel_locate, scene, rts, angles, locations, votes );
#else
    search( locate, scene, rts, angles, locations, votes );
#endif

    cv::Point best_location;
    if ( save_regression_data )
    {
        // save the results (for regression test)
        save_result("result.csv", argv[1], argv[2], locations, votes, scene, best_location);
    }

    // end of search

    if ( show_dbg_img )
    {
        draw_cross(model_img, centroid, 40, CV_RGB(255, 0, 0));
        cv::imshow("model", model_img);
        cv::imwrite("model.bmp", model_img);
        draw_cross_45(scene, best_location, 40, CV_RGB(255, 0, 0));
        cv::imshow("scene", scene);
        cv::imwrite("scene.bmp", scene);
        cv::waitKey();
    }

    return 0;
}

// Search for the model inside the scene.
// locate_fun is the function fot locating the model in an image
// scene      is the image where to search for the model
// rts        is the collection of the R tables "rotated" at various angles
// angles     is the collection of the rotation angles of the R tables
// locations  is the collection of found locations
// votes      is the collection of votes related to the locations
void search(locate_fun_t locate_fun, const cv::Mat &scene, const Model_Tables_t &rts, const Model_Angles_t &angles, Locations_t &locations, Votes_t &votes)
{
   // Compute the norm of the gradient
   cv::Mat scene_mag;
   gradient_L2_norm(scene, scene_mag);
   cv::convertScaleAbs(scene_mag, scene_mag);

   // Binarize the gradient magnitude and assuming that the resulting white pixels are the pixels belonging to the edges
   cv::Mat scene_mag_bin;
   cv::threshold(scene_mag, scene_mag_bin, 128, 255, cv::THRESH_BINARY);

   // Compute the phase of the gradient (todo: avoid to compute the phase for all the pixels, maybe it is faster to compute it just for the edge's pixels?)
   cv::Mat scene_phi_radian;
   gradient_phase(scene, scene_phi_radian, false);

   // Fill the collection containing the pixels on the edges
   std::vector< cv::Point > pixels_on_edge;
   for ( int x = 0; x < scene_mag_bin.cols; x += 1 )
   {
      for ( int y = 0; y < scene_mag_bin.rows; y += 1 )
      {
         if (scene_mag_bin.at<unsigned char>(y, x) == 255)
         {
            pixels_on_edge.push_back(cv::Point(x, y));
         }
      }
   }

   // Search for the model in the scene
   locations.clear();
   votes.resize(rts.size());
   for ( size_t i = 0; i < rts.size(); i++ )
   {
      cv::Point loc;
      // todo: potrebbe aver senso aggiornare gli accumulatori dei voti in un colpo solo, dato il pixel su edge ho il suo angolo del gradiente e con questo angolo vado a 
      // controllare tutte le R table e ad aggiornare tutti gli accumulatori, poi passo al pixel successivo.
      locate_fun(scene.rows, scene.cols, pixels_on_edge, scene_phi_radian, rts[i], loc, votes[i]);
      locations.insert(std::make_pair(angles[i], loc));
   }
}

void locate(int scene_rows, int scene_cols, const std::vector<cv::Point> &pixel_on_edge, const cv::Mat &gradient_phase_radians, const R_table_t &rt, cv::Point &location, vote_t &nvotes)
{
   cv::Mat accumulator = cv::Mat::zeros(scene_rows, scene_cols, cv::DataType<int>::type);
   for ( size_t i = 0; i < pixel_on_edge.size(); i++ )
   {
      // todo: avere gli angoli in gradi (quindi la conversione radianti -> gradi farla solo una volta)
      float angle = rad2deg(gradient_phase_radians.at<float>(pixel_on_edge[i].y, pixel_on_edge[i].x));
      auto range(rt.equal_range(round(angle)));
      for (auto it = range.first; it != range.second; ++it)
      {
         cv::Point candidate = it->second + pixel_on_edge[i];
         if (candidate.y >= 0 && candidate.y < accumulator.rows && candidate.x >= 0 && candidate.x < accumulator.cols)
         {
            accumulator.at<int>(candidate.y, candidate.x)++;
         }
      }
   }

   double a_min, a_max;
   cv::Point p_min, p_max;
   cv::minMaxLoc(accumulator, &a_min, &a_max, &p_min, &p_max);
   location = p_max;
   nvotes = a_max;
}

#ifdef _WIN64
void parallel_locate(int scene_rows, int scene_cols, const std::vector<cv::Point> &pixel_on_edge, const cv::Mat &gradient_phase_radians, const R_table_t &rt, cv::Point &location, vote_t &nvotes)
{
    typedef int acc_t;
    concurrency::combinable<cv::Mat> count([&scene_rows, &scene_cols]()
    {
        return cv::Mat::zeros(scene_rows, scene_cols, cv::DataType<acc_t>::type);
    });
    concurrency::parallel_for_each(pixel_on_edge.cbegin(), pixel_on_edge.cend(),
                                   [&count, &gradient_phase_radians, &rt, &scene_rows, &scene_cols](cv::Point p)
    {
        float angle = rad2deg(gradient_phase_radians.at<float>(p.y, p.x));
        auto range(rt.equal_range(round(angle)));
        for (auto it = range.first; it != range.second; ++it)
        {
            cv::Point candidate = it->second + p;
            if ( candidate.y >= 0 &&  candidate.y < scene_rows && candidate.x >= 0 && candidate.x < scene_cols)
            {
                count.local().at<acc_t>(candidate.y, candidate.x)++;
            }
        }
    });

    double a_min, a_max;
    cv::Point p_min, p_max;
    cv::minMaxLoc(count.combine( opencv_mat_plus() ), &a_min, &a_max, &p_min, &p_max);
    location = p_max;
    nvotes = a_max;
}
#endif

void draw_cross(cv::Mat &img, const cv::Point center, float arm_length, const cv::Scalar &color )
{
   cv::Point N(center - cv::Point(0, arm_length));
   cv::Point S(center + cv::Point(0, arm_length));
   cv::Point E(center + cv::Point(arm_length, 0));
   cv::Point W(center - cv::Point(arm_length, 0));
   cv::line(img, N, S, color);
   cv::line(img, E, W, color);
}

void draw_cross_45(cv::Mat &img, const cv::Point center, float arm_length, const cv::Scalar &color )
{
   cv::Point NE(center + cv::Point(arm_length, arm_length));
   cv::Point SW(center + cv::Point(-arm_length, -arm_length));
   cv::Point SE(center + cv::Point(arm_length, -arm_length));
   cv::Point NW(center + cv::Point(-arm_length, arm_length));
   cv::line(img, NE, SW, color);
   cv::line(img, SE, NW, color);
}

void gradient_L1_norm(const cv::Mat &img, cv::Mat &norm)
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

void gradient_L2_norm(const cv::Mat &img, cv::Mat &norm)
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
   grad_x.convertTo(grad_x_f, CV_32F);
   grad_y.convertTo(grad_y_f, CV_32F);

   cv::magnitude(grad_x_f, grad_y_f, norm);
}

void gradient_phase(const cv::Mat &img, cv::Mat &phase, bool is_degree )
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
   grad_x.convertTo(grad_x_f, CV_32F);
   grad_y.convertTo(grad_y_f, CV_32F);

   cv::phase(grad_x_f, grad_y_f, phase, is_degree);
}

void compute_R_table(const cv::Mat &img, const std::vector< cv::Point > &rotated_corners, R_table_t &rt, cv::Point &centroid)
{
   cv::Mat L2_gradient_magnitude;
   gradient_L2_norm(img, L2_gradient_magnitude);
   cv::convertScaleAbs(L2_gradient_magnitude, L2_gradient_magnitude);

   cv::Mat phi_radian;
   gradient_phase(img, phi_radian, false);

   std::vector< cv::Point > pixel_on_edge;
   compute_R_table(rotated_corners, L2_gradient_magnitude, phi_radian, rt, centroid, pixel_on_edge);
}

void compute_R_table(const std::vector< cv::Point > &rotated_corners, const cv::Mat &gradient_norm, const cv::Mat &gradient_phase_radians, R_table_t &rt, cv::Point &centroid, std::vector<cv::Point> &pixel_on_edge)
{
   rt.clear();
   pixel_on_edge.clear();

   cv::Mat mag_bin;
   cv::threshold(gradient_norm, mag_bin, 128, 255, cv::THRESH_BINARY);

   centroid = cv::Point(0, 0);
   for ( int x = 0; x < mag_bin.cols; x += 1 ) //15
   {
      for ( int y = 0; y < mag_bin.rows; y += 1 )
      {
         cv::Point query(x, y);
         // I use the polygon test to exclude pixels that are on the false edge due to rotation
         if ( cv::pointPolygonTest(rotated_corners, query, true) >= 2 && mag_bin.at<unsigned char>(y, x) == 255)
         {
            cv::Point p(x, y);
            pixel_on_edge.push_back(p);
            centroid += p;
         }
         else
         {
            mag_bin.at<unsigned char>(y, x) = 0;
         }
      }
   }

   if ( show_dbg_img )
   {
      static int idx = 0;

      std::ostringstream name_1;
      name_1 << idx << "_gradient_norm";
      imshow(name_1.str(), gradient_norm);

      std::ostringstream name_2;
      name_2 << idx << "_filtered_gradient_norm";
      imshow(name_2.str(), mag_bin);

      idx++;
   }

   if ( pixel_on_edge.empty() ) return;

   centroid.x /= pixel_on_edge.size();
   centroid.y /= pixel_on_edge.size();

   for ( size_t i = 0; i < pixel_on_edge.size(); i++ )
   {
      float angle = rad2deg(gradient_phase_radians.at<float>(pixel_on_edge[i].y, pixel_on_edge[i].x));
      cv::Point r = centroid - pixel_on_edge[i];
      rt.insert(std::make_pair(round(angle), r));
   }
}

void draw_R_table_sample(cv::Mat &img, const cv::Mat &gradient_phase_radians, const std::vector<cv::Point> &mask, size_t period, const cv::Point &centroid)
{
   for ( size_t i = 0; i < mask.size(); i++ )
   {
      if ( i % period == 0 )
      {
         float angle = gradient_phase_radians.at<float>(mask[i].y, mask[i].x);
         float arm_length = 50;
         cv::Point tip(mask[i].x + arm_length * cos(angle), mask[i].y + arm_length * sin(angle));
         cv::line(img, mask[i], tip, CV_RGB(255, 0, 0));
         cv::circle(img, tip, 2, CV_RGB(0, 255, 0));
         std::ostringstream oss;
         oss << i;
         cv::putText(img, oss.str(), tip, cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));
         cv::line(img, centroid, mask[i], CV_RGB(0, 255, 255));
      }
   }
   cv::circle(img, centroid, 2, CV_RGB(255, 255, 0));
   draw_cross(img, centroid, 100, CV_RGB(0, 0, 255));
}

/// Operator for printing an R table
std::ostream &operator<<(std::ostream &os, const R_table_t &rt)
{
   std::set< angle_t > angles;
   for ( auto it = rt.begin(); it != rt.end(); ++it ) {
      angles.insert(it->first);
   }

   os << "gradient_phase,r_x,r_y\n";
   for ( auto it = angles.begin(); it != angles.end(); ++it )
   {
      auto range(rt.equal_range(*it));
      for ( auto jt = range.first; jt != range.second; ++jt )
      {
         os << *it << "," << jt->second.x << "," << jt->second.y << "\n";
      }
   }
   return os;
}

double distance(const cv::Point &a, const cv::Point &b)
{
   return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void compute_model(const cv::Mat &model_img, const std::vector< int > &angles, Model_Tables_t &rts, cv::Point &centroid)
{
    // clear the return values
    rts.clear();

    // compute the R table for angle 0, to get an initial value for the centroid
    R_table_t rt0;
    std::vector< cv::Point > rotated_corners(4);
    rotated_corners[0] = cv::Point(0, 0);
    rotated_corners[1] = cv::Point(model_img.cols - 1, 0);
    rotated_corners[2] = cv::Point(model_img.cols - 1, model_img.rows - 1);
    rotated_corners[3] = cv::Point(0, model_img.rows - 1);
    compute_R_table(model_img, rotated_corners, rt0, centroid);

    // compute the R tables for various model orientation
    for ( size_t i = 0; i < angles.size(); i++ )
    {
        cv::Mat rotated;
        std::vector< cv::Point > rotated_corners(4);
        rotate( model_img, centroid, angles[i], rotated, rotated_corners );

        if ( show_dbg_img )
        {
            // save the rotated model image
            std::ostringstream name;
            name << "rotated" << angles[i] << ".bmp";
            cv::imwrite(name.str(), rotated);
        }

        // compute the R table for the rotated model image
        R_table_t rt;
        cv::Point dont_care;
        compute_R_table(rotated, rotated_corners, rt, dont_care);
        rts.push_back(rt);
    }
}

// Rotate the img about the point center, angle is in degree.
// rotated is the rotated image, rotated_corners are the position of
// the rotated corners of image.
void rotate( const cv::Mat &img, const cv::Point &center, angle_t angle, cv::Mat &rotated, std::vector< cv::Point > &rotated_corners )
{
   cv::Mat rot( 2, 3, cv::DataType<double>::type );
   rot = cv::getRotationMatrix2D( center, angle, 1 );

   // todo: I would like to use the cv::convertPointsToHomogeneous
   cv::Mat midpoint(3, 1, cv::DataType<double>::type);
   midpoint.at<double>(0, 0) = img.cols / 2.0;
   midpoint.at<double>(1, 0) = img.rows / 2.0;
   midpoint.at<double>(2, 0) = 1;

   cv::Mat top_left(3, 1, cv::DataType<double>::type);
   top_left.at<double>(0, 0) = 0;
   top_left.at<double>(1, 0) = 0;
   top_left.at<double>(2, 0) = 1;

   cv::Mat top_right(3, 1, cv::DataType<double>::type);
   top_right.at<double>(0, 0) = img.cols - 1;
   top_right.at<double>(1, 0) = 0;
   top_right.at<double>(2, 0) = 1;

   cv::Mat bottom_right(3, 1, cv::DataType<double>::type);
   bottom_right.at<double>(0, 0) = img.cols - 1;
   bottom_right.at<double>(1, 0) = img.rows - 1;
   bottom_right.at<double>(2, 0) = 1;

   cv::Mat bottom_left(3, 1, cv::DataType<double>::type);
   bottom_left.at<double>(0, 0) = 0;
   bottom_left.at<double>(1, 0) = img.rows - 1;
   bottom_left.at<double>(2, 0) = 1;

   cv::Mat top_left_rotated = rot * top_left;
   cv::Mat top_right_rotated = rot * top_right;
   cv::Mat bottom_right_rotated = rot * bottom_right;
   cv::Mat bottom_left_rotated = rot * bottom_left;
   cv::Mat rotated_midpoint = rot * midpoint;

   rotated_corners[0] = cv::Point(round(top_left_rotated.at<double>(0, 0)),round(top_left_rotated.at<double>(1, 0)));
   rotated_corners[1] = cv::Point(round(top_right_rotated.at<double>(0, 0)), round(top_right_rotated.at<double>(1, 0)));
   rotated_corners[2] = cv::Point(round(bottom_right_rotated.at<double>(0, 0)), round(bottom_right_rotated.at<double>(1, 0)));
   rotated_corners[3] = cv::Point(round(bottom_left_rotated.at<double>(0, 0)), round(bottom_left_rotated.at<double>(1, 0)));

   cv::Rect bb( cv::boundingRect( rotated_corners ) );
   cv::Size sz(bb.width, bb.height);

   cv::Point displacement(round(bb.width / 2.0 - rotated_midpoint.at<double>(0, 0)), round(bb.height / 2.0 - rotated_midpoint.at<double>(1, 0)));

   rot.at<double>(0, 2) += displacement.x;
   rot.at<double>(1, 2) += displacement.y;

   std::transform(rotated_corners.begin(), rotated_corners.end(), rotated_corners.begin(), [&displacement](const cv::Point & p)
   {
      return p + displacement;
   });

   cv::warpAffine(img, rotated, rot, sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT, CV_RGB(0, 0, 0));
}

void create_a_model()
{
   cv::Mat a_rect(99, 99, cv::DataType<unsigned char>::type);
   a_rect = 255 * cv::Mat::ones(99, 99, cv::DataType<unsigned char>::type);
   cv::Point top_left(33, 33);
   cv::Point bottom_right(65, 65);
   cv::rectangle(a_rect, top_left, bottom_right, 0);
   cv::line(a_rect, top_left, bottom_right, 0);
   cv::line(a_rect, cv::Point(46, 52), cv::Point(52, 46), 0);
   cv::rectangle(a_rect, bottom_right - cv::Point(5, 5), bottom_right, 0);
   imwrite("a_rect.bmp", a_rect);
}



void save_first_table(const char *filename, const R_table_t &t, const char *model, const char *scene)
{
   std::ofstream rt_file(filename);
   rt_file << "# " << model << " " << scene << "\n";
   rt_file << t;
   rt_file.close();
}

void save_result(const char* filename,const char* model,const char* scene,const Locations_t& locations,const Votes_t& votes, cv::Mat& scene_img, cv::Point& best_location)
{
   std::ofstream result_file(filename);
   result_file << "# " << model << " " << scene << "\n";
   result_file << "angle,votes,location_x,location_y\n";
   size_t i = 0;
   vote_t max_vote = 0;
   const size_t sz = 6;
   const cv::Scalar colors[sz] = {CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), CV_RGB(0, 255, 255), CV_RGB(255, 255, 0), CV_RGB(255, 0, 255)};
   for ( auto it = locations.begin(); it != locations.end(); ++it )
   {
      result_file << it->first << "," << votes[i] << "," << it->second.x << "," << it->second.y << "\n";
      draw_cross(scene_img, it->second, 40, colors[i % sz]);
      if ( votes[i] > max_vote )
      {
         best_location = it->second;
         max_vote = votes[i];
      }
      i++;
   }
   result_file.close();
}


bool Point_less(const cv::Point &lhs, const cv::Point &rhs)
{
   return std::tie(lhs.x,lhs.y) < std::tie(rhs.x,rhs.y);
}

double norm( const cv::Point& p )
{
   return sqrt( p.x*p.x + p.y*p.y );
}

void save_model_stats(const R_table_t &rt)
{
    size_t n_super_m = rt.size();
    std::set< angle_t > angles;
    for ( auto it = rt.begin(); it != rt.end(); ++it )
    {
        angles.insert( it->first );
    }

    std::ofstream file("wf.csv");
    file << "Delta_phi,eta\n";

    double epsilon = 0.5;
    double Delta_phi = deg2rad(0.1);
    double step = deg2rad(0.1);
    double max_Delta_phi = deg2rad(3.0);

    const size_t sz = max_Delta_phi/step;

    for ( size_t i = 0; i < sz; i++ )
    {
        size_t cnt = 0;
        for ( auto it = angles.begin(); it != angles.end(); ++it )
        {
            auto range( rt.equal_range( *it ) );
            for ( auto jt = range.first; jt != range.second; ++jt )
            {
                if ( norm(jt->second) <= (2 * epsilon) / Delta_phi )
                {
                    cnt++;
                }
            }
        }
        file << rad2deg(Delta_phi) << "," << static_cast<double>(cnt) / n_super_m << "\n";
        Delta_phi += step;
    }
    file.close();
}

