#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv){
  cv::VideoCapture capture(CAP_OPENNI);

  cv::Mat image;
  cv::Mat bgrImage;

  while(true){
    capture.grab();
    capture.retrieve(image, CAP_OPENNI_DEPTH_MAP);
    capture.retrieve(bgrImage, CAP_OPENNI_BGR_IMAGE);
    imshow("Image", image);
    imshow("Color", bgrImage);
    if(cv::waitKey(30) >= 0) break;
  }
  return 0;
}
