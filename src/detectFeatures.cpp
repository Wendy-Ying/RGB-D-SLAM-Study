#include <iostream>
#include "/home/wendy/slam_study/include/slamBase.h"
using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include </home/wendy/opencv-3.4.15/opencv_contrib-3.4.15/modules/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
 
int main(int argc,char** argv)
{
  // 声明并从data文件夹里读取两个rgb与深度图
  cv::Mat rgb1 = cv::imread( "/home/wendy/slam_study/data/rgb1.png");
  cv::Mat rgb2 = cv::imread( "/home/wendy/slam_study/data/rgb2.png");
  cv::Mat depth1 = cv::imread( "/home/wendy/slam_study/data/depth1.png", -1);
  cv::Mat depth2 = cv::imread( "/home/wendy/slam_study/data/depth2.png", -1);
 
  //声明特征提取器与描述子提取器
  cv::Ptr<cv::FeatureDetector> _detector;//智能指针，指向FeatureDetector对象
  cv::Ptr<cv::DescriptorExtractor> _descriptor;
 
 //初始化 创建对象
  _detector=cv::ORB::create();
  _descriptor=cv::ORB::create();
 
  vector<cv::KeyPoint> kp1,kp2;
  _detector->detect(rgb1,kp1);
  _detector->detect(rgb2,kp2);
 
  cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
 
  //画出第一张图片的关键点
  cv::Mat imgShow;
  cv::drawKeypoints(rgb1,kp1,imgShow,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("keypoints",imgShow);
  cv::imwrite("/home/wendy/slam_study/data/keypoint.png",imgShow);
  cv::waitKey(0);
 
  //计算描述子
  cv::Mat desp1,desp2;
  _descriptor->compute(rgb1,kp1,desp1);
  _descriptor->compute(rgb2,kp2,desp2);
 
  //匹配描述子
  //初始化匹配器
  cv::Ptr<cv::DescriptorMatcher> matcher;
  matcher=cv::DescriptorMatcher::create("BruteForce-Hamming");
  vector<cv::DMatch> matches;
  matcher->match(desp1,desp2,matches);
  cout<<"Find total "<<matches.size()<<" matches."<<endl;
 
  //显示匹配的特征
  cv::Mat imgMatches;
  cv::drawMatches(rgb1,kp1,rgb2,kp2,matches,imgMatches);
  cv::imshow("matches",imgMatches);
  cv::imwrite( "/home/wendy/slam_study/data/matches.png", imgMatches );
  cv::waitKey( 0 );
 
  //筛选匹配，把距离较大的去掉
  //准则是去掉大于四倍距离的匹配
  vector<cv::DMatch> goodMatches;
  double minDis=9999;
  for(size_t i=0;i<matches.size();i++ )
  {
    if(matches[i].distance<minDis)
    minDis=matches[i].distance;
  }
  for(size_t i=0;i<matches.size();i++)
  {
    if(matches[i].distance<=2*minDis)
    goodMatches.push_back(matches[i]);
  }
 
  // 显示 good matches
  cout<<"good matches="<<goodMatches.size()<<endl;
  cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
  cv::imshow( "good matches", imgMatches );
  cv::imwrite( "/home/wendy/slam_study/data/good_matches.png", imgMatches );
  cv::waitKey(0);
 
  //计算图像间的运动关系
  //cv::solvePnPRansac()
  //为调用此函数准备必要的参数
 
  //第一个帧的三维点
  vector<cv::Point3f> pts_obj;
  //第二个帧的图像点
  vector<cv::Point2f> pts_img;
 
  //相机内参
  CAMERA_INTRINSIC_PARAMETERS C;
  C.cx = 325.5;
  C.cy = 253.5;
  C.fx = 518.0;
  C.fy = 519.0;
  C.scale = 1000.0;
 
  for(size_t i=0;i<goodMatches.size();i++)
  {
    // query 是第一个, train 是第二个
    cv::Point2f p=kp1[goodMatches[i].queryIdx].pt;
    //获取d要注意行和列
    ushort d=depth1.ptr<ushort>(int(p.y))[int(p.x)];
    if(d==0)continue;
    pts_img.push_back(cv::Point2f(kp2[goodMatches[i].trainIdx].pt));
 
    //将(u\v\d)转换为x、y、z
    cv::Point3f pt(p.x,p.y,d);
    cv::Point3f pd=point2dTo3d(pt,C);
    pts_obj.push_back(pd);    
  }
  double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}};
  
  //构建相机矩阵
  cv::Mat cameraMatrix(3,3,CV_64F,camera_matrix_data);
  cv::Mat rvec,tvec,inliers;
 
  //求解PnP
  cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec,false,100,1.0,100,inliers);
 
  cout<<"inliers:"<<inliers.rows<<endl;
  cout<<"R="<<rvec<<endl;
  cout<<"t="<<tvec<<endl;
 
  //画出inliers匹配
  vector<cv::DMatch> matchesShow;
  for(size_t i=0;i<inliers.rows;i++)
  {
    matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
  }
  cv::drawMatches(rgb1,kp1,rgb2,kp2,matchesShow,imgMatches);
  cv::imshow( "inlier matches", imgMatches );
  cv::imwrite( "/home/wendy/slam_study/data/inlier.png", imgMatches );
}
