#include<iostream>
#include<fstream>
#include<sstream>

using namespace std;

#include"/home/wendy/slam_study/include/slamBase.h"
#include<pcl/visualization/cloud_viewer.h>

// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );
// 度量运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

int main( int argc, char** argv )
{
	ParameterReader pd;
	int startIndex = atoi( pd.getData( "start_index" ).c_str() );
	int endIndex = atoi( pd.getData( "end_index" ).c_str() );
	
	// initialize
	cout<<"Initializing ..."<<endl;
	int currIndex = startIndex; // 当前索引为currIndex
	FRAME lastFrame = readFrame( currIndex, pd ); // 上一帧数据
	
	// 我们总是在比较currFrame和lastFrame
	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.fx = atof( pd.getData( "camera.fx" ).c_str());
	camera.fy = atof( pd.getData( "camera.fy" ).c_str());
	camera.cx = atof( pd.getData( "camera.cx" ).c_str());
	camera.cy = atof( pd.getData( "camera.cy" ).c_str());
	camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
	
	computeKeyPointsAndDesp( lastFrame );
	PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, camera );
	
	pcl::visualization::CloudViewer viewer("viewer");
	
	//是否显示点云
	bool visualize = pd.getData("visualize_pointcloud")==string("yes");
	
	int min_inliers = atoi( pd.getData("min_inliers").c_str() );
	double max_norm = atof( pd.getData("max_norm").c_str() );
	
	for( currIndex = startIndex + 1; currIndex < endIndex; currIndex++ )
	{
		cout<<"Reading files "<<currIndex<<endl;
		
		FRAME currFrame = readFrame( currIndex, pd );//读取currFrame
		computeKeyPointsAndDesp( currFrame );
		
		RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, camera );
		if( result.inliers < min_inliers )//inliers不够，放弃该帧
			continue;
		//计算运动范围是否太大
		double norm = normofTransform(result.rvec, result.tvec);
		cout<<"norm= "<<norm<<endl;
		if( norm >= max_norm )
			continue;
		Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
		cout<<"T="<<T.matrix()<<endl;
		
		cloud = joinPointCloud( cloud, currFrame, T, camera );
		
		if( visualize == true )
		    viewer.showCloud( cloud );
			
		lastFrame = currFrame;
	}
	
	cout<<"PCD file has been saved to /home/wendy/slam_study/data/result2.pcd."<<endl;
	pcl::io::savePCDFile( "/home/wendy/slam_study/data/result2.pcd", *cloud );
	return 0;
}

FRAME readFrame( int index, ParameterReader& pd )
{
	FRAME f; 
	string rgbDir = pd.getData("rgb_dir");
	string depthDir = pd.getData("depth_dir");
	
	string rgbExt = pd.getData("rgb_extension");
	string depthExt = pd.getData("depth_extension");
	
	string sss;
	
	sss = rgbDir + to_string(index) + rgbExt;
	f.rgb = cv::imread( sss );
	
	sss = depthDir + to_string(index) + depthExt;
	f.depth = cv::imread( sss, -1 );
	
	return f;
}
//判断两帧是否相距过远
double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
	return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}
