#include<iostream>
#include<fstream>
#include<sstream>

using namespace std;

#include"/home/wendy/slam_study/include/slamBase.h"
#include<pcl/visualization/cloud_viewer.h>

//g2o的头文件
#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


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
	
	//g2o初始化
	typedef g2o::BlockSolver_6_3 SlamBlockSolver;
	typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;
	
	// 初始化求解器
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	linearSolver->setBlockOrdering( false );
	SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
 
	g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
	globalOptimizer.setAlgorithm( solver );
	// 不要输出调试信息
	globalOptimizer.setVerbose( false );
	
	//向globalOptimizer增加第一个顶点
	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setId( currIndex );
	v->setEstimate( Eigen::Isometry3d::Identity() );//估计为单位矩阵
	v->setFixed( true );//第一个顶点固定，不用优化
	globalOptimizer.addVertex( v );
	
	int lastIndex = currIndex;//上一帧的id
	
	for( currIndex = startIndex + 1; currIndex < endIndex; currIndex++ )
	{
		cout<<"Reading files "<<currIndex<<endl;
		
		FRAME currFrame = readFrame( currIndex, pd );//读取currFrame
		computeKeyPointsAndDesp( currFrame );
		//比较currFrame和lastFrame
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
		
		//cloud = joinPointCloud( cloud, currFrame, T, camera );
		
		//向g2o中增加这个顶点与上一帧联系的边
		//顶点部分，顶点只需设定id即可
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( currIndex );
		v->setEstimate( Eigen::Isometry3d::Identity() );
		globalOptimizer.addVertex(v);
		//边部分
		g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		//连接此边的两个顶点id
		edge->vertices()[0] = globalOptimizer.vertex( lastIndex );
		edge->vertices()[1] = globalOptimizer.vertex( currIndex );
		//信息矩阵
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6, 6 >::Identity();//pose为6D，信息矩阵为6*6，位置和角度估计精度均为0.1且互相独立，协方差矩阵对角为0.01，信息矩阵为100
		information(0,0) = information(1,1) = information(2,2) = 100;
		information(3,3) = information(4,4) = information(5,5) = 100;
		
		edge->setInformation( information );
		edge->setMeasurement( T );//边的估计是pnp求解结果
		
		globalOptimizer.addEdge(edge);//将此边加入图中
		
		lastFrame = currFrame;
		lastIndex = currIndex;
		
	}
	
	//优化所有边
	cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
	globalOptimizer.save("/home/wendy/slam_study/data/result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);//指定优化步数
	globalOptimizer.save("/home/wendy/slam_study/data/result_after.g2o");
	cout<<"Optimization done."<<endl;
	
	globalOptimizer.clear();
	
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
