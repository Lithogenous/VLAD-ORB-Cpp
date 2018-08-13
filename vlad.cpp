/**
* Date:  2016
* Author: Rafael Muñoz Salinas
* Description: demo application of DBoW3
* License: see the LICENSE.txt file
*/

#include <iostream>
#include <vector>
#include<io.h>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

#define COLUMNOFCODEBOOK 32 //码本大小
#define DESSIZE 32  //特征维数(orb是32维，sift是128维)



using namespace DBoW3;
using namespace std;

struct dist {
	int dis;
	int site;
};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
	cout << endl << "Press enter to continue" << endl;
	getchar();
}



//dbow3 demo.cpp函数 用于提取特征，输入图像数据集地址+所需特征，返回一个存储各图像特征的向量
vector< cv::Mat  >  loadFeatures(std::vector<string> path_to_images, string descriptor = "") throw (std::exception) {
	//select detector
	cv::Ptr<cv::Feature2D> fdetector;
	if (descriptor == "orb")        fdetector = cv::ORB::create();
	else if (descriptor == "brisk") fdetector = cv::BRISK::create();
#ifdef OPENCV_VERSION_3
	else if (descriptor == "akaze") fdetector = cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
	else if (descriptor == "surf")  fdetector = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

	else throw std::runtime_error("Invalid descriptor");
	assert(!descriptor.empty());
	vector<cv::Mat>    features;


	cout << "Extracting   features..." << endl;
	for (size_t i = 0; i < path_to_images.size(); ++i)
	{
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		cout << "reading image: " << path_to_images[i] << endl;
		cv::Mat image = cv::imread(path_to_images[i], 0);
		if (image.empty())throw std::runtime_error("Could not open image" + path_to_images[i]);
		cout << "extracting features" << endl;
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		features.push_back(descriptors);
		cout << "done detecting features" << endl;
	}
	return features;
}

// ----------------------------------------------------------------------------

//dbow3 demo.cpp自带函数，修改后仅用于创建码本
void testVocCreation(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook)
{
	// branching factor and depth levels
	const int k = COLUMNOFCODEBOOK;
	const int L = 1;
	const WeightingType weight = TF_IDF;
	const ScoringType score = L1_NORM;

	DBoW3::Vocabulary voc(k, L, weight, score);  //初始化码本

	cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
	voc.create(features);   //生成码本
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
		<< voc << endl << endl;

	codebook = voc;
	
	

	// save the vocabulary to disk
	cout << endl << "Saving vocabulary..." << endl;
	voc.save("small_voc.yml.gz");
	cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------



void getFiles(string path, vector<string>& files)
{
	/*
	*用于读取文件夹下所有文件地址
	*@param path 顶层文件夹地址
	*@return files 返回文件夹下所有文件地址
	*/
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


double euclidean_distance(cv::Mat baseImg, cv::Mat targetImg)
{
	/*
	*计算两个向量的欧氏距离
	*@param baseImg 一个向量
	*@param targetImg 一个向量
	*@return 两个向量的欧氏距离
	*/
	double sumDescriptor = 0;
	for (int i = 0; i < baseImg.cols; i++)
	{
		double numBase = abs(baseImg.at<float>(0, i));
		double numTarget = abs(targetImg.at<float>(0, i));
		sumDescriptor += pow(numBase - numTarget, 2);
	}
	double simility = sqrt(sumDescriptor);
	return simility;
}

int hammingDistance(cv::Mat baseImg, cv::Mat targetImg)
{
	/*
	*计算两个向量的汉明距离
	*@param baseImg 一个向量
	*@param targetImg 一个向量
	*@return 两个向量的汉明距离
	*/
	int sumDescriptor = 0;
	for (int i = 0; i < baseImg.cols; i++)
	{
		uint8_t numBase = baseImg.at<uint8_t>(0, i);
		uint8_t numTarget = targetImg.at<uint8_t>(0, i);
		//sumDescriptor += pow(numBase - numTarget, 2);
		if (numBase != numTarget) sumDescriptor++;
	}

	return sumDescriptor;
}


void calVLAD(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook, vector<cv::Mat> &vladBase) 
{
	/*
	*使用直接做差的方式计算残差
	*@param features 数据集中所有图片的所有特征
	*@param codebook 生成好的码本
	*@param vladBase 用于返回得到的所有图的vlad向量
	*/
	WordId wi;
	for (int i = 0; i < features.size(); ++i) {
		//遍历每张图片的特征


		//初始化每张图片的vlad矩阵
		vector< cv::Mat > vladMatrix;
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat Z = cv::Mat::zeros(1, DESSIZE, CV_32FC1);
			vladMatrix.push_back(Z);
		}

		for (int j = 0; j < features[i].rows; ++j) {
			//遍历一张图片的所有特征

			cv::Mat A = features[i](cv::Rect(0, j, DESSIZE, 1));
			wi = codebook.transform(A);   //寻找一个特征向量最近的聚类中心的id
			//cout << wi << " ";
			cv::Mat central = codebook.getWord(wi);  //取最近的聚类中心向量
			cv::Mat tmpA, tmpCen;

			//将该特征向量和找到的最近聚类中心向量转换为float型用于后续计算和归一化
			A.convertTo(tmpA, CV_32FC1);
			central.convertTo(tmpCen, CV_32FC1);

			vladMatrix[wi] += (tmpA - tmpCen);
		}

		cv::Mat vlad = cv::Mat::zeros(1, 0, CV_32FC1);
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::hconcat(vladMatrix[j], vlad, vlad);  //将vlad矩阵展开成向量
		}
		//cout << vlad << endl;

		cv::Mat vladNorm;
		vlad.copyTo(vladNorm);

		cv::normalize(vlad, vladNorm, 1, 0, cv::NORM_L2);  //对得到的vlad向量进行归一化

		vladBase.push_back(vladNorm);

		//cout << vladNorm << endl;
	}


}


/*

void calVLADHamming(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook, vector<cv::Mat> &vladBase)
{
	WordId wi;
	for (int i = 0; i < features.size(); ++i) {
		vector< cv::Mat > vladBitMatrix;
		vector <cv::Mat > vladMatrix;
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat Z = cv::Mat::zeros(1, DESSIZE * sizeof(uint8_t) * 8, CV_32FC1);
			cv::Mat X = cv::Mat::zeros(1, DESSIZE, CV_8U);
			vladBitMatrix.push_back(Z);
			vladMatrix.push_back(X);
		}

		int sum = 0;
		for (int j = 0; j < features[i].rows; ++j) {
			cv::Mat A = features[i](cv::Rect(0, j, DESSIZE, 1));
			wi = codebook.transform(A);
			//cout << wi << " ";
			cv::Mat central = codebook.getWord(wi);
			//cv::Mat tmpA, tmpCen;
			//A.convertTo(tmpA, CV_32FC1);
			//central.convertTo(tmpCen, CV_32FC1);

			//vladMatrix[wi] += (tmpA - tmpCen);

			int bitCur = 0;
			for (int k = 0; k < A.cols; ++k) {
				uint8_t a = A.at<uint8_t>(0, k);
				uint8_t cen = central.at<uint8_t>(0, k);
				uint8_t tmp = a ^ cen;
				uint8_t mask = 0x01;
				for (int l = 0; l < 8; ++l) {
					if (tmp & mask) {
						vladBitMatrix[wi].at<float>(0, bitCur)++;
						bitCur++;
						sum++;
					}
					else {
						bitCur++;
					}
					mask = mask << 1;
				}

			
			}
			

		}
		float thr = sum / (COLUMNOFCODEBOOK * DESSIZE * 8.0f);

		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			int bitCur = 0;
			for (int k = 0; k < vladMatrix[j].cols; ++k) {
				uint8_t mask = 0x01;
				for (int l = 0; l < 8; l++) {
					if (vladBitMatrix[j].at<float>(0, bitCur) > thr) {
						vladMatrix[j].at<uint8_t>(0, k) |= mask;

					}
					mask = mask << 1;
					bitCur++;
				}
			}
		}

		cv::Mat vladMat32;
		cv::Mat vlad = cv::Mat::zeros(1, 0, CV_32FC1);
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat vladMat32;
			vladMatrix[j].convertTo(vladMat32, CV_32FC1);
			cv::hconcat(vladMat32, vlad, vlad);
		}
		//cout << vlad << endl;

		cv::Mat vladNorm;
		vlad.copyTo(vladNorm);

		cv::normalize(vlad, vladNorm, 1, 0, cv::NORM_L2);

		vladBase.push_back(vladNorm);

		//cout << vladNorm << endl;
	}


}

*/



void calVLADHamming(const vector<cv::Mat> &features, DBoW3::Vocabulary &codebook, vector<cv::Mat> &vladBase)
{
	/*
	*以bit为单位使用汉明距离计算残差
	*@param features 数据集中所有图片的所有特征
	*@param codebook 生成好的码本
	*@param vladBase 用于返回得到的所有图的vlad向量
	*/

	WordId wi;
	for (int i = 0; i < features.size(); ++i) {
		//遍历每张图片

		//初始化vlad矩阵和vlad位矩阵分别用于返回最终的vlad矩阵和保存每bit的统计结果
		vector< cv::Mat > vladBitMatrix;
		vector <cv::Mat > vladMatrix;
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::Mat Z = cv::Mat::zeros(1, DESSIZE * sizeof(uint8_t) * 8, CV_32FC1);
			cv::Mat X = cv::Mat::zeros(1, DESSIZE * sizeof(uint8_t) * 8, CV_8U);
			vladBitMatrix.push_back(Z);
			vladMatrix.push_back(X);
		}

		int sum = 0;
		for (int j = 0; j < features[i].rows; ++j) {
			//遍历每张图片的所有特征


			cv::Mat A = features[i](cv::Rect(0, j, DESSIZE, 1));
			wi = codebook.transform(A);
			cv::Mat central = codebook.getWord(wi);
		
			
			int bitCur = 0;
			for (int k = 0; k < A.cols; ++k) {
				//取一张图片的一个特征向量和最近的聚类中心向量
				uint8_t a = A.at<uint8_t>(0, k); 
				uint8_t cen = central.at<uint8_t>(0, k);
				uint8_t tmp = a ^ cen; //遍历向量中每个元素并进行异或操作用于统计不同位数的个数
				uint8_t mask = 0x01;
				for (int l = 0; l < 8; ++l) {
					if (tmp & mask) {
						vladBitMatrix[wi].at<float>(0, bitCur)++;  //如果特征向量与聚类中心不同，则vlad位矩阵该位置+1
						bitCur++;
						sum++;  //统计一张图片中所有不同位数的个数
					}
					else {
						bitCur++;
					}
					mask = mask << 1;
				}


			}


		}
		float thr = sum / (COLUMNOFCODEBOOK * DESSIZE * 8.0f);  //用总的不同位数的个数除以所有位数计算一个阈值thr

		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			int bitCur = 0;
			for (int k = 0; k < DESSIZE; ++k) {
				//uint8_t mask = 0x01;
				for (int l = 0; l < 8; l++) {
					if (vladBitMatrix[j].at<uint8_t>(0, bitCur) > thr) {
						//如果vlad位矩阵中的不同位数累积结果大于阈值thr，则改位记为1，否则记为0
						vladMatrix[j].at<uint8_t>(0, bitCur) = 1;

					}
					//mask = mask << 1;
					bitCur++;
				}
			}
		}

		//将得到的vlad矩阵扩展成为向量
		cv::Mat vlad = cv::Mat::zeros(1, 0, CV_8U);
		for (int j = 0; j < COLUMNOFCODEBOOK; ++j) {
			cv::hconcat(vladMatrix[j], vlad, vlad);
		}
		//cout << vlad << endl;

		vladBase.push_back(vlad);

		//cout << vladNorm << endl;
	}


}

bool vecCmp(const dist &a, const dist &b) {
	     return a.dis < b.dis;
	
}



// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{

	try {
		// 读取图片集和查询集所有图片地址
		char* filePath = "D:\\datas\\jpg1\\jpg";
		char* retFilePath = "D:\\datas\\query1";
		vector<string> files;
		vector<string> retFiles;
		getFiles(filePath, files);
		getFiles(retFilePath, retFiles);

		vector<int> fileList;
		//将图片集每张图片的编号存入fileList向量中
		for (int i = 0; i < files.size(); ++i) {
			char secondSite = files[i][19] - '0';
			char thirdSite = files[i][20] - '0';
			char fourthSite = files[i][21] - '0';
			fileList.push_back(100 * secondSite + 10 * thirdSite + fourthSite);
		}


		//提取特征
		string descriptor = "orb";
		vector< cv::Mat   >   features = loadFeatures(files, descriptor);
		vector< cv::Mat   >   retFeatures = loadFeatures(retFiles, descriptor);

		//建立码本
		DBoW3::Vocabulary voc;
		testVocCreation(features, voc);
		
		//log
		cout << "features of data set:" << endl;
		for (int i = 0; i < features.size(); i++) {
			cout << i << "th images' features:" << endl;
			cout << features[i] << endl;
		}
		cout << "------------------------------------------" << endl;

		cout << "features of query set:" << endl;

		for (int i = 0; i < retFeatures.size(); i++) {
			cout << i << "th images' features:" << endl;
			cout << retFeatures[i] << endl;
		}
		cout << "*******************************************" << endl;

		cout << "output the codebook:" << endl;

		for (WordId wi = 0; wi < COLUMNOFCODEBOOK; wi++) {
			cv::Mat cen = voc.getWord(wi);
			cout << cen << endl;
		}


		//计算图片集和查询集每张图片的vlad向量
		vector<cv::Mat> vladBase;
		vector <cv::Mat> retVladBase;
		calVLADHamming(features, voc, vladBase);
		calVLADHamming(retFeatures, voc, retVladBase);

		//log
		cout << "vlad matrix of data set:" << endl;
		for (int i = 0; i < vladBase.size(); i++) {
			cout << i << "th vlad mat:" << endl;
			cout << vladBase[i] << endl;
		}
		cout << "-------------------------------------------" << endl;

		cout << "vlad matrix of query set:" << endl;
		for (int i = 0; i < retVladBase.size(); i++) {
			cout << i << "th vlad mat:" << endl;
			cout << retVladBase[i] << endl;
		}
		cout << "*******************************************" << endl;


		//测试、评价
		int cnt = 0;
		for (int i = 0; i < retVladBase.size(); ++i) {
			vector<struct dist> disVec; //该向量用于对测试结果进行存储和排序
			for (int j = 0; j < vladBase.size(); ++j) {
				
				int dis = hammingDistance(vladBase[j], retVladBase[i]); //计算图片集中每张图片与待查询图片的vlad向量的汉明距离
				dist tmp;
				tmp.dis = dis;
				tmp.site = j;
				disVec.push_back(tmp);
			}

			//对结果进行排序、输出结果
			sort(disVec.begin(), disVec.end(), vecCmp);
			for (int i = 0; i < disVec.size(); i++) {
				cout << i << "th sim is " << fileList[disVec[i].site] << " and distance is " << disVec[i].dis << endl;
			}

			//统计查询正确的个数
			if (i == fileList[disVec[1].site]) {
				cout << "find the most similar" << endl;
				cnt++;
			}
		}
		cout << "Top.1:\n" << cnt / 500.0 << endl;

		


		//testDatabase(features);

	}
	catch (std::exception &ex) {
		cerr << ex.what() << endl;
	}

	return 0;
}
