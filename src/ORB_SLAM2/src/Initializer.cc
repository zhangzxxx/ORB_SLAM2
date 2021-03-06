/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

/**
 * @brief Construct a new Initializer:: Initializer object
 *          根据参考帧构造初始化器
 * 
 * @param[in] ReferenceFrame    参考帧
 * @param[in] sigma             测量误差
 * @param[in] iterations        RANSAC迭代次数
 */
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    // 从参考帧中获取相机内参矩阵
    mK = ReferenceFrame.mK.clone();

    // 从参考帧中获取去畸变后的特征点
    mvKeys1 = ReferenceFrame.mvKeysUn;

    // 获取估计误差
    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

/**
 * @brief 计算基础矩阵和单应矩阵，选取最佳的来恢复出最开始两帧之间的相对位姿，并进行三角化得到初始地图点
 *  1. 重新记录特征点对的匹配关系
 *  2. 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
 *  3. 计算 fundamental homography矩阵，为了加速分别开了线程
 *  4. 计算得分比例来判断选取哪个模型来求位姿R, t
 * 
 * @param[in] CurrentFrame              当前帧，也就是SLAM意义上的第二帧
 * @param[in] vMatches12                当前帧2和参考帧1图像中特征点的匹配关系
 *                                      vMatches12[i]：其中i表示参考帧1中特征点的索引，vMatches12[i]表示当前帧中匹配的特征点索引。没有匹配关系的话值为-1
 * @param[in & out] R21                 相机从参考帧到当前帧的旋转     
 * @param[in & out] t21                 相机从参考帧到当前帧的平移
 * @param[in & out] vP3D                三角化测量之后的三维地图点
 * @param[in & out] vbTriangulated      标记三角化是否有效，有效为true
 * @return true                         该帧可以成功初始化， 返回true,否则返回false
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    // 获取当前帧去畸变后的特征点
    mvKeys2 = CurrentFrame.mvKeysUn;

    // mvMatches12记录匹配上的特征点对，记录的是帧2在帧1上的匹配索引
    mvMatches12.clear();
    // 预分配空间，大小和特征点数目一致
    mvMatches12.reserve(mvKeys2.size());

    // 记录参考帧1中的每个特征点是否有匹配的特征点，这个成员变量后面没有用到，后面只关心匹配上的特征点
    mvbMatched1.resize(mvKeys1.size());

    // Step 1 重新记录特征点对的匹配关系，存储在mvMatches12中，是否有匹配存储在mvbMatched1中
    // 将vMatches12 (有冗余) 转化为 mvMatches12 (只记录了匹配关系)
    // ? 有冗余没看懂 不是只记录了匹配关系吗
    // 回答：vMatches12数组记录了参考帧所有特征点索引，没有匹配上的值为-1，匹配上的值为当前帧中的匹配特征点索引
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        // 如果没有匹配关系的话, vMatches12[i] = -1
        if(vMatches12[i]>=0)
        {
            // i是参考帧特征点索引， vMatches12[i]是匹配上的帧中的特征点索引
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    // 有匹配的特征点对数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    // 新建一个容器存储特征点索引，并预分配空间
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    // 在RANSNC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引，故这里起的名字叫做可用的索引
    vector<size_t> vAvailableIndices;

    // 初始化所有特征点对的索引，索引值从0 ～ N-1
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计F矩阵和H矩阵
    // 共选择 mMaxIterations组（默认200）
    // mvSets用于保存每次迭代时所使用的向量
    mvSets = vector< vector<size_t> >(mMaxIterations,           // 最大的RANSNC迭代次数
                                      vector<size_t>(8,0));     // 第二维元素的初始值，其中包括8维

    // 设置随机数种子
    DUtils::Random::SeedRandOnce(0);

    // 开始每一次迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // 迭代开始时，所有的点都是可用的
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        // 选择最小的数据样本集，使用八点法求
        for(size_t j=0; j<8; j++)
        {
            // 随机产生一对点的id 范围0 ~ N-1
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            // idx表示哪一个索引对应的特征点对被选中
            int idx = vAvailableIndices[randi];

            // 将本次迭代中这个选中的第j个特征点对的索引添加到 mvSets 中
            mvSets[it][j] = idx;

            // 由于这个点在本次迭代中已经使用了，为了避免下一次被选中，将这个点用”可选列表“中最后一个点覆盖，并删除最后一个点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    // 分别开线程计算H矩阵和F矩阵
    // 这两个变量用于标记在H和F计算中哪些特征点被认为是 Inlier
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    // 计算出来的H矩阵和F矩阵的RANSNC评分，这里起始采用重投影来计算的
    float SH, SF;
    // 经过RANSNC算法计算出来的H和F矩阵
    cv::Mat H, F;

    // 构造线程来计算H和F矩阵及其得分
    // thread方法比较特殊，在传递引用的时候，外层需要用ref来进行引用传递，否则就是浅拷贝
    thread threadH(&Initializer::FindHomography,        // 该线程的主函数
                    this,                               // 由于主函数为类的成员函数，所以第一个参数就应该是当前对象的this指针
                    ref(vbMatchesInliersH),             // 输出，特征点对应的Inlier标记
                    ref(SH),                            // 输出，计算的单应矩阵的RANSNC评分
                    ref(H));                            // 输出，计算的单应矩阵
    // 计算基础矩阵F
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    // 等待两个线程结束
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // Step 4 计算得分比例来判断选取哪个模型来恢复R t
    // 计算H和F矩阵得分比率，不是简单比较大小，而是看评分的占比
    float RH = SH/(SH+SF);          // RH = Ratio of Homography

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 这里更倾向于用H矩阵恢复位姿。如果单应矩阵的评分占比达到了0.4以上，则从单应矩阵恢复运动，否则从基础矩阵恢复运动
    if(RH>0.40)
        // 更倾向于平面，此时从单应矩阵恢复位姿，函数ReconstructH返回bool型结果
        return ReconstructH(vbMatchesInliersH,      // 输入，匹配成功的特征点对 Inlier标记
                            H,                      // 输入，前面RANSNC计算的单应矩阵
                            mK,                     // 输入，相机内参矩阵
                            R21, t21,               // 输出 计算出来的相机存从参考帧1到当前帧2的旋转和平移
                            vP3D,                   // 特征点对经过三角测量之后的空间坐标，也就是地图点
                            vbTriangulated,         // 特征点对是否三角化成功
                            1.0,                    // 这个对应的形参为 minParallax，即认为某对特征点的三角化测量中，认为其测量有效时需要满足的最小视差角
                                                    // 如果视差角过小容易引起非常大的观测误差，单位是角度
                            50);                    // 为了进行运动恢复，需要的最少的三角化测量成功的点个数
    else //if(pF_HF>0.6)
        // 更倾向于非平面，从基础矩阵恢复位姿
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    // 一般程序应该不会执行到这里，除非程序起飞！！！
    return false;
}

/**
 * @brief 计算单应矩阵，假设场景为平面情况下通过前两帧求取 H矩阵，并得到该模型的评分
 *  算法原理
 *  1. 将当前帧和参考帧中的特征点坐标进行归一化
 *  2. 选择8个归一化后的点对进行迭代
 *  3. 八点法计算单应矩阵
 *  4. 利用重投影误差为档次的RANSNC评分
 *  5. 更新具有最优评分的单应矩阵计算结果，并保存所对应的特征点对的内点标记
 * 
 * @param[in & out] vbMatchesInliers  标记是否是外点
 * @param[in & out] score             计算单应矩阵得分
 * @param[in & out] H21               计算的单应矩阵
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    // 匹配的特征点对总数
    const int N = mvMatches12.size();

    // Normalize coordinates
    // Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
    // 具体就是将　mvKeys1和mvKeys2归一化到均值为０，一阶绝对矩为１，归一化矩阵分别为T1 和 T2
    // 一阶绝对矩就是随机变量到取值的中心的绝对值的平均值
    // 归一化矩阵就是把上述归一化操作用矩阵来表示。这样特征点坐标乘以归一化矩阵可以得到归一化后的坐标

    // 归一化后的参考帧1和当前帧2的特征点坐标
    vector<cv::Point2f> vPn1, vPn2;
    // 记录各自归一化矩阵
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    // 求当前帧特征点归一化矩阵的逆，用来辅助进行原始点恢复
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    // 记录最佳评分
    score = 0.0;
    // 取得历史最佳评分时，特征点对应的Inlier标记
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    // 某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    // 计算出来的单应矩阵及其逆矩阵
    cv::Mat H21i, H12i;

    // 每次RANSAC记录inlier和当前得分
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 开始进行每次RANSAC迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // Step 2 选择八个归一化之后的点对进行迭代
        for(size_t j=0; j<8; j++)
        {
            // 从mvSets中获取当前迭代的某个特征点对的索引信息
            int idx = mvSets[it][j];

            // vPn1i 和 vPn2i为匹配的特征点对的归一化后的坐标
            // 首先根据这个特征点对的索引信息分别找到两个特征点在各自图像特征点向量中的索引，然后读取其归一化之后的特征点坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];        // first存储的参考帧1中的特征点索引
            vPn2i[j] = vPn2[mvMatches12[idx].second];       // second存储的当前帧2中的特征点索引
        }

        // Step 3 八点法计算单应矩阵
        // ? 关于八点法计算单应矩阵 为什么要进行归一化
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);

        // 单应矩阵原理： p2 = H21*p1, 其中p1,p2为归一化前的特征点坐标
        // 特征点归一化 p1‘ = T1*p1 , p2' = T2*p2,  归一化后的单应矩阵 p2' = Hn * p1'  ---> T2 * p2 = Hn * T1 * p1  ---> p2 = T2_inv * Hn * T1 * p1
        // H21 = T2_inv * Hn * T1
        H21i = T2inv*Hn*T1;
        // 然后计算单应矩阵的逆
        H12i = H21i.inv();

        // Step 4 利用重投影误差为当次RANSAC的结果评分
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        // Step 5 更新具有更高评分的单应矩阵计算结果，并保存所对应的特征点对的内点标记
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


/**
 * @brief 计算基础矩阵，假设场景为非平面情况下通过前两帧求取基础矩阵，得到该模型的评分
 *  1. 将当前帧和参考帧中的特征点坐标进行归一化
 *  2. 选择8个归一化之后的点进行迭代
 *  3. 八点法计算基础矩阵
 *  4. 利用重投影误差为当次RANSAC的结果计算评分
 *  5. 更新具有最优评分的基础矩阵计算结果，并且保存所对应的特征点对的内点标记
 * 
 * @param[in] vbMatchesInliers  标记是否是内点
 * @param[in] score             计算基础矩阵得分
 * @param[in] F21               从img1到img2的基础矩阵
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // 计算基础矩阵

    // Number of putative matches
    // 匹配的特征点数
    // const int N = vbMatchesInliers.size();      // ? 此处源代码出错了吧，只计算个数的话好像不影响
    // 更改后的代码
    const int N  =  mvMatches12.size();

    // Step 1 归一化参考帧和当前帧的特征点坐标，主要是平移和尺度变换
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    // 归一化矩阵
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    // note 此处取归一化矩阵T2的转置，因为基础矩阵和单应矩阵的定义不同
    cv::Mat T2t = T2.t();

    // Best Results variables
    // 最优结果
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    // 某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    // 该次迭代中的基础矩阵
    cv::Mat F21i;

    // 每次RANSAC记录的inlier和当前得分
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 进行RANSAC迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // Step 2 选择8个归一化之后的点对进行迭代
        for(int j=0; j<8; j++)
        {
            // 获取该点对索引
            int idx = mvSets[it][j];
            // vPn1i vPn2i为匹配特征点归一化后的坐标，first为参考帧特征点索引，second为当前帧匹配特征点索引
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // Step 3 八点法计算基础矩阵
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        // 基础矩阵 p2^t * F * p1 = 0,  归一化的点 p2' = T2 * p2, p1' = T1 * p1
        // 八点法计算的归一化坐标的基础矩阵， p2'^t * Fn * p1' = 0  ---> (T2*p2)^t * Fn * (T1 * p1) = 0  ---> p2^t * T2^T * Fn * T1 * p1 = 0
        // ---> F = T2^t * Fn * T1
        F21i = T2t*Fn*T1;

        // Step 4 利用重投影误差为当次RANSAC的结果评分
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        // Step 5 更新具有最优评分的基础矩阵计算结果，并保存所对应的特征点对的内点标记
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


/**
 * @brief 用DLT方法求解单应矩阵H 
 * 由于一个点对可以提供两个约束方程，所以四个点就可以求解单应矩阵。
 * 不过因为基础矩阵需要八点法进行求解，所以统一用八点法求解
 * 
 * @param[in] vP1       参考帧中归一化后的特征点坐标
 * @param[in] vP2       当前帧中归一化后的特征点坐标
 * @return cv::Mat      计算的单应矩阵H
 */
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    /**
     * 基本原理
     *      |x'|        | h1 h2 h3 | |x|
     *      |y'| =  a * | h4 h5 h6 | |y|    即 P' = a*H*P  其中a为一个尺度因子
     *      |1 |        | h7 h8 h9 | |1|
     * 使用DLT求解该模型 两边同时用P'叉乘， ---> 0 = aP'× H*P  ---> Ah = 0
     */

    // 获取参与计算的特征点数目
    const int N = vP1.size();

    // 构造用于计算的矩阵A 每一个点的数据对应两行，9列
    cv::Mat A(2*N,9,CV_32F);

    // 构造矩阵A, 将每个特征点添加到矩阵A中
    // A = [0, 0, 0, -x, -y, -1, xy', yy', y';  -x, -y, -1, 0, 0, 0, xx', yx', x'];
    // h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]
    for(int i=0; i<N; i++)
    {
        // 获取特征点对应的像素坐标
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        // 生成该点第一行
        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        // 生成该点第二行
        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    // 定义输出变量，u是左边的正交矩阵U， w是中间的奇异矩阵， vt是右边的正交矩阵V的转置
    cv::Mat u,w,vt;

    // 使用openCV提供的函数进行奇异值分解
    cv::SVDecomp(A,w,u,vt,
                cv::SVD::MODIFY_A | cv::SVD::FULL_UV);  // 输入， MODIFY_A是指允许计算函数可以修改待分解的矩阵，可以加快计算速度、节省内存 FULL_UV把U和VT补充成单位正交矩阵

    // 返回最小奇异值所对应的右奇异向量
    // 是奇异矩阵V的最后一列，即转置后VT的最后一行
    return vt.row(8).reshape(0, 3);     // 转置后的通道数，设置成0表示与前面相同，转换后的行数为3行
}

/**
 * @brief 根据特征点匹配求基础矩阵 （归一化之后的八点法）
 * note 基础矩阵有秩为2的约束，所以需要两次SVD分解
 * 
 * @param[in] vP1   参考帧特征点归一化后的坐标
 * @param[in] vP2   当前帧特征点归一化后的坐标
 * @return cv::Mat  计算结果-基础矩阵
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    /**
     * p2^t * F * p1 = 0, 整理得 Af = 0
     * A = [x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]  f = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
     * 通过SVD求解 Af = 0, f=0不是想要的 添加约束 ||f|| = 1     A=UDV^T  min||Af||=min||UDV^T f||=min||DV^T f|| 且||f||=||V^T f||
     * 令 y = V^T f,则问题变成：min||Dy|| s.t, ||y||=1 (因为V为正交矩阵)
     * 由于D是一个对角矩阵，对角元素按降序排列，因此最优解在y=(0, 0, ... , 1)^T 时取得，又 x = Vy, 所以最优解就是Ａ最小奇异奇异值对应的Ｖ的列向量 
     */


    // 获取参与计算的特征点数目
    const int N = vP1.size();

    // 初始化Ａ矩阵　Ｎ×９维
    cv::Mat A(N,9,CV_32F);

    // 构造矩阵Ａ，将每个特征点对添加到矩阵Ａ中
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    // 对矩阵Ａ奇异值分解的结果变量
    cv::Mat u,w,vt;

    // 调用openCV中函数进行SVD分解
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 将v最后一列向量（最小奇异值对应的解）转换为矩阵形式
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // 基础矩阵秩为2，而我们不敢保证计算得到的结果的秩为2，所以需要通过第二次奇异值分解，来强制是秩为2
    // 对初步得到的基础矩阵进行第二次特征值分解
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 秩2的约束，强制将第3个奇异值置为0
    w.at<float>(2)=0;

    // 重新组合满足秩约束的基础矩阵，作为最终计算结果返回
    return  u*cv::Mat::diag(w)*vt;
}

/**
 * @brief 对给定的单应矩阵打分，需要使用卡方检验知识
 * 
 * @param[in] H21                   从参考帧到当前帧的单应矩阵
 * @param[in] H12                   从当前帧到参考帧的单应矩阵
 * @param[in] vbMatchesInliers      匹配好的特征点对的inlier标记
 * @param[in] sigma                 方差 默认为1
 * @return float                    返回得分
 */
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
    // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高
    // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
    // 算法目标： 检查单应变换矩阵
    // 检查方式：通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差

    // 算法流程
    // input: 单应性矩阵 H21, H12, 匹配点集 mvKeys1
    //    do:
    //        for p1(i), p2(i) in mvKeys:
    //           error_i1 = ||p2(i) - H21 * p1(i)||2
    //           error_i2 = ||p1(i) - H12 * p2(i)||2
    //           
    //           w1 = 1 / sigma / sigma
    //           w2 = 1 / sigma / sigma
    // 
    //           if error1 < th
    //              score +=   th - error_i1 * w1
    //           if error2 < th
    //              score +=   th - error_i2 * w2
    // 
    // note 该算法此步弄反了 误差大于阈值，应该是离群点，而不是内点
    //           if error_1i > th or error_2i > th
    //              p1(i), p2(i) are inner points
    //              vbMatchesInliers(i) = true
    //           else 
    //              p1(i), p2(i) are outliers
    //              vbMatchesInliers(i) = false
    //           end
    //        end
    //   output: score, inliers

    // 特征点匹配个数
    const int N = mvMatches12.size();

    // Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    // 获取从当前帧到参考帧的单应矩阵的各个元素
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    // 给特征点对的inlier标记预分配空间
    vbMatchesInliers.resize(N);

    // 初始化得分
    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的误差）
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float th = 5.991;

    // 信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算加权重投影误差
    // H21 表示img1 到 img2的变换矩阵 H12则相反
    for(int i=0; i<N; i++)
    {
        // 一开始都默认为 inlier
        bool bIn = true;

        // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Step 2.2 计算 img2到 img1的重投影误差
        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u2|   |h11 h12 h13||u1|   |u1in2|
        // |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
        // |1 |   |h31 h32 h33||1 |   |  1  |
		// 计算投影归一化坐标
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        // 不同层图像重投影误差利用协方差进行加权，起到归一化作用，使得不同层级的重投影误差阈值一样
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // Step 2.3 利用阈值标记离群点，内点的话就累加得分
        if(chiSquare1>th)
            bIn = false;
        else
            // 误差越大，得分越低
            score += th - chiSquare1;

        // 计算从img1到img2的重投影误差
        // Reprojection error in second image
        // x1in2 = H21*x1
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        // 计算得分
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        // Step 2.4 如果双向投影的误差均满足要求，则说明是Inlier point
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 对给定的基础矩阵计算RANSAC得分
 * 
 * @param[in] F21               当前帧和参考帧之间的基础矩阵
 * @param[in] vbMatchesInliers  匹配的特征点是否属于inlier标记
 * @param[in] sigma             方差，默认为1
 * @return float                计算得分
 */
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
    // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高
    // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
    // 算法目标：检查基础矩阵
    // 检查方式：利用对极几何原理 p2^T * F * p1 = 0
    // 假设：三维空间中的点 P 在 img1 和 img2 两图像上的投影分别为 p1 和 p2（两个为同名点）
    //   则：p2 一定存在于极线 l2 上，即 p2*l2 = 0. 而l2 = F*p1 = (a, b, c)^T
    //      所以，这里的误差项 e 为 p2 到 极线 l2 的距离，如果在直线上，则 e = 0
    //      根据点到直线的距离公式：d = (ax + by + c) / sqrt(a * a + b * b)
    //      所以，e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    

    // 获取匹配的特征点的总对数
    const int N = mvMatches12.size();

    // Step 1 提取基础矩阵中的元素数据
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    // inlier 数组预分配空间
    vbMatchesInliers.resize(N);

    // 评分初始值
    float score = 0;

    // 基于卡方检验计算出的阈值
    // 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
    // ? 自由度为啥为1
    const float th = 3.841;
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float thScore = 5.991;

    // 信息矩阵，即协方差矩阵的逆
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // Step 2 计算 img1 和 img2在该Ｆ时得分
    for(int i=0; i<N; i++)
    {
        // 默认这对特征点是inlier point
        bool bIn = true;

        // Step 2.1 提取参考帧和当前帧的特征点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // Step 2.2 计算 img1上的点在img2上投影得到的极线 l2: a2*x + b2*y + c = 0
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // Step 2.3 计算误差，点到直线的距离的平方
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        // 带权重误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // Step 2.4 误差大于阈值，说明这个点是 outlier
        // ? 为什么判断阈值和计算得分用自由度不同的阈值， 是为了和CheckHomography 得分统一？
        if(chiSquare1>th)
            bIn = false;
        else
            // 误差越大，得分越低
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        // 计算img2上的点在img1上的重投影误差，F21与F12为转置关系
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        // Step 2.5 保存结果
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 从基础矩阵F中求解位姿R,t 及3D点坐标
 * F分解出E，E有四组解，选择计算的有效三维点（在相机光心前方、投影误差小于阈值、视差角大于阈值）最多的作为最优的解
 * 
 * @param[in] vbMatchesInliers      匹配好的特征点对的Inlier标记
 * @param[in] F21                   从参考帧到当前帧的基础矩阵
 * @param[in] K                     相机内参矩阵
 * @param[in] R21                   待求解的从参考帧到当前帧的相机旋转
 * @param[in] t21                   待求解的从参考帧到当前帧的相机平移
 * @param[in] vP3D                  三角化测量之后的特征点的空间坐标
 * @param[in] vbTriangulated        特征点三角化成功标志
 * @param[in] minParallax           三角化有效的最小视角差
 * @param[in] minTriangulated       最小三角化成功点数
 * @return true                     初始化成功
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    // Step 1 统计有效匹配点个数
    // vbMatchesInliers中存储匹配点对是否有效
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Step 2 根据基础矩阵和相机内参矩阵计算本质矩阵
    // Compute Essential Matrix from Fundamental Matrix
    // F = K^(-T)*t^R*K^(-1)    --->    E = t^R = K^T * F * K
    cv::Mat E21 = K.t()*F21*K;

    // 定义本质矩阵分解结果，形成四组解，分别是 (R1, t), (R1, -t), (R2, t), (R2, -t)
    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 从本质矩阵求解两个R和两个t解，共4个解
    /**
     * 由于两个t解互为相反数，故只求一个就行
     * 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度
     * 因为CreateInitialMapMonocular 函数对3D 点深度会缩放，然后反过来对 t 有改变
     *                              |0 -1  0|
     * Ｅ = U Sigma Vt      let W = |1  0  1|
     *                              |0  0 -1|
     * 得到４个解　Ｅ=[R|t]     R1 = U W Vt     R2 = U Wt Vt    t1 = U3       t2 = -U3
     */
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    // Step 4 分别验证求解的4种R和t 的组合，选出最佳组合
    // 原理：若某一组合使恢复得到的3D点位于相机正前方的数量最多，那么该组合就是最佳组合
    // 实现：根据计算的解组合成四中情况，并依次调用Initializer::CheckRT()进行检验，得到可以三角化测量的点的数目
    // 定义四组解分别在同一匹配点集进行三角化测量之后的特征点空间坐标
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    // 定义四组解对同一匹配点集的有效三角化结果
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    // 定义四组解对应的比较大的特征点对视差角
    float parallax1,parallax2, parallax3, parallax4;

    // Step 4.1 使用同样的匹配点分别检查四组解，记录当前计算的3D点在相机光心前方且投影误差小于阈值的个数，记为有效3D点个数
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    // Step 4.2 选取最大可三角化测量的点的数目
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    // 重置变量，并在后面赋值为最佳R,t
    R21 = cv::Mat();
    t21 = cv::Mat();

    // Step 4.3 确定最小的可以三角化的点数
    // 在0.9倍的内点数 和 指定minTriangulated=50中的最大值，即至少50个    
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    // 统计四组解中重建的有效3D点个数 > 0.7 * maxGood 的解的数目
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // Step 4.4 四个结果中如果没有明显的最优结果，或者没有足够数量的三角化点，则返回失败
    // 条件1：如果四组解能够重建的最多3D点个数小于所要求的最少3D点个数(nMinGood)，则失败
    // 条件2：如果存在两组及以上的解能三角化出 > 0.7*maxGood 的点，说明没有明显最优结果，失败
    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    //  Step 4.5 选择最佳解记录结果
    // 条件1: 有效重建最多的3D点，即maxGood == nGoodx，也即是位于相机前方的3D点个数最多
    // 条件2: 三角化视差角 parallax 必须大于最小视差角 minParallax，角度越大3D点越稳定

    // 看最好的good点是在哪种解下生成的
    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        // 如果该解下的parallax1大于给定的视差角阈值
        if(parallax1>minParallax)
        {
            // 存储3D坐标
            vP3D = vP3D1;
            // 获取特征点向量的三角化测量标记
            vbTriangulated = vbTriangulated1;

            // 存储相机位姿
            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    // 如果有最优解但是不满足视差角阈值要求，同样返回失败
    return false;
}
/**
 * @brief 从H矩阵恢复R,t 和三维点
 * H矩阵分解常见的有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decompisition
 * 本代码使用了前者。参考文献 Motion and structure from motion in a piecewise planar environment. 1988
 * 
 * @param[in] vbMatchesInliers      匹配点对的内点标记
 * @param[in] H21                   从参考帧到当前帧的单应矩阵
 * @param[in] K                     相机内参矩阵
 * @param[in] R21                   计算出的相机旋转
 * @param[in] t21                   计算出的相机平移
 * @param[in] vP3D                  世界坐标下，三角化测量特征点对之后得到的特征点的空间坐标
 * @param[in] vbTriangulated        特征点是否三角化成功标记
 * @param[in] minParallax           对特征点的三角化测量中，认为其测量有效时需要的最小视角差（视角差过小会引起非常大的误差），单位是角度
 * @param[in] minTriangulated       为了进行运动恢复，所需要的最少的三角化测量成功的点个数
 * @return true                     单应矩阵成功计算出位姿和三维点，否则初始化失败
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    /**
     * 目的：通过单应矩阵恢复图像之间的旋转R和平移t
     * 参考：https://www.researchgate.net/publication/243764888_Motion_and_Structure_from_Motion_in_a_Piecewise_Planar_Environment
     * 流程：
     *  1. 根据H矩阵的奇异值 d'=d2 或 d'=-d2 分别计算H矩阵分解的8组解
     *      1.1 讨论d'>0时的四组解
     *      1.2 讨论d'<0时的四组解
     *  2. 对8组解进行验证，并选择产生相机前方最多3D点的解为最优解
     */

    // 统计匹配的特征点对中属于内点（有效点）的个数
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    /**
     * 从H中恢复R，t 参考SLAM十四讲P169-172
     * H = K * (R - t*n/d) * K_inv
     * 其中： K表示相机内参矩阵，R,t表示旋转和平移，n表示平面法向量
     * 令 H = K * A * K_inv, A = k_inv * H * K
     */
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    // 对矩阵A进行SVD分解，U是左奇异矩阵，w是奇异值矩阵，Vt是右奇异矩阵（函数返回的转置）， FULL_UV 全部分解    A = U *w* Vt
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    // 计算变量 s = det(U) * det(V), 因为 det(V) = det(Vt) 转置不改变行列式大小
    float s = cv::determinant(U)*cv::determinant(Vt);

    // 获取矩阵奇异值
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解正常情况下，奇异值应该是正的，且d1 >= d2 >= d3
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    // 定义8中情况下的旋转矩阵、平移向量和空间向量
    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    // Step 1.1 讨论 d' > 0时的四组解
    /**
     * 根据论文 eq.(12)有
     *  x1 = e1 * sqrt((d1*d1-d2*d2)/(d2*d2-d3*d3))
     *  x2 = 0
     *  x3 = e3 * sqrt((d2*d2-d3*D3)/(d1*d1-d2*d2))
     *  令 aux1 = sqrt((d1*d1-d2*d2)/(d2*d2-d3*d3)), aux3 = sqrt((d2*d2-d3*D3)/(d1*d1-d2*d2))
     *  则 x1 = e1*aux1, x3 = e3*aux3
     * 参考 https://zhuanlan.zhihu.com/p/140035703
     */
    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1， 
    // 因为e1,e3=1 or -1， 所以 x1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    /**
     * 根据论文 eq.(13)有
     * sin(theta) = (d1-d3)/d2 * x1*x3 = e1*e3*sqrt((d1*d1-d2*d2)/(d2*d2-d3*d3))/(d1-d3)/d2
     * cos(theta) = (d1*x3*x3-d3*x1*x1)/d2 = (d2*d2+d1*d3)/(d1+d3)/d2
     * 令 aux_stheta = sqrt((d1*d1-d2*d2)/(d2*d2-d3*d3))/(d1-d3)/d2
     * 则 sin(theta) = e1*e3*aux_stheta
     *   
     * 因为 e1 e2 e3 = 1 or -1
     * 所以 sin(theta) = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}
     */
    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    /**
     * 计算旋转矩阵R’
     * 根据不同的e1 e3组合所得出来的四种R t的解
     *       | ctheta      0   -aux_stheta|       | aux1|
     *  Rp = |    0        1       0      |  tp = |  0  |
     *       | aux_stheta  0    ctheta    |       |-aux3|
     * 
     *       | ctheta      0    aux_stheta|       | aux1|
     *  Rp = |    0        1       0      |  tp = |  0  |
     *       |-aux_stheta  0    ctheta    |       | aux3|
     * 
     *       | ctheta      0    aux_stheta|       |-aux1|
     *  Rp = |    0        1       0      |  tp = |  0  |
     *       |-aux_stheta  0    ctheta    |       |-aux3|
     * 
     *       | ctheta      0   -aux_stheta|       |-aux1|
     *  Rp = |    0        1       0      |  tp = |  0  |
     *       | aux_stheta  0    ctheta    |       | aux3|
     */
    // 开始遍历这四种情况中的每一种
    for(int i=0; i<4; i++)
    {
        // 生成Rp，就是eq.(8)中的R'
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        // eq.(8)中计算R     R' = sU_t*R*V
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        // eq.(14) 生成 t'  t' = (d1-d3)*(x1, 0, x3)^T
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度，因为CreateInitialMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
        // eq.(8) 恢复t     t' = U_t * t
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        // 构造平面法向量np={x1, x2, x3}，其中x2=0
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        // eq.(8) 恢复原始法向量    n' = V_t*n
        cv::Mat n = V*np;
        // 保证法向量指向上方
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    // Step 1.2 讨论d' = -d2 < 0 时的四组解
    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    // cos项
    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    // 考虑 e1 e3 = 1 or -1 sin(theta)有两种可能解
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    // 对于每组e1 e3讨论四种可能取值
    for(int i=0; i<4; i++)
    {
        // 计算旋转矩阵Ｒ'
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        // 恢复原来的Ｒ
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        // 计算平移向量t'
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        // 恢复平移向量t
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        // 计算平面法向量np
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        // 恢复平面法向量n
        cv::Mat n = V*np;
        // 保证方向量方向向上
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    // 最佳的good点
    int bestGood = 0;
    // 次佳的good点
    int secondBestGood = 0;
    // 最佳的解的索引，初始值为-1    
    int bestSolutionIdx = -1;
    // 最大的视差角
    float bestParallax = -1;
    // 最佳的解对应的三角化测量的点的结果
    vector<cv::Point3f> bestP3D;
    // 最佳解对应的可以被三角化的点的标记
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // Step 2 对8组解进行验证，并选择产生相机前方最多3D点的解为最优解
    for(size_t i=0; i<8; i++)
    {
        // 第i组解对应的比较大的视差
        float parallaxi;
        // 第i组解对应三角化测量之后的特征点的空间坐标
        vector<cv::Point3f> vP3Di;
        // 特征点是否被三角化的标记
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],                    // 第i组解 R, t
                            mvKeys1,mvKeys2,                // 特征点
                            mvMatches12,vbMatchesInliers,   // 特征匹配关系以及inlier标记
                            K,                              // 相机内参矩阵
                            vP3Di,                          // 存储三角化测量之后的特征点的空间坐标
                            4.0*mSigma2,                    // 三角化过程中允许的最大重投影误差
                            vbTriangulatedi,                // 特征点是否被三角化成功的标志
                            parallaxi);                     // 这组解在三角化测量的时候比较大的视差

        // 更新最优和次优解
        // ? 保留最优和次优解，保留次优解的目的是看最优解是否突出
        if(nGood>bestGood)
        {
            // 如果当前组的解是历史最优，则更新
            secondBestGood = bestGood;
            bestGood = nGood;
            // 更新当前最优解索引
            bestSolutionIdx = i;
            // 更新视差角，3D点坐标，三角化成功标记
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            // 只是大于次优的话，只需更新次优解的点个数
            secondBestGood = nGood;
        }
    }

    // Step 3 选择最优解。需要满足以下4个条件
    // 1. good点数最优解明显大于次优解，取0.75为经验值
    // 2. 视差角大于规定的阈值
    // 3. good点数要大于规定的最小被三角化的点数量
    // 4. good点数要足够多，达到总数的90%以上
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        // 从最佳的解索引访问到R,t
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        // 获得最佳解时，成功三角化的3D点，以后作为初始地图点使用
        vP3D = bestP3D;
        // 获取特征点中成功三角化的标记
        vbTriangulated = bestTriangulated;

        // 返回true，找到了最优的解
        return true;
    }

    // 没找到最优解
    return false;
}

/**
 * @brief 给定投影矩阵P1, P2和图像上的匹配特征点对kp1,kp2 计算3D空间坐标
 * 
 * @param[in] kp1       参考帧中特征点
 * @param[in] kp2       当前帧中特征点
 * @param[in] P1        参考帧相机投影矩阵
 * @param[in] P2        当前帧相机投影矩阵
 * @param[in] x3D       计算的3D空间点坐标
 */
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    // 原理
    // Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
    // x' = P'X  x = PX
    // 它们都属于 x = aPX模型
    //                         |X|
    // |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
    // |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
    // |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
    // 采用DLT的方法：x叉乘PX = 0
    // |yp2 -  p1|     |0|
    // |p0 -  xp2| X = |0|
    // |xp1 - yp0|     |0|
    // 两个点:
    // |yp2   -  p1  |     |0|
    // |p0    -  xp2 | X = |0| ===> AX = 0
    // |y'p2' -  p1' |     |0|
    // |p0'   - x'p2'|     |0|
    // 变成程序中的形式：
    // |xp2  - p0 |     |0|
    // |yp2  - p1 | X = |0| ===> AX = 0
    // |x'p2'- p0'|     |0|
    // |y'p2'- p1'|     |0|
    // 然后就组成了一个四元一次正定方程组，SVD求解，右奇异矩阵的最后一行就是最终的解.

    // 构造参数矩阵矩阵A 4*4
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    // 对A进行SVD分解
    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    // 右奇异矩阵最后一行就是最终的解,更习惯用列向量表示空间点坐标
    x3D = vt.row(3).t();
    // 为了符合其次坐标形式,使最后一行为1
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

/**
 * @brief 归一化特征点到同一尺度，作为后续 normalize DLT的输入
 *  [x', y', 1]' = T*[x, y, 1]'
 * 归一化后x' y'均值为0， 一阶绝对矩为1 sum(abs(x_i' - 0)) = 1 sum(abs(y_i' - 0)) = 1
 *      为什么要归一化？
 * 在相似变换之后（点在不同的坐标系下），它们的单应矩阵是不相同的。如果图像存在噪声，使得点的坐标发生了变化，那么它的单应矩阵也会发生变化
 * 采取的办法是将点的坐标放到同一坐标系下，并将缩放尺度也进行统一，对同一幅图像的坐标进行相同的变换，不同的图像进行不同的变换
 * 缩放尺度是为了让噪声对于图像的影响在一个数量级上
 * 
 *  1. 计算特征点 X， Y坐标均值
 *  2. 计算特征点 X， Y坐标离均值的平均偏离程度
 *  3. 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶矩都为1
 *  4. 计算归一化矩阵：即将前面的操作用矩阵变换来表示
 * @param[in] vKeys                 待归一化的特征点
 * @param[in] vNormalizedPoints     特征点归一化后的坐标
 * @param[in] T                     归一化特征点的变换矩阵
 */
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    // 归一化的是这些点在x方向和y方向上的一阶绝对矩（随机变量的期望）

    // Step 1 计算特征点X Y 的坐标
    float meanX = 0;
    float meanY = 0;

    // 获取特征点数量
    const int N = vKeys.size();

    // 用来存储归一化后特征点的向量大小
    vNormalizedPoints.resize(N);

    // 遍历特征点
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    // 计算X Y均值
    meanX = meanX/N;
    meanY = meanY/N;

    // Step 2 计算特征点X Y坐标离均值的平均偏离程度 （不是标准差）
    float meanDevX = 0;
    float meanDevY = 0;

    // 将原始特征点减去均值坐标后，使x和y均值均为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 计算平均偏离程度
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    // 设置其倒数为一个缩放因子
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // Step 3 将x坐标和y坐标进行尺度归一化，使得x坐标和y坐标的一阶绝对矩为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // Step 4 计算归一化变换矩阵
    // [sX  0   -meanX*sX]
    // [0   sY  -meanY*sY]
    // [0   0       1    ]
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

/**
 * @brief 用位姿来对特征匹配点进行三角化，从中筛选合格的三维点
 * 
 * @param[in] R                     旋转矩阵
 * @param[in] t                     平移向量
 * @param[in] vKeys1                参考帧特征点
 * @param[in] vKeys2                当前帧特征点
 * @param[in] vMatches12            两帧特征点的匹配关系
 * @param[in] vbMatchesInliers      特征点对内点标记
 * @param[in] K                     相机内参矩阵
 * @param[in] vP3D                  三角化测量之后的特征点的3D空间坐标
 * @param[in] th2                   重投影误差的阈值
 * @param[in] vbGood                标记特征点是否成功三角化
 * @param[in] parallax              计算出来的比较大的视差（不一定是最大）
 * @return int                      返回good点计数
 */
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // 对给出的特征点对及 R t，通过三角化检查解的有效性，也称为 cheirality check

    // Calibration parameters
    // 获取相机内参参数
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    // 特征点是否是good点的标记，这里的特征点值的是参考帧中的特征点
    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    // 计算出来的每对特征点的视差
    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    // Step 1 计算相机的投影矩阵
    // 投影矩阵P 是一个3*4的矩阵，可以将一个空间点投影到平面上，获得其平面坐标，这里均指的是齐次坐标
    // 对于第一个相机的投影矩阵是 P1=K*[I|0] = [K|0]

    // 以第一个相机的光心作为世界坐标系,定义相机的投影矩阵
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    // 将K矩阵拷贝到P1矩阵的左侧3*3矩阵
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    // 第一个相机的光新设置为世界坐标系下的原点
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // 计算第二个相机的投影矩阵 P2=K*[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    // 第二个相机光心在世界坐标系下的坐标
    cv::Mat O2 = -R.t()*t;

    // 在开始遍历之前,将good点技术置为0
    int nGood=0;

    // 遍历所有匹配特征点对
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        // 跳过outliers
        if(!vbMatchesInliers[i])
            continue;

        // Step 2 获取特征点对,调用Triangulate()函数进行三角化,得到三角化测量之后的3D点坐标
        // kp1 kp2是匹配好的有效特征点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        // 存储3D点坐标
        cv::Mat p3dC1;

        // 利用三角法恢复3D点
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        // Step 3 第一关:检查三角化的三维坐标是否合法(非无穷值)
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // Step 4 第二关:通过三维点深度值正负、两相机光心视差角大小来检查是否合法

        // 得到向量PO1
        cv::Mat normal1 = p3dC1 - O1;
        // 计算距离
        float dist1 = cv::norm(normal1);

        // 得到向量PO2
        cv::Mat normal2 = p3dC1 - O2;
        // 计算距离
        float dist2 = cv::norm(normal2);

        // 有余弦公式 ab=|a||b|cos<a,b>
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 如果深度值为负值，为非法三维点，则跳过该匹配点对
        // ? 视差比较小时重投影误差比较大，则应该跳过。0.99998对应的角度为0.36°，大于0.36°，应该是cosParallax > 0.99998吧？？？
        // note 后面判断 vbGood 点时的条件也是 cosParallax < 0.99998。这样会不会导致初始化不稳定
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 将空间点变换到第二个相机坐标下
        cv::Mat p3dC2 = R*p3dC1+t;
        // 判断和上面相同
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Step5 第三关：计算空间点在参考帧和当前帧上的重投影误差，如果大于阈值则舍弃
        // Check reprojection error in first image
        // 计算在图像1上的重投影误差
        // 投影像素坐标为im1x, im1y
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        // 投影到参考帧图像上，因为参考帧下的相机坐标系和世界坐标系重合，所以直接投影就可以
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        // 计算参考帧上的重投影误差
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        // 如果误差大于阈值，则舍弃跳过
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        // 计算在图像2上的重投影误差
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        // p3dC2已经是第二个相机坐标系下的三维点，直接投影
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        // 计算当前帧上的重投影误差
        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
        // 如果误差大于阈值，则跳过
        if(squareError2>th2)
            continue;

        // Step 6 统计通过检验的3D点个数，记录3D点视差角
        // 到此步，说明当前遍历的这个特征点对靠谱，经过了检验，说明是一个合格的点，称之为good
        vCosParallax.push_back(cosParallax);
        // 存储这个三角化测量后的3D点在世界坐标系下的坐标
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        // good 点计数++
        nGood++;
        // 判断视差角，只有视差角稍大一点才会打good
        // bug 前面已经判断过了，而且good点计数都++了，还在这里判断呢！ 而且这判断的是视差角小于0.36°，也不对吧
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    // Step 7 得到3D中较小的视差角，并且转换成角度制表示
    if(nGood>0)
    {
        // 从小到大排序，cosParallax值越大，视差越小
        sort(vCosParallax.begin(),vCosParallax.end());

        // ? 排序后并没有选取最小的视差角，而是取一个较小的视差角
        // 作者做法，如果经过检验后的3D点个数小于50个，那么就取那个最小的视差角(cos值最大)，否则，就去排名第50个的较小的视察叫即可，为了避免3D点太多时出现太小的视差角
        size_t idx = min(50,int(vCosParallax.size()-1));
        // 转换为角度制
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
    // 如果没有good点，则视差角直接置为0
        parallax=0;

    // 返回good点计数
    return nGood;
}


/**
 * @brief 分解Essential矩阵得到Ｒ,t
 *    分解Ｅ矩阵将得到４组解
 * 
 * @param[in] E     本质矩阵
 * @param[in] R1    旋转矩阵１
 * @param[in] R2    旋转矩阵２
 * @param[in] t     平移向量，另一个取反
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    // 对本质矩阵进行SVD分解
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    // 左奇异矩阵U的最后一列就是t，对其进行归一化
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    // 构造一个绕Z轴旋转 PI/2 的旋转矩阵W，则可以得到旋转矩阵 R1 = U W Vt
    // 计算完成后，需检查旋转矩阵 det(R) = 1
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    // 旋转矩阵行列式有为1的约束，如果算出来结果为-1需要取反
    if(cv::determinant(R1)<0)
        R1=-R1;

    // 对W转置相当于绕Z轴旋转 -PI/2
    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
