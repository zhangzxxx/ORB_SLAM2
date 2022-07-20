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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

/**
 * @brief Construct a new Frame:: Frame object 单目构造函数
 * 
 * @param[in] imGray            灰度图
 * @param[in] timeStamp         时间戳
 * @param[in] extractor         ORB特征提取器
 * @param[in] voc               ORB字典句柄
 * @param[in] K                 相机内参矩阵
 * @param[in] distCoef          相机畸变参数
 * @param[in] bf                baseline * f
 * @param[in] thDepth           区分远近点的深度阈值
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    // Step1 帧的ID 自增
    mnId=nNextId++;

    // Step2 计算图像金字塔的参数
    // Scale Level Info
    // 获取金字塔层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    // 获取金字塔缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // 获取缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
    // 获取每层图像缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // 获取每层图像缩放因子倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // 获取每层图像缩放因子平方
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // 获取每层图像缩放因子平方倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // Step3 对这个单目图像进行特征点提取，0表示左图，1表示右图，单目为左图
    ExtractORB(0,imGray);

    // 提取的关键点个数
    N = mvKeys.size();

    // 如果没有提取到关键点，直接返回
    if(mvKeys.empty())
        return;

    // Step4 用OPENCV的矫正函数、内参对提取的特征点进行矫正
    UndistortKeyPoints();

    // Set no stereo information
    // 由于单目相机无法直接获得立体信息，所以这里给右图像对应点和深度值赋值-1，表示没有相关信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 记录地图点是否为外点，初始化均为外点 false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // Step5 计算去畸变后的图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数重新发生变换之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变后的图像边界
        ComputeImageBounds(imGray);

        // 表示一个图像像素相当于多少图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 给类的静态成员变量赋值
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 可能这种除法计算时间长，这里直接存储计算结果，以便进行乘法
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 特殊的初始化完成，将标志复位
        mbInitialComputations=false;
    }

    // 计算基线
    mb = mbf/fx;

    // 分配特征点到图像网格中
    AssignFeaturesToGrid();
}

/**
 * @brief 将提取到的ORB特征点分配到图像网格中
 * 
 */
void Frame::AssignFeaturesToGrid()
{
    // Step1 给存储特征点的网格数组 Frame::mGrid分配空间
    // ? 乘以0.5为什么
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    // 开始对每mGrid这个二维数组中的每一个vector元素遍历并分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // Step2 遍历所有特征点，将每个特征点在mKeyUn中的索引值放到对应的网格mGrid中
    for(int i=0;i<N;i++)
    {
        // 从类的乘以变量中获取已经去除畸变后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];

        // 存储某个特征点所在的网格坐标 nGridPosX范围[0, FRAME_GRID_COLS]  nGridPosY [0, FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX，nGridPosY中，返回true，否则返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            // 如果找到特征点所在网格坐标，将该特征点索引存储在对应的网格数组中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/**
 * @brief 提取特征点
 * 
 * @param[in] flag 0 左图，1 右图
 * @param[in] im   待提取特征点输入图像
 */
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        // 左图的话就使用左图指定的特征点提取器，并将提取结果保存到对应的变量中
        // 这里使用了仿函数来完成，重载了括号运算符，ORBextractor::operator()
        (*mpORBextractorLeft)(im,               // 待提取特征点图像
                              cv::Mat(),        // 掩码图像，实际代码中没有用到
                              mvKeys,           // 特征点保存数组
                              mDescriptors);    // 特征点描述子
    else
        // 如果是右图，就使用右图指定的特征点提取器，并将结果保存到对应变量中
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief 找到以x, y 为中心，半径为r的圆形内且金字塔层级在 [minLevel, maxLevel]的特征点
 * 
 * @param[in] x                 特征点坐标x
 * @param[in] y                 特征点坐标y
 * @param[in] r                 搜索半径
 * @param[in] minLevel          最小金字塔层级
 * @param[in] maxLevel          最大金字塔层级
 * @return vector<size_t>       返回搜索到的候选匹配点id
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    // 存储搜索的结果
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // Step 1 计算半径为r圆 上下左右边界所在的网格列和行的id
    // 查找半径为r的圆左侧边界所在网格列坐标。
    // (minMaxX - mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
    // mfGridElementWidthInv = FRAME_GRID_COLS/(mnMaxX - mnMinX) 是上面倒数，表示每个像素平均可以分得几个网格（肯定小于1）
    // (x - mnMinX - r) 可以看做是从图像左边角 mnMinX 到半径r的圆的左边界区域占的像素列数
    // 两者相乘，就是求出半径为r的圆的左侧边界在哪个网格中，保证minCellX大于0
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));

    // 如果最终求得的圆的左侧边界所在的网格列超过了设定的上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    // 计算圆右边界所在的网格列数
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    // 计算圆上边界所在的网格行数
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    // 计算圆下边界所在网格行数
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    // 检查需要搜索的金字塔层数是否符合要求
    // ? 若minLevel > 0 则 maxLevel　>= 0 肯定成立
    // ? 改为　const bool bCheckLevels = (minLevel>= 0) || (maxLevel>=0);
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放入输出数组中
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // 获取该网格内所有特征点在　Frame::mvKeysUn 中的索引
            const vector<size_t> vCell = mGrid[ix][iy];
            // 如果这个网格中没有特征点，则跳过
            if(vCell.empty())
                continue;

            // 如果这个网格中有特征点，则遍历该网格中的所有特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // 通过索引获取该特征点的引用
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                // 检查金字塔层数，保证范围合理
                if(bCheckLevels)
                {
                    // 如果该特征点所在金子塔层数　不在搜索层级之间　则跳过
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 计算候选帧点到圆心的距离，查看是否在这个圆形范围之内
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 如果特征点在该搜索圆范围内，存储该特征点索引作为候选特征点
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/**
 * @brief 计算特征点对应的网格坐标
 * 
 * @param[in] kp        特征点
 * @param[in] posX      特征点所在网格横坐标X
 * @param[in] posY      特征点所在网格纵坐标Y
 * @return true         找到返回真
 * @return false        
 */
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // 计算特征点x,y坐标落在哪个网格内
    // mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
    // mfGridElementHeightInv=(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
    // kp.pt.x-mnMinX单位像素，mfGridElementWidthInv 单位网格/像素
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 因为特征点进行了去畸变且计算网格坐标时进行了取整，得到的坐标可能会落在网格坐标外面，需要进行检查
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


/**
 * @brief 用内参对特征点去畸变，结果保存在mvKeyUn中
 * 
 */
void Frame::UndistortKeyPoints()
{
    // Step1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话说明畸变参数都是0
    // mDistCoef 存储opencv指定参数顺序为 k1 k2 p1 p2 k3
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }
    // Step2 用opencv函数进行去畸变
    // Fill matrix with points
    // 将N个特征点保存在N×2的矩阵中
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn, int rows = 0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了调用opencv中的函数来进行去畸变，需要先将矩阵调整为两通道，（对应坐标x y）
    mat=mat.reshape(2);
    cv::undistortPoints(mat,    // 输入特征点坐标
                        mat,    // 输出的矫正后的特征点坐标，覆盖原矩阵
                        mK,mDistCoef,   // 矫正参数
                        cv::Mat(),      // 一个空矩阵，对应函数原型中的R
                        mK);            // 新内参矩阵，对应函数原型中的P
    // 调整回只有一个通道，回归正常处理方式
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // Step3 存储矫正后的特征点
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

/**
 * @brief 计算去畸变图像边界
 * 
 * @param[in] imLeft 需要计算边界的图像
 */
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变第一个参数为不为0，用OpenCV函数进行畸变操作
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像的四个边界点坐标： (0, 0), (cols, 0), (0, rows), (cols, rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;             // left up
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;     // right up
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;     // left bottom
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;     // right bottom

        // Undistort corners
        // 和前面矫正特征点一样操作，将这几个边界点作为输入进行矫正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // 矫正后的四个边界点已经不能够围成一个严格的矩形，因此在这四个边界点的外侧加框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));        // 左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));        // 右上和右下横坐标最小的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));        // 左上和右上纵坐标最大的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));        // 左下和右下纵坐标最大的

    }
    else
    {
        // 如果畸变参数为0，就直接获取图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
