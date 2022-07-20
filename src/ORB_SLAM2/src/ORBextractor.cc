/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

/**
 * @brief 用于计算特征点方向
 * 计算特征点方向是为了使得提取的特征点具有旋转不变性
 * 方法是灰度质心法：以几何中心和灰度质心连线作为该特征点的方向
 * @param image     要进行操作的某层金字塔图像
 * @param pt        特征点坐标
 * @param u_max     图像块的每一行的坐标边界
 * @return float    返回特征点的角度，范围为 0~360°，精度为0.3°
 */
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    // m_01 y方向上的矩，m_10 x方向上的矩，矩按图像块的坐标x,y加权
    int m_01 = 0, m_10 = 0;

    // 获得这个特征点所在图像块中心点坐标灰度值的指针
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    // v = 0 这条中心线需要特殊计算
    // 这里下标u可以是负的！中心水平线上的像素按x (u)坐标加权
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    // step1表示这个图像每一行包含的字节总数
    int step = (int)image.step1();
    // 以v=0为对称轴，每次遍历两行加速计算。v从1开始
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        // 获取某行像素横坐标最大范围，图像块是圆形。
        int d = u_max[v];
        // 在坐标范围内挨个遍历像素，实际上是一次遍历两个
        // 假设每次处理的两个点坐标，中心线下方为(x, y)，中心线上方为(x, -y)
        // 对于某次待处理的两个点：m_10 = sum x*I(x,y) = x*I(x,y) + x*I(x,-y)
        // 对于某次待处理的两个点: m_10 = sum y*I(x,y) = y*I(x,y) + (-y)*I(x,(-y)) = y*(I(x,y)-I(x,-y))
        for (int u = -d; u <= d; ++u)
        {
            // 得到需要进行加运算和减运算的像素灰度值
            // val_plus:在中心线下方x=u时的像素灰度值
            // val_plus：在中心线上方x=u时的像素灰度值
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            // 在v（y轴)上，2行所有像素灰度值之差
            v_sum += (val_plus - val_minus);
            // 在u （x轴）上，用u坐标加权和(u也有正负号)
            m_10 += u * (val_plus + val_minus);
        }
        // 在轴上进行加权和
        m_01 += v * v_sum;
    }

    // 为了加快计算还使用了fastAtan2函数，输出 0~360°角度，精度0.3°
    return fastAtan2((float)m_01, (float)m_10);
}

// 乘数因子，一度对应多少弧度
const float factorPI = (float)(CV_PI/180.f);
/**
 * @brief 计算关键点ORB描述子
 * 
 * @param[in] kpt   关键点
 * @param[in] img   图像
 * @param[in] pattern   随机点集
 * @param[in] desc      描述子
 */
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    // 关键点方向，单位弧度
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    // 获取图像关键点中心指针
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    // 获取图像每行的字节数
    const int step = (int)img.step;

    // 原始的BRIEF描述子没有方向不变性，通过加入关键点的方向来计算描述子，称之为Steer BRIEF，具有极好旋转不变性
    // 具体地，在计算的时候需要将这里选取的采样模板中点的x轴方向旋转到特征点方向。
    // 获取采样点中某个idx所对应的点的灰度值，这里旋转前的坐标为(x, y)，旋转后的坐标为(x', y') x'=xcos(theta) - ysin(theta), y'=xsin(theta) + ycos(theta)
    // 下面表示 y'*step + x'
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    // 每个关键点BRIEF由32*8位构成
    // 其中每一位来自两个像素点灰度值的直接比较，所以每比较出8bit结果，需要16个随机点，这也是为什么pattern需要 +=16 的原因
    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;                // 参与比较的2个特征点的灰度值，以及比较结果0或1
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;                              // 描述子本字节的bit0
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;                      // 描述子本字节的bit7

        // 保存当前比较出来的描述子的这个字节
        desc[i] = (uchar)val;
    }

    // 为了避免和程序中其他部分冲突，使用完之后就取消这个宏定义
    #undef GET_VALUE
}

// 预先定义好的随机点集，256是值可以提取出256bit的描述子信息，每个bit由一对点比较得来，4=2*2，前面的2是指两个点进行比较，后面的2是指一个点有两个坐标
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


// 特征点提取构造函数
ORBextractor::ORBextractor(
                int _nfeatures,             // 提取特征点数
                float _scaleFactor,         // 缩放因子
                int _nlevels,               // 金字塔层数
                int _iniThFAST,             // FAST特征点提取阈值
                int _minThFAST):            // 如果提取不到足够的特征点，使用该阈值
                    nfeatures(_nfeatures), 
                    scaleFactor(_scaleFactor), 
                    nlevels(_nlevels),
                    iniThFAST(_iniThFAST), 
                    minThFAST(_minThFAST)
{
    // 存储每层图像缩放系数的vector
    mvScaleFactor.resize(nlevels);
    // 存储这个sigma的平方，也就是每层图像相对原始图像缩放因子的平方
    mvLevelSigma2.resize(nlevels);
    // 原始图像，这两个参数都是1
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    // 逐层计算图像金字塔中图像相当于原始图像的缩放系数
    for(int i=1; i<nlevels; i++)
    {
        // 当前层缩放因子=下一层缩放因子×缩放系数
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        // sigma平方就是每层图像相对于原始图像缩放银子的平方
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    // 上面两个参数的倒数
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    // 存储图像金字塔中图像个数
    mvImagePyramid.resize(nlevels);

    // 每层图像金字塔中应提取的特征点个数
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    // 单位面积应该提取的特征点数：特征点总数/所有层图像面积，第0层应该分配的特征点数
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    // 用于累计已分配的特征点数
    int sumFeatures = 0;
    // 第i层特征点数 = i-1层×缩放因子的平方  缩放因子小于1大于0的数
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    // 由于前面的特征点个数取整操作，可能会导致剩余一些特征点数没有得到分配。将其分配到最上层图像中
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // 成员变量pattern的长度，也就是点的个数，512表示512个点，256个点对
    const int npoints = 512;
    // 获取用于计算BRIEF描述子的随机采样点 点集 头指针
    // pattern0数据类型为Point*， bit_pattern_31为int[]型，需要强制类型转换
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    // c++11 std::back_inserter:参数 支持push_back操作的容器，返回能用于添加到运算容器尾端的std::back_insert_inerator
    // 将pattern插入到pattern0+npoints后面 pattern0数据首地址+偏移量 = 数据尾地址
    // 即将在全局变量区域的int格式的随机采样点以cv::point格式复制到当前类对象中的成员变量中
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    // 以下内容与特征点旋转计算有关
    // 预先计算patch中行的结束位置 +1中的1表示那个圆的中间行 HALF_PATCH_SIZE=15即为 计算特征点角度的圆的半径
    umax.resize(HALF_PATCH_SIZE + 1);

    // cvFloor返回不大于参数的最大整数值，cvCeil返回不小于参数的最小整数值，cvRound则是四舍五入
    // 1.最大行数vmax初始化为r*sin45° + 1，+1是为了vmax和vmin边界值在遍历的过程中产生交叉，因为做了取整操作防止漏掉，这里的最大行号是指计算的时候的最大行号
    // 2.最大行数vmax就是0~45°这个过程中的最大行数，不是这个圆的最大行数
    // 3.vmin最小行数向上取整，是90°～45°这个过程中的行数
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    // 半径平方 整个PATCH是31*31
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    // umax 16个数值依次是：15 15 15 15 14 14 14 13 13 12 11 10 9 8 6 3
    // umax[v] 其中索引v是行数，umax[v]的值是列数
    // 图像坐标系,四分之一圆
    /**
     * ------------------------------------>X
     * | * * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * * *
     * | * * * * * * * * * * * * *
     * | * * * * * * * * * * * * *
     * | * * * * * * * * * * * *
     * | * * * * * * * * * * *
     * | * * * * * * * * * *
     * | * * * * * * * * *
     * | * * * * * * * *
     * | * * * * * *
     * | * * *
     * |
     * |
     * Y
     */
    for (v = 0; v <= vmax; ++v)
        // 勾股定理 r^2 - v*v
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    //cvRound很容易出现不对称的情况，随机采样的特征点集也不能够满足旋转之后的采样不变性。所以使用对称的方式计算1/4圆
    // 计算v = vmin至HALF_PATCH_SIZE时的 umax[v], v 从11到15之间的值依次为 10 9 8 6 3
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

/**
 * @brief 计算特征点方向
 * 
 * @param image         特征点所在当前金字塔图像
 * @param keypoints     特征点集
 * @param umax          每个特征点所在图像区块的每行的边界
 */
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    // 遍历所有特征点
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        // 调用IC_Angle计算特征点方向
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}


/**
 * @brief 将提取器节点分成4个子节点，同时也完成图像区域的划分、特征点归属的划分、以及相关标志位的置位
 * 
 * @param[in & out] n1 提取器节点1：左上
 * @param[in & out] n2 提取器节点1：右上
 * @param[in & out] n3 提取器节点1：左下
 * @param[in & out] n4 提取器节点1：右下
 */
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    // 得到当前提取器节点所在图像的一半长宽，结果需要取整 
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    // 将一个图像区块细分为四个小块
    // n1 存储左上区域的边界
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    // 用来存储在该节点对应的图像网格中提取出来的特征点的vector
    n1.vKeys.reserve(vKeys.size());

    // n2 存储右上区域的边界
    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    // n3 存储左下区域的边界
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    // n4 存储右下区域的边界
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    // 遍历当前提取器节点的vkeys中存储的特征点
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        // 判断这个特征点在当前特征点提取器节点图像中的哪个区域，即哪个子图像区块，然后将其添加到那个特征点提取器节点的vKeys中
        // NOTICE BUG 这里是直接进行比较的，但是特征点的坐标是在“半径扩充图像”坐标系下的，而节点区域的坐标则是在“边缘扩充图像”坐标系下的
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    // 判断每个子特征点提取器节点所在的图像中特征点的数目（分配给子节点的特征点数目），然后做标记
    // 1 表示这个节点不再继续往下分裂
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}

/**
 * @brief 使用四叉树对金字塔图像中的特征点进行平均和分发
 * 
 * @param vToDistributeKeys     // ? 等待进行分配的四叉树中的特征点  
 * @param minX                  当前图像的边界，坐标是在“半径扩充图像”坐标系下的坐标
 * @param maxX 
 * @param minY 
 * @param maxY 
 * @param N                     希望能提取出的特征点个数
 * @param level                 金字塔层数
 * @return vector<cv::KeyPoint> 已经均匀分配好的特征点容器
 */
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes   
    // Step1 根据宽高比例确定初始节点数目
    // width = maxBorderX - minBorderX; 原始图像宽752， 可提取的特征图像宽752-32=720    maxBorderX=736，minBorderX=16
    // heigh = maxBorderY - minBorderY; 原始图像高480， 可提取的特征图像宽为480-32=448
    // 计算应该生成的初始节点的个数，根节点的数量nIni是根据边界的宽高比值确定的，一般是1或2
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
    // ? 如果图像是宽高比小于0.5，nIni = 0 则hx会报错
    // 720/2 = 360
    const float hX = static_cast<float>(maxX-minX)/nIni;

    // 存储提取的节点的链表
    list<ExtractorNode> lNodes;

    // 存储初始提取节点指针的vector，用于索引节点
    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    // Step2 生成初始提取器节点
    // 存储每个节点的起始点坐标和终止点坐标
    // (UL.x, UL.y) = (0, 0)  ...... (UR.x, UR.y) = (360, 0) (UR.x, UR.y) = (360, 0)     ...... (UR.x, UR.y) = (720, 0)
    // (BL.x, BL.y) = (0, 448)...... (BR.x, BR.y) = (360, 448) (BR.x, BR.y) = (360, 448) ...... (BR.x, BR.y) = (720, 448)
    for(int i=0; i<nIni; i++)
    {
        // 生成提取器节点
        ExtractorNode ni;
        // 设置提取器节点的图像边界
        // 这里和提取FAST角点区域相同，都是“半径扩充图像”，特征点坐标从0开始
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);            // UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);          // UpRight
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);                     // BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);                     // BottomRight
        // 重设vKeys大小, vToDistributeKeys
        ni.vKeys.reserve(vToDistributeKeys.size());

        // 将刚才生成的提取节点添加到链表中
        // note 虽然这里的ni是局部变量，但由于这里的push_back()是拷贝参数的内容到一个新的对象中然后在添加到链表中，所以当本函数退出之后这里的内存不会成为“野指针”
        lNodes.push_back(ni);
        // 存储这个初始的提取器节点句柄 list::back()输出容器中最后一个元素
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    // Step3 将特征点分配到子提取器节点中
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        // 获取这个特征点对象
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        // 按特征点的横轴位置，分配给属于那个图像区域的提取器节点（最初的提取器节点）
        // kp.pt.x/hX 只有两个结果：小于1都放在第一个节点里、大于1都放在第二个节点里
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    // Step4 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节目的
    // ? 这个过程是否可以通过判断 nIni个数和vKeys.size()来完成
    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        // 如果初始的提取器节点所分配的特征点的个数为1
        if(lit->vKeys.size()==1)
        {
            // 那么就将标志位置位true，表示此节点不可再分
            lit->bNoMore=true;
            // 更新迭代器
            lit++;
        }
        // 如果一个提取器节点没有分配到特征点，那么就从列表中直接删除它，注意lit没有必要++，否则会造成跳过元素的情况
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
        // 如果上面的情况和当前的特征点提取器节点无关，只需更新迭代器
            lit++;
    }

    // 结束标志位置位false
    bool bFinish = false;

    // 记录迭代次数，只是记录，并没有起到作用
    int iteration = 0;

    // 声明一个vector用于存储节点的vSize和句柄对
    // 这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    // 调整大小，将一个初始化节点“分裂”成4个
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    // Step5 利用四叉树方法对图像进行划分区域，均匀分配特征点
    while(!bFinish)
    {
        // 更新迭代次数计数器，只是记录，没有起到作用
        iteration++;

        // 保存当前节点个数
        int prevSize = lNodes.size();

        // 重新定位迭代器指向列表头部
        lit = lNodes.begin();

        // 需要展开的节点个数，这个一直保持累计，不清零
        int nToExpand = 0;

        // 因为是在循环中，前面的循环体可能污染了这个变量，所以清空
        // 这个变量也只是统计了某一个循环中的点，记录了一次在分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目及其句柄
        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            // 如果提取器节点中只有一个特征点
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                // 没有必要继续细分，跳过当前节点
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                // 当前节点有超过一个特征点，要继续进行分裂
                ExtractorNode n1,n2,n3,n4;
                // 细分成四个子区域
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                // 如果这里分裂出来的子区域中由特征点，就将这个子区域的节点添加到提取器节点的链表中
                // 这里的条件是，有节点即可
                if(n1.vKeys.size()>0)
                {
                    // 添加顺序，是将分裂的子区域节点添加到链表头部
                    lNodes.push_front(n1);
                    // 再判断其子区域节点中特征点数目是否大于1                
                    if(n1.vKeys.size()>1)
                    {
                        // 如果有超过一个特征点，那么待展开的节点计数+1
                        nToExpand++;
                        // 保存这个特征点数目和节点指针的信息
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        // lNodes.front().lit 和前面迭代的lit不同，迭代的lit是while循环里面作者命名的遍历的指针名称，而这里是node结构体里的一个指针用来记录节点的位置
                        // ? 这个结构体中lit 貌似没有用到
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                // 当这个母节点expand之后，就从链表中将其删除，能够进行分裂操作说明至少有一个子节点的区域中其特征点的数量是 > 0的
                // 分裂的方式是 后加的节点先分裂，先加的节点后分裂
                lit=lNodes.erase(lit);
                continue;
            }   // 判断当前节点中是否有超过一个特征点
        }  // 遍历所有

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        // 停止这个过程的条件：1. 当前的节点数已经超过了要求的特征点数 或 2. 当前的所有节点中都只包含一个特征点
        // prevSize保存的是分裂之前的节点个数，如果分裂之前和分裂之后的节点个数一样，说明当前所有节点区域都只有一个特征点，不能再细分了
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            // 结束标志位
            bFinish = true;
        }
        // Step6 当再划分之后所有的节点数大于要求数目时，就慢慢划分知道使其刚刚达到或者超过要求的特征点个数
        // 可以展开的子节点个数 nToExpand*3，是因为1分为四之后，会删除原来的主节点，所以乘以3
        /**
         * // ? nToExpand是一直累计的。如果因为特征点过少，跳过了下面的else-if，又进行了一次上面的while循环 遍历list操作之后，lNode.size()增加了，nToExpand也增加
         *  在很多次操作之后， ((int)lNodes.size()+nToExpand*3)>N 很快就能满足，但是此时只进行一次对vSizeAndPointerToNode中点进行分裂的操作是肯定不够的；
         * 理想中，作者下面的for 理论上只要进行一次就能满足，不过作者所考虑的”不理想情况“应该是分裂后出现的节点所在区域可能没有特征点，因此将for循环放在了一个while循环里，
         * 通过再次进行for循环，在分裂一次解决这个问题。但如果”不理想情况“是因为前面的一次对vSizeAndPointerToNode中的特征点进行for循环不够，需要将其放在另一个循环（也就是作者的while循环）
         * 中不断尝试直到达到退出条件
         */
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {
            // 如果再分裂一次，数目就要超了，这里想办法尽可能使其刚刚达到或超过要求的特征点个数时就退出
            // 这里的nToExpand和vSizeAndPointerToNode不是一次循环对应一次循环关系，前置是累计计数，后者是某一个循环的
            // 一直循环，直到标志位为true
            while(!bFinish)
            {
                // 获取当前list节点个数
                prevSize = lNodes.size();

                // 保留那些还可以进行分裂的节点信息，这是深拷贝
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                // 清空
                vSizeAndPointerToNode.clear();

                // 对需要分裂的节点进行排序，对pair的第一个元素进行排序，默认从小到大
                // 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
                // note 这里的排序规则非常重要，会导致每次最后产生的特征点都不一样。建议使用stable_sort (如果出现相同的元素，不改变其顺序)
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                // 从后往前遍历vector中的pair
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    // 对每个需要进行分裂的节点进行分裂
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    // 执行和前面一样的操作，已经是二级子节点了
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            // 因为这里还会有对vSizeAndPointerToNode操作，所以前面需要备份vSizeAndPointerToNode中的数据，为后续可能的又一次for循环做准备
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    // 删除母节点，这里其实应该是一级子节点
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    // 判断是否节点数超过了所需特征点数，使得话就退出，不是的话就继续这个分裂过程，知道刚刚达到或超过要求的特征点个数
                    // 这个判断，是再分裂一次之后判断下一次分裂是否会超过N，如果不是就全部进行分裂（多了一个判断，运算速度会稍微快些）
                    if((int)lNodes.size()>=N)
                        break;
                }   // 遍历vPrevSizeAndPointerToNode并对其中指定的节点进行分裂，直到达到或刚刚超过特征数
                
                // 这里理想中应该是一个for循环就达到结束条件了，但是作者想的可能是，有些子节点所在的区域会没有特征点，因此很有可能一次for循环之后的数目还是不能满足要求
                // 所以还需要判断结束条件 并且再来一次
                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }   // 一直进行nToExpand累加的节点分裂过程，知道分裂后的节点数刚刚达到或超过特征点数目
        }   // 当本次分裂后达不到结束条件但是再进行一次完整的分裂之后就可以达到结束条件时
    }   //利用4叉树方法对图像进行划分区域

    // Retain the best point in each node
    // Step7 保留每个区域内响应值最大的特征点
    // 使用vector存储感兴趣的特征点过滤的结果
    vector<cv::KeyPoint> vResultKeys;
    // 重置大小为要提取的特征点数目
    vResultKeys.reserve(nfeatures);
    // 遍历节点链表
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        // 得到该节点区域中的特征点容器句柄
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;

        // 得到指向该节点区域第一个特征点的指针，后面作为最大响应值对应的关键点
        cv::KeyPoint* pKP = &vNodeKeys[0];

        // 用一个关键点响应值初始化最大响应值
        float maxResponse = pKP->response;

        // 遍历该节点区域中的特征点容器中的所有特征点，注意是从1开始，0已经用过了
        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                // pKP指向具有最大相应值的关键点
                pKP = &vNodeKeys[k];
                // 更新最大响应值
                maxResponse = vNodeKeys[k].response;
            }
        }
        // 将该区域最大响应值特征点放入结果容器中
        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

// 计算四叉树特征点，函数后面的OctTree只是说明了在过滤和分配特征点时所使用的方式
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    // 重新调整图像层数
    allKeypoints.resize(nlevels);

    // 图像Cell的尺寸，是个正方形
    const float W = 30;

    // 对每一层图像做处理
    for (int level = 0; level < nlevels; ++level)
    {
        // 计算这层图像的坐标边界
        //  note EDGE_THRESHOLD=19值得是可以提取特征点的有效图像边界，FAST关键点计算需要半径为3的圆，BRIEF描述子计算角度需要半径为16的圆
        const int minBorderX = EDGE_THRESHOLD-3;                // -3 是因为在计算FAST关键点时需要建立一个半径为3的圆
        const int minBorderY = minBorderX;                      
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        // 存储需要进行平均分配的特征点
        vector<cv::KeyPoint> vToDistributeKeys;
        // 一般都是过量采集，所以预分配的空间大小是ｎFeatrues*10
        vToDistributeKeys.reserve(nfeatures*10);

        // 计算特征点提取的图像区域尺寸
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        // 计算网格Cell在当前层图像的有效行数和列数
        const int nCols = width/W;
        const int nRows = height/W;
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        // 遍历图像网格，以行开始遍历
        for(int i=0; i<nRows; i++)
        {
            // 计算当前网格的初始行坐标，有效坐标应该是 iniY + 3
            const float iniY =minBorderY+i*hCell;
            //计算当前网格的最大行坐标，+6=+3+3，即多加一个3是为了cell边界像素进行FAST特征点提取用
            // 前面的EDGE_THRESHOLD=19是为了进行高斯模糊而预留的边界
            // 目测一个图像网格的大小是25×25
            float maxY = iniY+hCell+6;

            // 如果初始的行坐标超过了有效图像（即原始图像）边界了，就跳过。
            if(iniY>=maxBorderY-3)
                continue;
            // 如果图像的大小导致不能够划分出整齐的图像网格，就只好委屈最后一行了，最下一个网格变小
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            // 遍历图像列
            for(int j=0; j<nCols; j++)
            {
                // 计算初始列坐标
                const float iniX =minBorderX+j*wCell;
                // 计算该网格的列坐标
                float maxX = iniX+wCell+6;
                // 判断坐标是否在有效图像（原始图像，可以提取特征点的图像）中，超出，则跳过
                // bug 应该是 -3 吧
                if(iniX>=maxBorderX-6)
                    continue;
                // 如果最大坐标越界，则委屈一下
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                // FAST提取角点，划分cell就是为了自适应阈值，可能有些地方会提取不到角点，划分为cell后则可以降低该网格的阈值来强制提取角点
                // 存储该cell中的关键点
                vector<cv::KeyPoint> vKeysCell;
                // 调用opencv库中的FAST()来提取角点
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),     // 待检测的图像
                     vKeysCell,             // 存储关键点位置的容器
                     iniThFAST,             // 检测阈值
                     true);                 // 是否使用非极大值抑制

                // 如果该图像块中没提取到关键点，则使用最小阈值再提取一次
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFAST,true);
                }

                // 当图像cell中提取到了关键点时
                if(!vKeysCell.empty())
                {
                    // 遍历所有FAST角点
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        // note 到目前为止，这些角点的坐标都是基于图像cell的，提取之后需要将其恢复到当前的[坐标边界]下的坐标
                        // 这样做是因为在使用八叉树法整理特征点的时候将会使用得到这个坐标
                        // 在后面会被继续转换成在当前图像层的扩充图像坐标系下的坐标
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        // 然后将其加入到”等待被分配“的特征点容器中
                        vToDistributeKeys.push_back(*vit);
                    } // 遍历图像cell中提取到的所有FAST角点，并且恢复其在整个图像金字塔当前层下的坐标
                }

            }
        }

        // 声明一个对当前图像层特征点的引用容器
        vector<KeyPoint> & keypoints = allKeypoints[level];
        // 重置容器大小为所需特征点数目（此时也是扩大了，因为不可能一层就提取到所有关键点）
        keypoints.reserve(nfeatures);

        // 根据 mnFeaturesPerLevel[level] 该层需要的特征点数，对特征点进行删除
        // 返回值是一个保存有剔除后保留下来的特征点的vector容器，得到的特征点坐标依然是在当前图像层下的
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        //  PATCH_SIZE是对于底层图像来说的，现在需要根据当前图层的尺度缩放倍数进行缩放得到缩放后的PATCH大小 和特征点的方向计算有关
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        // 获取保留下来的特征点数目
        const int nkps = keypoints.size();
        // 遍历这些特征点，恢复其在当前图像层坐标系下的坐标
        for(int i=0; i<nkps ; i++)
        {
            // 对每一个保留下来的特征点，恢复到相对于当前图层“边缘扩充图像下”的坐标系的坐标
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            // 记录特征点来源的金字塔图层
            keypoints[i].octave=level;
            // 记录计算方向的PATCH，缩放后对应的大小， 又被称作特征点半径
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    // 然后计算这些特征点的方向信息，注意分层计算
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

// 使用旧方法计算关键点：将图像划分为小区域，每个区域使用单独阈值计算FAST角点
// 检测完成之后使用DistributeOctTree函数对检测到的所有角点进行筛选，使得角点均匀分布
void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    // 根据图像金字塔层数调整关键点层数
    allKeypoints.resize(nlevels);

    // 计算底层图像的长宽比，也是所有图像的长宽比
    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    // 遍历金字塔图像层
    for (int level = 0; level < nlevels; ++level)
    {
        // 该层需要提取的特征点数
        const int nDesiredFeatures = mnFeaturesPerLevel[level];
        // 分别按行列划分为多少个网格
        // ? 为啥乘以5，为啥这样计算本层图像中网格的行数和列数
        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        // 边界，图像的有效区域，最新的计算方法中边界为了计算边界FAST角点会进行半径扩充 minBorderX = EDGE_THRESHOLD - 3
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        // 坐标区域的宽和高，以及每个网格的宽高
        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        // 计算本层图像中的总Cell数
        const int nCells = levelRows*levelCols;
        // 计算每个Cell中需要提取的特征点数
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        // 第三层vector：当前Cell的特征点容器
        // 第二层vector：包含了一行Cell中，每个Cell的上面的特征点容器
        // 第一层vector：按列划分Cell，存储了所有行Cell的容器
        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        // 每个Cell中应该保留的特征点数
        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        // 每个Cell中实际提取出来的特征点数
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        // 每个Cell中是否只提取出了一个特征点的标记
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        // 保存每个Cell的起始x,y坐标
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        // 那些不满足Cell应该提取到的特征点数的Cell个数
        int nNoMore = 0;
        // 存储需要继续分裂的Cell的计数
        int nToDistribute = 0;

        // 考虑提取FAST角点时的半径图像，计算Cell边界使用的增量
        // +3+3 代表起始侧和终止测提取FAST角点时的那个半径为3个像素的的圆， cellH代表有效的可以计算FAST角点的图像长度
        float hY = cellH + 6;

        // 按图像行网格数遍历
        for(int i=0; i<levelRows; i++)
        {
            // Cell起始y/v坐标 -3考虑了半径为3的圆
            const float iniY = minBorderY + i*cellH - 3;
            // 记录，因为同样的网格分布在后面进行特征点均匀分布时同样会用到
            iniYRow[i] = iniY;

            // 如果是行最后一个网格，最下面的网格
            if(i == levelRows-1)
            {
                // 计算当前网格起始到终止位置的增量（考虑了半径3）
                hY = maxBorderY+3-iniY;
                // 如果为负，说明这个地方不够再划分一个网格来提取特征点了
                /**
                 * ? 如果是最后一行网格计算的意义 要求 hY > 0 即 maxBorderY+3-iniY > 0. 但是前面iniY已经减过3了。 
                 * 设原始图像的起始坐标为BorderY,那么就有 iniY = BorderY - 3. 则最后一行要求为： maxBorderY + 6 - BorderY > 0
                 * 但 BorderY 是不可能超过 maxBorderY的，那么这里的判断意义是什么？是否有必要？
                 */
                if(hY<=0)
                    continue;
            }

            // 计算一个Cell起始列到终止列的坐标增量 +6=+3+3，前面的+3是为了弥补 iniX 相对于minBorderX + j*cellW后面的减3，
            // 后面的+3则是在minBorderX + （j+1)*cellW基础上表示考虑到半径为3的圆
            float hX = cellW + 6;

            // 按网格列开始遍历
            for(int j=0; j<levelCols; j++)
            {
                // 当前网格起始列坐标
                float iniX;

                // 如果是网格是在第一行
                if(i==0)
                {
                    // 就计算初始边界
                    iniX = minBorderX + j*cellW - 3;
                    // 并且记录以备后用
                    iniXCol[j] = iniX;
                }
                else
                {
                    // 如果不是第一行网格，那么就可以读取之前计算好的初始坐标
                    iniX = iniXCol[j];
                }

                // 如果是最后一列网格
                if(j == levelCols-1)
                {
                    // 计算它的列坐标增量
                    hX = maxBorderX+3-iniX;
                    // 如果不满足FAST角点半径3的要求，则跳过
                    if(hX<=0)
                        continue;
                }

                // 当前金字塔图像层的网格区域图像
                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                // 重置当前网格特征点容器空间大小，预留了当前网格应该需要提取的特征点数的五倍
                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                // 调用openCV中的FAST算法提取特征点
                FAST(cellImage,             // 待计算FAST角点的图像
                     cellKeyPoints[i][j],   // 存储特征点容器
                     iniThFAST,             // FAST提取阈值
                     true);                 // 是否使用非极大值抑制

                // 如果当前网格提取的特征点个数小于3，使用最小阈值重新提取
                if(cellKeyPoints[i][j].size()<=3)
                {
                    // 清空特征点容器
                    cellKeyPoints[i][j].clear();
                    // 重新提取FAST角点
                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }

                // 当前网格提取到的特征点个数
                const int nKeys = cellKeyPoints[i][j].size();
                // 记录下来
                nTotal[i][j] = nKeys;

                // 如果提取的特征点数目满足每个网格应该提取的特征点数目
                if(nKeys>nfeaturesCell)
                {
                    // 就需要将保留的特征点数目设置成想要为这个网格提取的特征点数目
                    nToRetain[i][j] = nfeaturesCell;
                    // 该网格不需要更多特征点了
                    bNoMore[i][j] = false;
                }
                else
                {
                    // 如果没有满足，只能保留当前提取到的特征点数目
                    nToRetain[i][j] = nKeys;
                    // 需要分配到这里的特征点数目就是当前提取到的特征点数目和应该提取到的特征点数之差，可以理解为特征点个数的缺口
                    // 累加
                    nToDistribute += nfeaturesCell-nKeys;
                    // 置位true 表示这个网格需要的特征点数没有满足，需要更多特征点
                    bNoMore[i][j] = true;
                    // 计数++
                    nNoMore++;
                }

            }// 遍历图像中每一列Cell
        } // 遍历图像中每一行Cell


        // Retain by score
        // 根据评分决定哪个特征点会被留下
        // 进行的条件 1. nToDistribute 需要分配给特征点少的Cell的特征点数  2. nNoMore 不是所有的Cell都nomore了
        // 即，这个过程停止的条件一是没有需要再进行均匀分配的特征点了，二是当前循环时所有的特征点数目都不足以补充特征点个数的缺口了
        while(nToDistribute>0 && nNoMore<nCells)
        {
            // 对于那些在上一轮中达到特征点数要求的Cell，计算出本轮循环中，需要补足本层图像特征点总个数的缺口
            // 这些Cell所需要重新提取的“期望提取出来”的特征点数， nToDistribute 特征点数缺口 nCells-nNoMore 那些满足上一次提取的Cell个数
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            
            // 即使是补充，仍然会有一些Cell能满足第一轮提取的特征点数目要求，但是不能满足第二轮的要求（甚至第三轮、第四轮等等）
            // 需要将特征点数目缺口重新置为0，以便记录该轮过后仍需要补充的特征点数目
            nToDistribute = 0;

            // 按行遍历每个Cell
            for(int i=0; i<levelRows; i++)
            {
                // 按列遍历每个Cell
                for(int j=0; j<levelCols; j++)
                {
                    // 如果该Cell在上一轮满足要求，说明可以提取更多特征点
                    if(!bNoMore[i][j])
                    {
                        // 如果上一轮提取到的特征点数大于当前轮Cell需要提取的特征点数
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            // 满足要求，直接重新设置要保存的特征点数为当前轮Cell需要提取的特征点数
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            // 如果不能满足要求，只好将就，将上一轮提取到的特征点数保存
                            nToRetain[i][j] = nTotal[i][j];
                            // 仍需要分配的特征点数累加
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            // 这个Cell没有满足本轮
                            bNoMore[i][j] = true;
                            // 没有满足要求的Cell个数+1
                            nNoMore++;
                        }
                    } // 判断这个Cell是否满足要求 true不满足
                } // 按列遍历Cell
            } // 按行遍历Cell
        } // 判断是否达到了停止条件

        // 执行到这里，完成了每个图像Cell中特征点的提取 + 每个Cell中应该保留的特征点数
        // 下面才是正式开始对每个Cell中vector中的特征点进行删减（过滤）

        // 声明一个对当前金字塔图像层特征点的容器的引用
        vector<KeyPoint> & keypoints = allKeypoints[level];
        // 预分配两倍期望特征点数的大小 （因为实际提取过程中，都是按照稍微超过特征点数进行提取的
        keypoints.reserve(nDesiredFeatures*2);

        // 计算在本层图像的时候，图像PATCH块经过尺度缩放之后的大小（这里的缩放因子就是自己认为的缩放因子）
        // ? 这里的PATCH_SIZE用于进行什么操作 为什么要在当前层而不是底层图像上进行，是否与计算特征点方向有关？
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        // 按行遍历当前层图像Cell
        for(int i=0; i<levelRows; i++)
        {
            // 按列遍历当前层图像Cell
            for(int j=0; j<levelCols; j++)
            {
                // 声明一个对当前Cell所提取出的特征点容器的引用
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                // 调用openCV函数，根据特征点的评分（即响应值），保留每一个Cell中指定数量的特征点
                KeyPointsFilter::retainBest(keysCell,               // 输入输出，用于保留特征点的容器，操作完成之后将保留的特征点存放在里面，即将响应值低的特征点删除了
                                            nToRetain[i][j]);       // 需要保留的特征点数
                // 如果由于小数取整等原因（前面都是向多了取整的），对于当前的Cell，经过去除之后的特征点数还是大于设定的要保留的特征点数
                // 就强制丢弃后面的特征点以满足保留的特征数目要求
                if((int)keysCell.size()>nToRetain[i][j])
                    // ? KeyPointsFilter::retainBest() 函数完成特征点过滤之后是否 对vector中特征点按响应值大小进行了排序？否则直接丢弃后面的特征点，是否会丢到“一些好的”特征点
                    keysCell.resize(nToRetain[i][j]);

                // 遍历过滤后当前Cell的特征点容器，进行坐标的转换，以及添加相关信息
                // note 这里的特征点还只是在一个Cell中的，其坐标也是在Cell坐标下的，需要变换到当前图像层坐标下
                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    // 转换坐标（用到了之前保存的坐标）
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    // 特征点所在图像金字塔层数
                    keysCell[k].octave=level;
                    // 特征点所在图层PATCH的缩放大小
                    keysCell[k].size = scaledPatchSize;
                    // keypoints是 allkeypoints中存储图像金字塔层所有特征点当前层的一个引用
                    // 将转换之后的结果放入
                    keypoints.push_back(keysCell[k]);
                } // 遍历过滤后的Cell中的特征点 进行坐标转换 添加信息等
            } // 按列遍历Cell
        } // 按行遍历Cell

        // 如果经过上面的剔除操作后，当前层提取的关键点数目还是大于该层所期望提取的数目
        // 和上面不同，上面是判断每个Cell中的，这个是判断当前层图像中的
        if((int)keypoints.size()>nDesiredFeatures)
        {
            // 使用openCV的过滤函数，将特征点 按响应值大小进行过滤
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            // 保留所期望提取的特征点
            keypoints.resize(nDesiredFeatures);
        }
    } // 遍历每个图层

    // and compute orientations
    // 最后计算每层图像特征点的方向
    // 遍历图像金字塔中的每个图层
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

/**
 * @brief 全局静态函数 不属于该类的任何对象 static修饰符限定其只能被本文件中的函数调用
 *          计算某层金字塔图像的描述子
 * 
 * @param[in] image         某层金字塔图像
 * @param[in] keypoints     该层关键点
 * @param[in] descriptors   描述子信息
 * @param[in] pattern       256对匹配点
 */
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    // 清空保存描述子信息的容器
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    // 遍历所有关键点，计算ORB描述子
    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i],              // 关键点
                             image,                     // 图像
                             &pattern[0],               // 随机点集的首地址
                             descriptors.ptr((int)i));  // 提取出来的描述子保存地址
}

/**
 * @brief 用仿函数（通过重载括号运算符）来计算图像特征点
 * 
 * @param[in] _image            计算图像
 * @param[in] _mask             掩码mask
 * @param[in] _keypoints        关键点
 * @param[in] _descriptors      关键点对应描述子
 */
void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{ 
    // Step1 检查图像有效性
    if(_image.empty())
        return;

    // 获取图像大小
    Mat image = _image.getMat();
    // 判断图像格式是否正确，要求是单通道图像
    assert(image.type() == CV_8UC1 );

    // Pre-compute the scale pyramid
    // Step2 构建图像金字塔
    ComputePyramid(image);

    vector < vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

/**
 * @brief 构建图像金字塔
 * 
 * @param[in] image 输入图像，这个输入图像所有像素都是有效的，也就是说可以在其上提取出FAST角点的
 */
void ORBextractor::ComputePyramid(cv::Mat image)
{
    // 开始遍历所有图层
    for (int level = 0; level < nlevels; ++level)
    {
        // 获取本层图像缩放系数
        float scale = mvInvScaleFactor[level];
        // 计算本层图像像素尺寸大小
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        // 全尺寸图像，包括无效图像区域，将图像“补边”，EDGE_THRESHOLD=19=16+3 该区域外的图像不进行FAST角点检测
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        // 定义两个变量，temp是扩充了边界的图像，masktemp并未使用
        Mat temp(wholeSize, image.type()), masktemp;
        // mvImagePyramid 刚开始是个空的vector
        // 把图像金字塔该层的图像指针mvImagePyramid指向temp的中间部分（这里为浅拷贝，内存相同）
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        // 如果不是原始图像，即图像金字塔最下层图像，计算resized后的图像
        if( level != 0 )
        {
            // 将上一层金字塔图像根据设定sz缩放到当前层级
            resize(mvImagePyramid[level-1],     // 输入图像
                   mvImagePyramid[level],       // 输出图像
                   sz,                          // 输出图像的尺寸
                   0,                           // 水平方向上的系数，0表示自动计算
                   0,                           // 垂直方向上的系数，0表示自动计算
                   INTER_LINEAR);               // 图像缩放的差值算法类型，这里表示双线性插值

            //?  原代码mvImagePyramid 并未扩充，上面resize应该改为如下
            // resize(image,	                //输入图像
			// 	   mvImagePyramid[level], 	//输出图像
			// 	   sz, 						//输出图像的尺寸
			// 	   0, 						//水平方向上的缩放系数，留0表示自动计算
			// 	   0,  						//垂直方向上的缩放系数，留0表示自动计算
			// 	   cv::INTER_LINEAR);		//图像缩放的差值算法类型，这里的是线性插值算法

            // 把源图像拷贝到目的图像中央，四面填充指定的像素，如果图片已经拷贝到中间，只填充边界，这样做是为了正确提取FAST角点
            // EDGE_THRESHOLD=19 指的这个边界的宽度，由于这个边界之外的像素不是源图像而是算法生成出来的，所以不能在EDGE_THRESHOLD之外提取角点
            copyMakeBorder(mvImagePyramid[level],       // 源图像
                           temp,                        // 目标图像
                           EDGE_THRESHOLD, EDGE_THRESHOLD,      // top bottom 边界
                           EDGE_THRESHOLD, EDGE_THRESHOLD,      // left right 边界
                           BORDER_REFLECT_101+BORDER_ISOLATED);            // 扩充方式
                    /*Various border types, image boundaries are denoted with '|'
                    * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
                    * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
                    * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
                    * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
                    * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
                    */
                //    BORDER_ISOLATED	表示对整个图像进行操作
        }
        else
        {
            // 如果是第0层未缩放图像，直接将图像深拷贝到temp中间，并且对其周围进行边界扩充。此时temp就是对源图像扩展后的图像
            copyMakeBorder(image,       // 这是原图像
                           temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);      //? 并未扩充原图像      
        }
        // //? 原代码mvImagePyramid 并未扩充，应该添加下面一行代码
        // mvImagePyramid[level] = temp;
    }

}

} //namespace ORB_SLAM
