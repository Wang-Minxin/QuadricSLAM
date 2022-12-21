#include <iostream>
#include <fstream>
#include <istream>
#include <vector>
#include <string>
#include <opencv/cv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>

using namespace std;
const string data_path = "../datasets/";
/*******************************文件读取操作*************************************/
/**
 * @brief 从文件中读取物体的BoundingBox
 * 
 * @param file_name 
 * @return Eigen::Vector4d x_min, y_min, width, height
 */
Eigen::Vector4d readBoundingBox(const string &file_name)
{
    Eigen::Vector4d vec = Eigen::Vector4d::Zero();
    ifstream ifile;
    ifile.open(data_path+file_name,ios::in);
    if(!ifile.good())
    {
        cout << "Can't open file : "<< data_path +file_name << endl;
        return vec;
    }
    string line;
    getline(ifile,line);
    istringstream line_stream(line);
    float tmp;
    line_stream >> tmp; // skip class id
    for(size_t i = 0; i < 4; i++)
    {
        line_stream >> tmp;
        vec[i] = tmp;
    }
    ifile.close();
    return vec;
}
/**
 * @brief 读取相机的真实位姿
 * 
 * @param file_name 
 * @return vector<Eigen::Matrix4d> 
 */
vector<Eigen::Matrix4d> readCameraPose(const string &file_name)
{
    vector<Eigen::Matrix4d> Twcs;
    ifstream ifile(data_path+file_name,ios::in);
    if(!ifile.good())
    {
        cout << "Can't open file : "<< data_path +file_name << endl;
        return Twcs;
    }

    string line;
    getline(ifile,line); // skip the annotation;
    while(!ifile.eof())
    {
        getline(ifile,line);
        istringstream line_stream(line);
        float tmp;
        line_stream >> tmp;
        vector<float> datas;
        while(line_stream>>tmp)
        {
            datas.push_back(tmp);
        }
        Eigen::Quaterniond quat(datas[6],datas[3],datas[4],datas[5]);
        quat.normalize();
        Eigen::Matrix3d R = quat.toRotationMatrix();
        Eigen::Vector3d t(datas[0],datas[1],datas[2]);
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block<3,3>(0,0) = R;
        Twc.block<3,1>(0,3) = t;
        Twcs.push_back(Twc);
    }
    ifile.close();
    return Twcs;
}

/*******************************恢复物体参数*************************************/
/**
 * @brief 得到直线
*/
Eigen::MatrixXd fromDetectionsToLines(Eigen::Vector4d &detections)
{

    double x1 = detections(0);
    double y1 = detections(1);
    double x2 = detections(2);
    double y2 = detections(3);

    Eigen::Vector3d line1(1,0,-x1);
    Eigen::Vector3d line2(0,1,-y1);
    Eigen::Vector3d line3(1,0,-x2);
    Eigen::Vector3d line4(0,1,-y2);


    Eigen::MatrixXd line_selected(3, 0);
    line_selected.conservativeResize(3, line_selected.cols()+1);
    line_selected.col(line_selected.cols()-1) = line1;   

    line_selected.conservativeResize(3, line_selected.cols()+1);
    line_selected.col(line_selected.cols()-1) = line2; 

    line_selected.conservativeResize(3, line_selected.cols()+1);
    line_selected.col(line_selected.cols()-1) = line3;

    line_selected.conservativeResize(3, line_selected.cols()+1);
    line_selected.col(line_selected.cols()-1) = line4;

    return line_selected;
}
/**
 * @brief Get the Tangent Planes object
 * 
 * @param K 相机内参
 * @param bboxs x_min,y_min,width,height
 * @param Twcs 相机位姿
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd getTangentPlanes(const Eigen::Matrix3d &K, Eigen::MatrixXd &bboxs, vector<Eigen::Matrix4d> &Twcs)
{
    Eigen::MatrixXd planes(4,0);
    for(size_t i = 0; i < bboxs.cols(); i++)
    {
        Eigen::Vector4d box = bboxs.col(i);
        float x1 = box[0], y1 = box[1];
        float x2 = box[2], y2 = box[3];
        Eigen::Vector4d detection(x1,y1,x2,y2);
        Eigen::MatrixXd lines = fromDetectionsToLines(detection);

        // 投影矩阵
        Eigen::Matrix4d Twc = Twcs[i];
        Eigen::Matrix3d Rwc = Twc.block<3,3>(0,0);
        Eigen::Matrix3d Rcw = Rwc.transpose();
        Eigen::Vector3d twc = Twc.block<3,1>(0,3);
        Eigen::Vector3d tcw = -Rcw * twc;
        Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
        Tcw.block<3,3>(0,0) = Rcw;
        Tcw.block<3,1>(0,3) = tcw;
        Eigen::Matrix3Xd P;
        Eigen::Matrix3Xd identity3x4;
        identity3x4.resize(3,4);
        identity3x4.col(3) = Eigen::Vector3d(0,0,0);
        identity3x4.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity(3,3);
        P = K * identity3x4 * Tcw;

        // 恢复平面
        Eigen::MatrixXd plane = P.transpose() * lines;
        for(int m = 0; m < plane.cols(); m++)
        {
            planes.conservativeResize(planes.rows(), planes.cols()+1);
            planes.col(planes.cols() -1) = plane.col(m);
        }
    }

    // 平面转化为向量
    Eigen::MatrixXd plane_vector(0,10);
    int cols = planes.cols();
    for(size_t i = 0; i < cols; i++)
    {
        Eigen::VectorXd p = planes.col(i);
        Eigen::Vector<double,10> v;
        v << p(0)*p(0),2*p(0)*p(1),2*p(0)*p(2),2*p(0)*p(3),p(1)*p(1),2*p(1)*p(2),2*p(1)*p(3),p(2)*p(2),2*p(2)*p(3),p(3)*p(3);
        plane_vector.conservativeResize(plane_vector.rows()+1, plane_vector.cols());
        plane_vector.row(plane_vector.rows()-1) = v;
    }
    return plane_vector;
}

void getObject(const Eigen::MatrixXd &planes, Eigen::Matrix3d &R, Eigen::Vector3d &t, Eigen::Vector3d &s)
{
    Eigen::MatrixXd A = planes;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV );
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd qj_hat = V.col(V.cols()-1);
    qj_hat = qj_hat/(-qj_hat(9));
    // Get QStar
    Eigen::Matrix4d QStar;
    QStar <<
        qj_hat(0),qj_hat(1),qj_hat(2),qj_hat(3),
        qj_hat(1),qj_hat(4),qj_hat(5),qj_hat(6),
        qj_hat(2),qj_hat(5),qj_hat(7),qj_hat(8),
        qj_hat(3),qj_hat(6),qj_hat(8),qj_hat(9);
    Eigen::Matrix4d Q = QStar.inverse() * cbrt(QStar.determinant());
    Q = Q/(-Q(3,3));
    Eigen::Matrix3d Q33 = Q.block<3,3>(0,0);
    double k = Q.determinant()/ Q33.determinant();
    t = QStar.block<3,1>(0,3)/QStar(3,3);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Q33);
    R = es.eigenvectors();
    Eigen::Vector3d lambda = es.eigenvalues().array().inverse();
    s = (-k * lambda).array().abs().sqrt();
}
void getORBPoints(vector<Eigen::Vector3d> &pts, const Eigen::Matrix3d &K, const Eigen::MatrixXd &bbox, vector<Eigen::Matrix4d> &Twcs)
{
    int N = Twcs.size();
    double fx = K(0,0), fy = K(1,1);
    double cx = K(0,2), cy = K(1,2);
    for(size_t i = 0; i < N; i++)
    {
        Eigen::Matrix4d Twc = Twcs[i];
        Eigen::Vector4d box = bbox.col(i);
        int x1 = (int)(box(0)), y1 = (int)(box(1));
        int x2 = x1 + (int)(box(2)),y2 = y1 +(int)(box(3));
        // 加载图像
        string img_name = to_string(i+1)+"_depth.png";
        cv::Mat depthImg = cv::imread(data_path+img_name,cv::IMREAD_UNCHANGED);
        depthImg.convertTo(depthImg,CV_32F,1/5000.0);
        for(size_t i = x1; i < x2; i+=3)
        {
            for(size_t j = y1; j < y2; j+=3)
            {
                float z = depthImg.at<float>(j,i);
                if(z == 0.0 || z > 2.0)
                    continue;
                float x = z*(i-cx)/fx;
                float y = z*(j-cy)/fy;
                Eigen::Vector4d pt_c(x,y,z,1);
                Eigen::Vector4d pt_w = Twc * pt_c;
                Eigen::Vector3d pt;
                pt << pt_w(0),pt_w(1),pt_w(2);
                pts.push_back(pt);
            }
        }
    }
}
int main()
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    const int N = 4;
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K << 535.5, 0.0, 320.1,
         0.0  , 539.2, 247.6,
         0.0  , 0.0,   1.0;
    // 读取bounding box
    Eigen::MatrixXd bboxs(4,0);
    for(size_t i = 0; i < N; i++)
    {
        string file_name = to_string(i+1) + ".txt";
        Eigen::Vector4d bbox = readBoundingBox(file_name);
        bboxs.conservativeResize(bboxs.rows(),bboxs.cols()+1);
        bboxs.col(bboxs.cols()-1) = bbox;
    }

    // 读取相机位姿
    vector<Eigen::Matrix4d> Twcs;
    const string file_name = "groundtruth.txt";
    Twcs = readCameraPose(file_name);

    Eigen::MatrixXd planes = getTangentPlanes(K,bboxs,Twcs);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t;
    Eigen::Vector3d s;
    getObject(planes,R,t,s);
    Eigen::Matrix4d Two = Eigen::Matrix4d::Identity();
    Two.block<3,3>(0,0) = R;
    Two.block<3,1>(0,3) = t;
    // 提取地图点
    vector<Eigen::Vector3d> points;
    getORBPoints(points,K,bboxs,Twcs);

    // 绘图
    pangolin::CreateWindowAndBind("My Window");
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 100), 
        pangolin::ModelViewLookAt( 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    );

    pangolin::Handler3D hander(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f).SetHandler(&hander);
    while (! pangolin::ShouldQuit())
    {
        glClearColor(1.0,1.0,1.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        // 绘制坐标
        glLineWidth(5.0);
        glBegin(GL_LINES);
        glColor3f(1.0,0.0,0.0);     // x
        glVertex3f(0.0,0.0,0.0);
        glVertex3f(1.0,0.0,0.0);
        glColor3f(0.0,1.0,0.0);     // y
        glVertex3f(0.0,0.0,0.0);
        glVertex3f(0.0,1.0,0.0);
        glColor3f(0.0,0.0,1.0);     // z
        glVertex3f(0.0,0.0,0.0);
        glVertex3f(0.0,0.0,1.0);
        glEnd();

        // 绘制地图点
        glPointSize(2.0);
        glColor3f(1.0,0.0,0.0);
        glBegin(GL_POINTS);
        for(size_t i = 0; i < points.size(); i++)
        {
            glVertex3f(points[i].x(),points[i].y(),points[i].z());
        }
        glEnd();
        // 绘制椭球
        glColor3f(0.0,1.0,0.0);
        glLineWidth(2.0);
        GLUquadricObj *pObj = gluNewQuadric();
        glPushMatrix();
        glMultMatrixd(Two.data());
        glScalef( (GLfloat)(s(0)), (GLfloat)(s(1)),(GLfloat)(s(1)));
        gluQuadricDrawStyle(pObj,GLU_LINE);
        gluQuadricNormals(pObj,GLU_NONE);
        glBegin(GL_COMPILE);
        gluSphere(pObj,1.0,15,10);
        glEnd();
        glPopMatrix();

        pangolin::FinishFrame();
    }
    return 0;
}

