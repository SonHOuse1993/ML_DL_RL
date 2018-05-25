//作者：潘智渊
//日期：2018-03-26

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "memory.h"

class BpNet
{
private:
    int _nInput;            //输入层节点个数
    int _nHide;             //隐含层节点个数
    int _nOutput;           //输出层节点个数

    double **_pplfWeight1;  //输入层-隐含层权系数
    double **_pplfWeight2;  //隐含层-输出层权系数

    double *_plfb1;            //隐含层结点的阈值
    double *_plfb2;            //输出层结点的阈值

    double *_plfHideIn, *_plfHideOut;       //隐含层的网络输入和输出
    double *_plfOutputIn, *_plfOutputOut;   //输出层的网络输入和输出

    double _a;  //学习率
    double _e;  //目标误差

private:

    static double sigmoid(double x);

    double(*f)(double);    //激活函数

    void Init();

public:

    BpNet(int nInput, int nHide, int nOutput);

    virtual ~BpNet();

    void Set(double a, double e);

    void GetBasicInformation(int &nInput, int &nHide, int &nOutput, double &lfA, double &lfE);

    bool Train(int n, double **pplfInput, double **pplfDesire);

    void Classify(double plfInput[], double plfOutput[]);
};

void showInfo(int n, int nInput, int nHide, int nOutput, double lfA, double lfE, double **ppInput, double **ppOutput)
{
    //n代表学习样本的个数，学习样本的维数为nInput
    printf("输入层节点数：%d； 隐层节点数：%d； 输出层节点数：%d\n", nInput, nHide, nOutput);
    printf("学习因子：%lf; 最大允许误差：%lf\n", lfA, lfE);
    printf("学习样本为：\n");
    printf("输入：\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < nInput; j++) printf("%8.5lf", ppInput[i][j]);
        printf("\n");
    }
    printf("期望输出：\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < nOutput; j++) printf("%8.5lf", ppOutput[i][j]);
        printf("\n");
    }
}

BpNet::BpNet(int nInput, int nHide, int nOutput)
{
    _a = 0.2;
    _e = 0.01;

    _nInput = nInput;
    _nHide = nHide;
    _nOutput = nOutput;

    Init();

    srand((unsigned)time(NULL));
    //初始化输入层-隐含层权系数
    for (int i = 0; i < _nInput; i++)
        for (int j = 0; j < _nHide; j++)
            //使用随机数生成-0.1~0.09999的随机系数矩阵
            //_pplfWeight1[i][j] = (double)(rand() % 20000 - 10000) / 100000;
            //使用同一值系数矩阵
            _pplfWeight1[i][j] = 0.1;
    //初始化隐含层-输出层权系数
    for (int i = 0; i < _nHide; i++)
        for (int j = 0; j < _nOutput; j++)
            //使用随机数生成-0.1~0.09999的随机系数矩阵
            //_pplfWeight2[i][j] = (double)(rand() % 20000 - 10000) / 100000;
            //使用同一值系数矩阵
            _pplfWeight2[i][j] = 0.1;
    //初始化隐含层的阈值
    for (int i = 0; i < _nHide; i++)
        //使用随机数生成-0.1~0.09999的随机系数
        //_plfb1[i] = (double)(rand() % 20000 - 10000) / 100000;
        //使用同一值系数
        _plfb1[i] = 0.05;
    //初始化输出层的阈值
    for (int i = 0; i < _nOutput; i++)
        //使用随机数生成-0.1~0.09999的随机系数
        //_plfb2[i] = (double)(rand() % 20000 - 10000) / 100000;
        //使用同一值系数
        _plfb2[i] = 0.05;
    //使用sigmoid激活函数
    f = sigmoid;
}

BpNet::~BpNet()
{
    delete[]_plfHideIn;
    delete[]_plfHideOut;
    delete[]_plfOutputIn;
    delete[]_plfOutputOut;
    delete[]_plfb1;
    delete[]_plfb2;

    for (int i = 0; i < _nInput; i++) delete[]_pplfWeight1[i];
    for (int i = 0; i < _nHide; i++) delete[]_pplfWeight2[i];
    delete[]_pplfWeight1;
    delete[]_pplfWeight2;
}

void BpNet::Init()
{
    _pplfWeight1 = new double *[_nInput];
    for (int i = 0; i < _nInput; i++) _pplfWeight1[i] = new double[_nHide];

    _pplfWeight2 = new double *[_nHide];
    for (int i = 0; i < _nHide; i++) _pplfWeight2[i] = new double[_nOutput];

    _plfb1 = new double[_nHide];
    _plfb2 = new double[_nOutput];


    _plfHideIn = new double[_nHide];
    _plfHideOut = new double[_nHide];
    _plfOutputIn = new double[_nOutput];
    _plfOutputOut = new double[_nOutput];
}

void BpNet::Set(double a, double e)
{
    _a = a;
    _e = e;
}

void BpNet::GetBasicInformation(int &nInput, int &nHide, int &nOutput, double &lfA, double &lfE)
{
    nInput = _nInput;
    nHide = _nHide;
    nOutput = _nOutput;
    lfA = _a;
    lfE = _e;
}

double BpNet::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

void BpNet::Classify(double plfInput[], double plfOutput[])
{
    memset(_plfHideIn, 0, sizeof(double)* _nHide);
    memset(_plfHideOut, 0, sizeof(double)* _nHide);
    memset(_plfOutputIn, 0, sizeof(double)* _nOutput);
    memset(_plfOutputOut, 0, sizeof(double)* _nOutput);

    //输入层到隐含层的正向传播
    for (int i = 0; i < _nInput; i++)
        for (int j = 0; j < _nHide; j++)
            _plfHideIn[j] += plfInput[i] * _pplfWeight1[i][j];
    for (int j = 0; j < _nHide; j++) _plfHideOut[j] = (*f)(_plfHideIn[j] + _plfb1[j]);

    //隐含层到输出层的正向传播
    for (int j = 0; j < _nHide; j++)
        for (int k = 0; k < _nOutput; k++)
            _plfOutputIn[k] += _plfHideOut[j] * _pplfWeight2[j][k];
    for (int k = 0; k < _nOutput; k++) _plfOutputOut[k] = (*f)(_plfOutputIn[k] + _plfb2[k]);

    if (plfOutput != NULL)
        memcpy(plfOutput, _plfOutputOut, sizeof(double)* _nOutput);
}

bool BpNet::Train(int n, double **pplfInput, double **pplfDesire)
{
    double lfE = _e + 1;

    //输入层-隐含层权系数增量
    double **pplfDeltaWeight1 = new double *[_nInput];
    for (int i = 0; i < _nInput; i++)
    {
        pplfDeltaWeight1[i] = new double[_nHide];
        memset(pplfDeltaWeight1[i], 0, sizeof(double)* _nHide);
    }

    //隐含层-输出层权系数增量
    double **pplfDeltaWeight2 = new double *[_nHide];
    for (int i = 0; i < _nHide; i++)
    {
        pplfDeltaWeight2[i] = new double[_nOutput];
        memset(pplfDeltaWeight2[i], 0, sizeof(double)* _nOutput);
    }

    //隐含层的阈值增量
    double *plfDeltaBias1 = new double[_nHide];
    memset(plfDeltaBias1, 0, sizeof(double)*_nHide);

    //输出层的阈值增量
    double *plfDeltaBias2 = new double[_nOutput];
    memset(plfDeltaBias2, 0, sizeof(double)*_nOutput);


    long nCount = 0;
    while (lfE > _e)
    {
        lfE = 0;
        //对每一个样本进行处理
        for (int i = 0; i < n; i++)
        {
            double *plfInput = pplfInput[i];        //样本输入
            double *plfDesire = pplfDesire[i];      //样本期望输出

            //计算样本实际输出plfOutput
            Classify(plfInput, NULL);

            //计算误差测度
            double lfEp = 0;
            for (int j = 0; j < _nOutput; j++)
                lfEp += (plfDesire[j] - _plfOutputOut[j]) * (plfDesire[j] - _plfOutputOut[j]) / 2;
            lfE += lfEp;

            //计算隐含层-输出层权系数增量
            double *plfChange2 = new double[_nOutput];

            for (int j = 0; j < _nOutput; j++)
                plfChange2[j] = _plfOutputOut[j] * (1 - _plfOutputOut[j]) * (plfDesire[j] - _plfOutputOut[j]);
            for (int j = 0; j < _nHide; j++)
                for (int k = 0; k < _nOutput; k++)
                    pplfDeltaWeight2[j][k] = _a * _plfHideOut[j] * plfChange2[k];
            for (int k = 0; k < _nOutput; k++)
                plfDeltaBias2[k] = _a*plfChange2[k];

            //计算输入层-隐含层权系数增量
            double *plfChange1 = new double[_nHide];
            memset(plfChange1, 0, sizeof(double)* _nHide);
            for (int j = 0; j < _nHide; j++)
            {
                for (int k = 0; k < _nOutput; k++)
                    plfChange1[j] += _pplfWeight2[j][k] * plfChange2[k];
                plfChange1[j] *= _plfHideOut[j] * (1 - _plfHideOut[j]);
            }
            for (int j = 0; j < _nInput; j++)
                for (int k = 0; k < _nHide; k++)
                    pplfDeltaWeight1[j][k] = _a * plfInput[j] * plfChange1[k];
            for (int k = 0; k < _nHide; k++)
                plfDeltaBias1[k] = _a*plfChange1[k];

            delete[]plfChange1;
            delete[]plfChange2;

            //更新Bp网络权值
            for (int i = 0; i < _nInput; i++)
                for (int j = 0; j < _nHide; j++)
                    _pplfWeight1[i][j] += pplfDeltaWeight1[i][j];

            for (int i = 0; i < _nHide; i++)
                for (int j = 0; j < _nOutput; j++)
                    _pplfWeight2[i][j] += pplfDeltaWeight2[i][j];

            //更新BP网络的阈值
            for (int i = 0; i < _nOutput; i++)
                _plfb2[i] += plfDeltaBias2[i];

            for (int i = 0; i < _nHide; i++)
                _plfb1[i] += plfDeltaBias1[i];
        }
        nCount++;
        if (nCount % 1000 == 0) printf("第%d次迭代的误差为：%lf\n", nCount,lfE);
        if (nCount >= 1000000) break;
    }

    for (int i = 0; i < _nInput; i++) delete[]pplfDeltaWeight1[i];
    for (int i = 0; i < _nHide; i++) delete[]pplfDeltaWeight2[i];
    delete[] pplfDeltaWeight1;
    delete[] pplfDeltaWeight2;

    delete[] plfDeltaBias1;
    delete[] plfDeltaBias2;

    printf("迭代在 %ld 步后收敛\n", nCount);

    return true;
}

int main()
{
    int n = 10;                                         //共10天（组）的数据
    int nInput = 15; int nHide = 15; int nOutput = 12;    //隐含层满足kolmogorov定理
    double lfA = 0.2;                                    //**********************学习因子***************************
    double lfE = 0.01;                                    //********************最大允许误差*************************
    double Input[10][15] = {
            0.2452, 0.1466, 0.1314, 0.2243, 0.5523, 0.6642, 0.7015, 0.6981, 0.6821, 0.6945, 0.7549, 0.8215, 0.2415, 0.3027, 0,
            0.2217, 0.1581, 0.1408, 0.2304, 0.5134, 0.5312, 0.6819, 0.7125, 0.7265, 0.6847, 0.7826, 0.8325, 0.2385, 0.3125, 0,
            0.2525, 0.1627, 0.1507, 0.2406, 0.5502, 0.5636, 0.7051, 0.7352, 0.7459, 0.7015, 0.8064, 0.8156, 0.2216, 0.2701, 1,
            0.2016, 0.1105, 0.1243, 0.1987, 0.5021, 0.5232, 0.6819, 0.6952, 0.7015, 0.6825, 0.7825, 0.7895, 0.2352, 0.2506, 0.5,
            0.2115, 0.1201, 0.1312, 0.2019, 0.5532, 0.5736, 0.7029, 0.7032, 0.7189, 0.7019, 0.7965, 0.8025, 0.2542, 0.3125, 0,
            0.2335, 0.1322, 0.1534, 0.2214, 0.5662, 0.5827, 0.7198, 0.7176, 0.7359, 0.7506, 0.8092, 0.8221, 0.2601, 0.3198, 0,
            0.2368, 0.1432, 0.1653, 0.2205, 0.5823, 0.5971, 0.7136, 0.7129, 0.7263, 0.7153, 0.8091, 0.8217, 0.2579, 0.3099, 0,
            0.2342, 0.1368, 0.1602, 0.2131, 0.5726, 0.5822, 0.7101, 0.7098, 0.7127, 0.7121, 0.7995, 0.8126, 0.2301, 0.2867, 0,
            0.2113, 0.1212, 0.1305, 0.1819, 0.4952, 0.5312, 0.6886, 0.6898, 0.6999, 0.7323, 0.7721, 0.7956, 0.2234, 0.2799, 1,
            0.2005, 0.1121, 0.1207, 0.1605, 0.4556, 0.5022, 0.6553, 0.6673, 0.6798, 0.7023, 0.7521, 0.7756, 0.2314, 0.2977, 0 };

    double Output[10][12] = {
            0.2217, 0.1581, 0.1408, 0.2304, 0.5134, 0.5312, 0.6819, 0.7125, 0.7265, 0.6847, 0.7826, 0.8325,
            0.2525, 0.1627, 0.1507, 0.2406, 0.5502, 0.5636, 0.7051, 0.7352, 0.7459, 0.7015, 0.8064, 0.8156,
            0.2016, 0.1105, 0.1243, 0.1987, 0.5021, 0.5232, 0.6819, 0.6952, 0.7015, 0.6825, 0.7825, 0.7895,
            0.2115, 0.1201, 0.1312, 0.2019, 0.5532, 0.5736, 0.7029, 0.7032, 0.7189, 0.7019, 0.7965, 0.8025,
            0.2335, 0.1322, 0.1534, 0.2214, 0.5662, 0.5827, 0.7198, 0.7176, 0.7359, 0.7506, 0.8092, 0.8221,
            0.2368, 0.1432, 0.1653, 0.2205, 0.5823, 0.5971, 0.7136, 0.7129, 0.7263, 0.7153, 0.8091, 0.8217,
            0.2342, 0.1368, 0.1602, 0.2131, 0.5726, 0.5822, 0.7101, 0.7098, 0.7127, 0.7121, 0.7995, 0.8126,
            0.2113, 0.1212, 0.1305, 0.1819, 0.4952, 0.5312, 0.6886, 0.6898, 0.6999, 0.7323, 0.7721, 0.7956,
            0.2005, 0.1121, 0.1207, 0.1605, 0.4556, 0.5022, 0.6553, 0.6673, 0.6798, 0.7023, 0.7521, 0.7756,
            0.2123, 0.1257, 0.1343, 0.2079, 0.5579, 0.5716, 0.7059, 0.7145, 0.7205, 0.7401, 0.8019, 0.8316 };

////////////////////////////////////////////////////////////////////////
//将数据存入数组
    double **ppInput = new double *[n];
    double **ppOutput = new double *[n];


    for (int i = 0; i < n; i++)
    {
        ppInput[i] = new double[nInput];
        ppOutput[i] = new double[nOutput];
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nInput; j++)
            ppInput[i][j] = Input[i][j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nOutput; j++)
            ppOutput[i][j] = Output[i][j];
////////////////////////////////////////////////////////////////////////
//在控制台上显示信息
    showInfo(n, nInput, nHide, nOutput, lfA, lfE, ppInput, ppOutput);
////////////////////////////////////////////////////////////////////////
//初始化BP网络类并设置学习因子和最大允许误差
    BpNet bpNet(nInput, nHide, nOutput);
    bpNet.Set(lfA, lfE);
    bpNet.Train(n, ppInput, ppOutput);

    for (int i = 0; i < n; i++)
    {
        delete[]ppInput[i];
        delete[]ppOutput[i];
    }
    delete[]ppInput;
    delete[]ppOutput;
////////////////////////////////////////////////////////////////////////
//训练成功，测验
    printf("训练完毕，输入检索数据：\n");

    double * pInput = new double[nInput];
    double * pOutput = new double[nOutput];


    printf("输入：\n");
    //for (int i = 0; i < nInput; i++) scanf_s("%lf", &pInput[i]);            //scanf被scanf_s替换
    //0.2123 0.1257 0.1343 0.2079 0.5579 0.5716 0.7059 0.7145 0.7205 0.7401 0.8019 0.8316 0.2317 0.2936 0
    pInput[0] = 0.2123;
    pInput[1] = 0.1257;
    pInput[2] = 0.1343;
    pInput[3] = 0.2079;
    pInput[4] = 0.5579;
    pInput[5] = 0.5716;
    pInput[6] = 0.7059;
    pInput[7] = 0.7145;
    pInput[8] = 0.7205;
    pInput[9] = 0.7401;
    pInput[10] = 0.8019;
    pInput[11] = 0.8316;
    pInput[12] = 0.2317;
    pInput[13] = 0.2936;
    pInput[14] = 0;
    for (int i = 0; i < nInput; i++) printf("%8.5lf", pInput[i]);


    bpNet.Classify(pInput, pOutput);

    printf("\n输出：\n");
    for (int i = 0; i < nOutput; i++) printf("%8.5lf", pOutput[i]);

    double data[12] = { 0.2119, 0.1215, 0.1621, 0.2161, 0.6171, 0.6159, 0.7115, 0.7201, 0.7243, 0.7298, 0.8179, 0.8229 };
    printf("\n实际数据：\n");
    for (int i = 0; i < nOutput; i++) printf("%8.5lf", data[i]);

    printf("\n误差：\n");
    for (int i = 0; i < nOutput; i++)
        printf("%8.5lf",data[i] - pOutput[i]);
    printf("\n平均误差：");
    double ave = 0;
    for (int i = 0; i < nOutput; i++)
    {
        ave = ave + fabs(data[i] - pOutput[i]);
        if (i == (nOutput - 1))
            printf("%8.5lf", ave / nOutput);
    }
    return 0;
}

