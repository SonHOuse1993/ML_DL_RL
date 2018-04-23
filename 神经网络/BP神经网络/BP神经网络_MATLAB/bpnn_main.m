%% BP的主函数

% 清空
clear all;
clc;

% 导入数据
load data;

%从1到2000间随机排序
k=rand(1,2000);
[m,n]=sort(k);

%输入输出数据
input=data(:,2:25);
output1 =data(:,1);

%把输出从1维变成4维
for i=1:2000
    switch output1(i)
        case 1
            output(i,:)=[1 0 0 0];
        case 2
            output(i,:)=[0 1 0 0];
        case 3
            output(i,:)=[0 0 1 0];
        case 4
            output(i,:)=[0 0 0 1];
    end
end

%随机提取1500个样本为训练样本，500个样本为预测样本
trainCharacter=input(n(1:1600),:);
trainOutput=output(n(1:1600),:);
testCharacter=input(n(1601:2000),:);
testOutput=output(n(1601:2000),:);

% 对训练的特征进行归一化
[trainInput,inputps]=mapminmax(trainCharacter');

%% 参数的初始化

% 参数的初始化
inputNum = 24;%输入层的节点数
hiddenNum = 50;%隐含层的节点数
outputNum = 4;%输出层的节点数

% 权重和偏置的初始化
w1 = rands(inputNum,hiddenNum);
b1 = rands(hiddenNum,1);
w2 = rands(hiddenNum,outputNum);
b2 = rands(outputNum,1);

% 学习率
yita = 0.1;

%% 网络的训练
for r = 1:30
    E(r) = 0;% 统计误差
    for m = 1:1600
        % 信息的正向流动
        x = trainInput(:,m);
        % 隐含层的输出
        for j = 1:hiddenNum
            hidden(j,:) = w1(:,j)'*x+b1(j,:);
            hiddenOutput(j,:) = g(hidden(j,:));
        end
        % 输出层的输出
        outputOutput = w2'*hiddenOutput+b2;
        
        % 计算误差
        e = trainOutput(m,:)'-outputOutput;
        E(r) = E(r) + sum(abs(e));
        
        % 修改权重和偏置
        % 隐含层到输出层的权重和偏置调整
        dw2 = hiddenOutput*e';
        db2 = e;
        
        % 输入层到隐含层的权重和偏置调整
        for j = 1:hiddenNum
            partOne(j) = hiddenOutput(j)*(1-hiddenOutput(j));
            partTwo(j) = w2(j,:)*e;
        end
        
        for i = 1:inputNum
            for j = 1:hiddenNum
                dw1(i,j) = partOne(j)*x(i,:)*partTwo(j);
                db1(j,:) = partOne(j)*partTwo(j);
            end
        end
        
        w1 = w1 + yita*dw1;
        w2 = w2 + yita*dw2;
        b1 = b1 + yita*db1;
        b2 = b2 + yita*db2;  
    end
end

%% 语音特征信号分类
testInput=mapminmax('apply',testCharacter',inputps);

for m = 1:400
    for j = 1:hiddenNum
        hiddenTest(j,:) = w1(:,j)'*testInput(:,m)+b1(j,:);
        hiddenTestOutput(j,:) = g(hiddenTest(j,:));
    end
    outputOfTest(:,m) = w2'*hiddenTestOutput+b2;
end

%% 结果分析
%根据网络输出找出数据属于哪类
for m=1:400
    output_fore(m)=find(outputOfTest(:,m)==max(outputOfTest(:,m)));
end

%BP网络预测误差
error=output_fore-output1(n(1601:2000))';

k=zeros(1,4);  
%找出判断错误的分类属于哪一类
for i=1:400
    if error(i)~=0
        [b,c]=max(testOutput(i,:));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
            case 3 
                k(3)=k(3)+1;
            case 4 
                k(4)=k(4)+1;
        end
    end
end

%找出每类的个体和
kk=zeros(1,4);
for i=1:400
    [b,c]=max(testOutput(i,:));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
        case 3
            kk(3)=kk(3)+1;
        case 4
            kk(4)=kk(4)+1;
    end
end

%正确率
rightridio=(kk-k)./kk