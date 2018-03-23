%% BP��������

% ���
clear all;
clc;

% ��������
load data;

%��1��2000���������
k=rand(1,2000);
[m,n]=sort(k);

%�����������
input=data(:,2:25);
output1 =data(:,1);

%�������1ά���4ά
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

%�����ȡ1500������Ϊѵ��������500������ΪԤ������
trainCharacter=input(n(1:1600),:);
trainOutput=output(n(1:1600),:);
testCharacter=input(n(1601:2000),:);
testOutput=output(n(1601:2000),:);

% ��ѵ�����������й�һ��
[trainInput,inputps]=mapminmax(trainCharacter');

%% �����ĳ�ʼ��

% �����ĳ�ʼ��
inputNum = 24;%�����Ľڵ���
hiddenNum = 50;%������Ľڵ���
outputNum = 4;%�����Ľڵ���

% Ȩ�غ�ƫ�õĳ�ʼ��
w1 = rands(inputNum,hiddenNum);
b1 = rands(hiddenNum,1);
w2 = rands(hiddenNum,outputNum);
b2 = rands(outputNum,1);

% ѧϰ��
yita = 0.1;

%% �����ѵ��
for r = 1:30
    E(r) = 0;% ͳ�����
    for m = 1:1600
        % ��Ϣ����������
        x = trainInput(:,m);
        % ����������
        for j = 1:hiddenNum
            hidden(j,:) = w1(:,j)'*x+b1(j,:);
            hiddenOutput(j,:) = g(hidden(j,:));
        end
        % ���������
        outputOutput = w2'*hiddenOutput+b2;
        
        % �������
        e = trainOutput(m,:)'-outputOutput;
        E(r) = E(r) + sum(abs(e));
        
        % �޸�Ȩ�غ�ƫ��
        % �����㵽������Ȩ�غ�ƫ�õ���
        dw2 = hiddenOutput*e';
        db2 = e;
        
        % ����㵽�������Ȩ�غ�ƫ�õ���
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

%% ���������źŷ���
testInput=mapminmax('apply',testCharacter',inputps);

for m = 1:400
    for j = 1:hiddenNum
        hiddenTest(j,:) = w1(:,j)'*testInput(:,m)+b1(j,:);
        hiddenTestOutput(j,:) = g(hiddenTest(j,:));
    end
    outputOfTest(:,m) = w2'*hiddenTestOutput+b2;
end

%% �������
%������������ҳ�������������
for m=1:400
    output_fore(m)=find(outputOfTest(:,m)==max(outputOfTest(:,m)));
end

%BP����Ԥ�����
error=output_fore-output1(n(1601:2000))';

k=zeros(1,4);  
%�ҳ��жϴ���ķ���������һ��
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

%�ҳ�ÿ��ĸ����
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

%��ȷ��
rightridio=(kk-k)./kk