% http://blog.sina.com.cn/s/blog_4dd4d3a5010009z9.html
clc;
clear;
close all;

array1=[17.6 17.7 17.7 17.7 17.8 17.8 17.9 18.0 18.1 18.2 18.4 18.6 ...
  18.7 18.9 19.1 19.3 19.6 19.9 20.2 20.6 21.0 21.5 22.0];
%接着下一个的期望输出是22.7;
length=length(array1);
for i=1:(length-3)
  fortrain(i,:)=array1(i:i+3); %训练集长度为4 
  if i~=length-3
    forresult(i)=array1(i+4); %结果为训练集紧跟的样本值
  else
    t(i)=22.7;  %预测期望为22.7
  end
end

P=fortrain(1:length-4,:)';
T=forresult;
Ppre=fortrain(length-3,:)';
Tpre=-99999;

net=newff(P,T,[length-3 20 1],{'logsig' 'logsig' 'purelin'},'trainlm');
Tpre = sim(net,Ppre)

net.trainParam.show=10;  %显示训练迭代过程
net.trainParam.lr=0.0001;  %学习率
net.trainParam.epochs=400; %最大训练次数
net.trainParam.goal=1e-3; %训练要求精度
net = train(net,P,T); %网络训练

Tpre = sim(net,Ppre)