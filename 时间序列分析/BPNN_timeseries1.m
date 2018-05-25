% http://blog.sina.com.cn/s/blog_4dd4d3a5010009z9.html
clc;
clear;
close all;

array1=[17.6 17.7 17.7 17.7 17.8 17.8 17.9 18.0 18.1 18.2 18.4 18.6 ...
  18.7 18.9 19.1 19.3 19.6 19.9 20.2 20.6 21.0 21.5 22.0];
%������һ�������������22.7;
length=length(array1);
for i=1:(length-3)
  fortrain(i,:)=array1(i:i+3); %ѵ��������Ϊ4 
  if i~=length-3
    forresult(i)=array1(i+4); %���Ϊѵ��������������ֵ
  else
    t(i)=22.7;  %Ԥ������Ϊ22.7
  end
end

P=fortrain(1:length-4,:)';
T=forresult;
Ppre=fortrain(length-3,:)';
Tpre=-99999;

net=newff(P,T,[length-3 20 1],{'logsig' 'logsig' 'purelin'},'trainlm');
Tpre = sim(net,Ppre)

net.trainParam.show=10;  %��ʾѵ����������
net.trainParam.lr=0.0001;  %ѧϰ��
net.trainParam.epochs=400; %���ѵ������
net.trainParam.goal=1e-3; %ѵ��Ҫ�󾫶�
net = train(net,P,T); %����ѵ��

Tpre = sim(net,Ppre)