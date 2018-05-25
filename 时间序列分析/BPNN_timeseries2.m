x=[54167

55196

56300

57482

58796

60266

61465

62828

64653

65994

67207

66207

65859

67295

69172

70499

72538

74542

76368

78534

80671

82992

85229

87177

89211

90859

92420

93717

94974

96259

97542

98705

100072

101654

103008

104357

105851

107507

109300

111026

112704

114333

115823

117171

118517

119850

121121

122389

123626

124761

125786

126743

127627

128453

129227

129988

130756

131448

132129

132802

134480

135030

135770

136460

137510]';

[x,ps]=mapminmax(x,0,1);

% 该脚本用来做NAR神经网络预测
% 作者：Macer程
% https://wenku.baidu.com/view/ea69dfc077232f60ddcca197.html

lag=3; % 自回归阶数
iinput=x; % x为原始序列（行向量）
n=length(iinput);

%准备输入和输出数据
inputs=zeros(n-lag,lag);
for i=1:n-lag
inputs(i,:)=iinput(i:i+lag-1)';
end

targets=x(lag+1:end)

%创建网络

P=inputs;

P=P';

T=targets;

net=newff(minmax(P),[10,1],{'logsig','purelin'},'trainlm');

inputWeights=net.IW{1,1} ;

inputbias=net.b{1};

layerWeights=net.LW{2,1} ;

layerbias=net.b{2,1} ;

% 避免过拟合，划分训练，测试和验

net.trainParam.show = 50;

net.trainParam.lr = 0.1;

net.trainParam.mc = 0.04;

net.trainParam.epochs = 500;

net.trainParam.goal = 1e-5;

%训练网络

[net,tr] = train(net,P,T);

%% 根据图表判断拟合好坏

yn=net(P);

errors=T-yn;

figure, ploterrcorr(errors)

figure, parcorr(errors)

%绘制偏相关情况

%[h,pValue,stat,cValue]=

lbqtest(errors)

%Ljung－Box Q检验（20lags）

figure,plotresponse(con2seq(targets),con2seq(yn))

%看预测的趋势与原趋势

%figure, ploterrhist(errors)

%误差直方图

%figure, plotperform(tr)

%误差下降线

%% 下面预测往后预测几个时间段

fn=7; %预测步数为fn。

f_in=iinput(n-lag+1:end)';

f_out=zeros(1,fn); %预测输出

% 多步预测时，用下面的循环将网络

for i=1:fn

f_out(i)=net(f_in);

f_in=[f_in(2:end);f_out(i)];

end

% 画出预测图

figure,plot(1949:2013,iinput,'b',2013:2020,[iinput(end),f_out],'r') 