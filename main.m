%% 此程序为基于灰狼优化算法优化的BP神经网络
% 清空环境变量
clear
close all
clc
warning off

%% 导入数据
% 读取输入的自变量
P = xlsread('训练和测试集的全部数据.xlsx','Sheet1','A2:B2000')';
% 读取输出的因变量
T =  xlsread('训练和测试集的全部数据.xlsx','Sheet1','C2:C2000')';

% 设定训练和测试的样本个数
N = 50;  % 设定50个数据测试
M = size(T, 2) - N;  % 其余的样本用于训练

% 随机抽取训练集测试集
indx = randperm(size(T, 2) );
P = P(:,indx);
T = T(:, indx);

%训练集
input_train = P(:, 1:M);
output_train = T(:, 1:M);
% 测试集
input_test= P(:, M+1:M+N);
output_test= T(:, M+1:M+N);

%% 归一化
% 训练集
[inputn,inputps] = mapminmax(input_train,-1,1);
inputn_test = mapminmax('apply',input_test,inputps);
% 测试集
[outputn,outputps] = mapminmax(output_train,-1,1);
outputn_test = mapminmax('apply',output_test,outputps);

%% 定义优化参数
%节点个数
inputnum=size(input_train,1);%输入数量
hiddennum=10;
outputnum=size(output_train,1);

%构建网络
net=newff(inputn,outputn,hiddennum);
%网络参数
net.trainParam.epochs=1000;   % 最大训练代数
net.trainParam.lr=0.01;   % 学习率
net.trainParam.goal=0.00001;  % 目标误差
net.trainParam.show=100;   % 显示频率

% 参数初始化
dim=inputnum * hiddennum + hiddennum*outputnum + hiddennum + outputnum;
Max_iteration=20;   % 迭代次数  % 一般取30-100  为了加快优化，设置小一些
pop=10;  %种群规模  % 一般取10-50
lb=-3;  %权值阈值下边界
ub=3;   %权值阈值上边界
fobj = @(x) fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
[Leader_score,Leader_pos,Convergence_curve]=GWO(pop,Max_iteration,lb,ub,dim,fobj); %调用子函数，开始优化
% 结束优化，作出进化曲线
string = 'GWO';
figure
semilogy(1:Max_iteration, Convergence_curve,'b-', 'color',[0.1 0.1 0.5],'linewidth',1.5);
grid on
xlabel('迭代次数')
ylabel('适应度函数')
title([string,'进化曲线'])

%% 把最优初始阀值权值赋予BP网络预测
x=Leader_pos;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% 优化后的BP神经网络训练
[net,tr]=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本
view(net)
figure,plotperform(tr)                          %误差MSE下降线

%% 优化后的BP神经网络预测
an=sim(net,inputn_test); %用训练好的模型进行仿真

%% 预测结果反归一化与误差计算     
test_simu=mapminmax('reverse',an,outputps); %把仿真得到的数据还原为原始的数量级
error=abs(test_simu-output_test);      %预测值和期望值的误差
errorPer=error./output_test;      %预测值和期望值的误差

[c,len]=size(output_test);   %获取测试样本的总数

% 决定系数R^2
R2 = (len* sum(test_simu .* output_test) - sum(test_simu) * sum(output_test))^2 / ((len * sum((test_simu).^2) - (sum(test_simu))^2) * (len * sum((output_test).^2) - (sum(output_test))^2)); 
%误差分析
MAE=sum(abs(error))/len;
MSE=error*error'/len;
RMSE=MSE^(1/2);

%% 网络预测图形
figure
plot(output_test,'ro-','linewidth',0.8)
hold on
plot(test_simu,'b*-','linewidth',0.8)
title({[string,'优化BP神经网络测试集预测值与真实值的对比'],['RMSE = ',num2str(RMSE), ', R^2 = ', num2str(R2)]}, 'fontsize',12)
legend('真实值','预测值')
xlabel('数据组数')

figure
plot(errorPer,'b*-','linewidth',0.5)
title({[string,'优化BP神经网络预测值和真实值的相对误差图'],['平均相对误差 ',num2str(mean(errorPer)*100),'%']},'fontsize',12)
xlabel('样本编号')
ylabel('相对误差')


%输出结果
disp(['平均绝对误差, MAE = ',num2str(MAE)])
disp(['均方误差, MSE = ',num2str(MSE)])
disp(['根均方误差, RMSE = ',num2str(RMSE)])
disp(['决定系数R^2 = ',num2str(R2)])

save myworkspace




