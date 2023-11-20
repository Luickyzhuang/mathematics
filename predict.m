%% 此程序为调用训练好的GWO-BP神经网络
% 清空环境变量
clear
close all
clc
warning off

%% 读取数据
new_input=xlsread('预测输入.xlsx');  % 修改为自己的数据

load myworkspace

%% 数据归一化
[new_inputn]=mapminmax('apply',new_input',inputps);%归一化

%% 预测结果
an0=sim(net,new_inputn); %用训练好的模型进行仿真
new_predict=mapminmax('reverse',an0,outputps);

%
disp('新数据的预测结果')
new_predict'



