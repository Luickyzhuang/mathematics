%% �˳���Ϊ����ѵ���õ�GWO-BP������
% ��ջ�������
clear
close all
clc
warning off

%% ��ȡ����
new_input=xlsread('Ԥ������.xlsx');  % �޸�Ϊ�Լ�������

load myworkspace

%% ���ݹ�һ��
[new_inputn]=mapminmax('apply',new_input',inputps);%��һ��

%% Ԥ����
an0=sim(net,new_inputn); %��ѵ���õ�ģ�ͽ��з���
new_predict=mapminmax('reverse',an0,outputps);

%
disp('�����ݵ�Ԥ����')
new_predict'



