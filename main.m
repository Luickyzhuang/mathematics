%% �˳���Ϊ���ڻ����Ż��㷨�Ż���BP������
% ��ջ�������
clear
close all
clc
warning off

%% ��������
% ��ȡ������Ա���
P = xlsread('ѵ���Ͳ��Լ���ȫ������.xlsx','Sheet1','A2:B2000')';
% ��ȡ����������
T =  xlsread('ѵ���Ͳ��Լ���ȫ������.xlsx','Sheet1','C2:C2000')';

% �趨ѵ���Ͳ��Ե���������
N = 50;  % �趨50�����ݲ���
M = size(T, 2) - N;  % �������������ѵ��

% �����ȡѵ�������Լ�
indx = randperm(size(T, 2) );
P = P(:,indx);
T = T(:, indx);

%ѵ����
input_train = P(:, 1:M);
output_train = T(:, 1:M);
% ���Լ�
input_test= P(:, M+1:M+N);
output_test= T(:, M+1:M+N);

%% ��һ��
% ѵ����
[inputn,inputps] = mapminmax(input_train,-1,1);
inputn_test = mapminmax('apply',input_test,inputps);
% ���Լ�
[outputn,outputps] = mapminmax(output_train,-1,1);
outputn_test = mapminmax('apply',output_test,outputps);

%% �����Ż�����
%�ڵ����
inputnum=size(input_train,1);%��������
hiddennum=10;
outputnum=size(output_train,1);

%��������
net=newff(inputn,outputn,hiddennum);
%�������
net.trainParam.epochs=1000;   % ���ѵ������
net.trainParam.lr=0.01;   % ѧϰ��
net.trainParam.goal=0.00001;  % Ŀ�����
net.trainParam.show=100;   % ��ʾƵ��

% ������ʼ��
dim=inputnum * hiddennum + hiddennum*outputnum + hiddennum + outputnum;
Max_iteration=20;   % ��������  % һ��ȡ30-100  Ϊ�˼ӿ��Ż�������СһЩ
pop=10;  %��Ⱥ��ģ  % һ��ȡ10-50
lb=-3;  %Ȩֵ��ֵ�±߽�
ub=3;   %Ȩֵ��ֵ�ϱ߽�
fobj = @(x) fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
[Leader_score,Leader_pos,Convergence_curve]=GWO(pop,Max_iteration,lb,ub,dim,fobj); %�����Ӻ�������ʼ�Ż�
% �����Ż���������������
string = 'GWO';
figure
semilogy(1:Max_iteration, Convergence_curve,'b-', 'color',[0.1 0.1 0.5],'linewidth',1.5);
grid on
xlabel('��������')
ylabel('��Ӧ�Ⱥ���')
title([string,'��������'])

%% �����ų�ʼ��ֵȨֵ����BP����Ԥ��
x=Leader_pos;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% �Ż����BP������ѵ��
[net,tr]=train(net,inputn,outputn);%��ʼѵ��������inputn,outputn�ֱ�Ϊ�����������
view(net)
figure,plotperform(tr)                          %���MSE�½���

%% �Ż����BP������Ԥ��
an=sim(net,inputn_test); %��ѵ���õ�ģ�ͽ��з���

%% Ԥ��������һ����������     
test_simu=mapminmax('reverse',an,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
error=abs(test_simu-output_test);      %Ԥ��ֵ������ֵ�����
errorPer=error./output_test;      %Ԥ��ֵ������ֵ�����

[c,len]=size(output_test);   %��ȡ��������������

% ����ϵ��R^2
R2 = (len* sum(test_simu .* output_test) - sum(test_simu) * sum(output_test))^2 / ((len * sum((test_simu).^2) - (sum(test_simu))^2) * (len * sum((output_test).^2) - (sum(output_test))^2)); 
%������
MAE=sum(abs(error))/len;
MSE=error*error'/len;
RMSE=MSE^(1/2);

%% ����Ԥ��ͼ��
figure
plot(output_test,'ro-','linewidth',0.8)
hold on
plot(test_simu,'b*-','linewidth',0.8)
title({[string,'�Ż�BP��������Լ�Ԥ��ֵ����ʵֵ�ĶԱ�'],['RMSE = ',num2str(RMSE), ', R^2 = ', num2str(R2)]}, 'fontsize',12)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('��������')

figure
plot(errorPer,'b*-','linewidth',0.5)
title({[string,'�Ż�BP������Ԥ��ֵ����ʵֵ��������ͼ'],['ƽ�������� ',num2str(mean(errorPer)*100),'%']},'fontsize',12)
xlabel('�������')
ylabel('������')


%������
disp(['ƽ���������, MAE = ',num2str(MAE)])
disp(['�������, MSE = ',num2str(MSE)])
disp(['���������, RMSE = ',num2str(RMSE)])
disp(['����ϵ��R^2 = ',num2str(R2)])

save myworkspace




