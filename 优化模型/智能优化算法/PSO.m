%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��
%% ����Ⱥ�Ż��㷨
clear;clc
f= @(x) x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x); % �������������Сֵ  

N = 20;                         % ��ʼ��Ⱥ����  
d = 1;                          % ���н�ά��  
ger = 100;                      % ����������       
limit = [0, 50];                % ����λ�ò���������  
vlimit = [-10, 10];             % �����ٶ�������  
w = 0.8;                        % ����Ȩ��  
c1 = 0.5;                       % ����ѧϰ����  
c2 = 0.5;                       % Ⱥ��ѧϰ����   
figure(1);
ezplot(f,[0,0.01,limit(2)]);   % ֱ�ӻ���������ͼ�Σ���������׼��

% һЩ��ʼ��
x = limit(1) + ( limit(2) -  limit(1) ) .* rand(N, d);%��ʼ��Ⱥ��λ��  
v = rand(N, d);                  % ��ʼ��Ⱥ���ٶ�  
xm = x;                          % ÿ���������ʷ���λ��  
ym = zeros(1, d);                % ��Ⱥ����ʷ���λ��  
fxm = ones(N, 1)*inf;            % ÿ���������ʷ�����Ӧ��  
fym = inf;                       % ��Ⱥ��ʷ�����Ӧ��  
hold on  
plot(xm, f(xm), 'ro');
title('���ӳ�ʼ״̬ͼ');  
%% Ⱥ�����  
figure(2)  
iter = 1;  
record = zeros(ger, 1);
while iter <= ger  
     fx = f(x) ; % ���嵱ǰ��Ӧ��     
     for i = 1:N        
        if fx(i) < fxm(i) 
            fxm(i) = fx(i);     % ���¸�����ʷ�����Ӧ��  
            xm(i,:) = x(i,:);   % ���¸�����ʷ���λ��
        end   
     end  
    if  min(fxm)  < fym 
        [fym, nmin] = min(fxm);   % ����Ⱥ����ʷ�����Ӧ��  
        ym = xm(nmin, :);      % ����Ⱥ����ʷ���λ��  
    end  
    v = v * w + c1 * rand * (xm - x) + c2 * rand * (repmat(ym, N, 1) - x);% �ٶȸ���  
    % �߽��ٶȴ���  
    v(v > vlimit(2)) = vlimit(2);  
    v(v < vlimit(1)) = vlimit(1);  
    x = x + v; % λ�ø���  
    % �߽�λ�ô���  
    x(x > limit(2)) = limit(2);  
    x(x < limit(1)) = limit(1);  
    record(iter) = fym; % ��Сֵ��¼  
    x0 = 0 : 0.01 : limit(2);  
    subplot(1,2,1)
    plot(x0, f(x0), 'b-', x, f(x), 'ro');
    title('״̬λ�ñ仯')
    subplot(1,2,2);
    plot(record);
    title('������Ӧ�Ƚ�������')  
    pause(0.01)  
    iter = iter+1;  

end  

x0 = 0 : 0.01 : limit(2);  
figure(4);
plot(x0, f(x0), 'b-', x, f(x), 'ro');
title('����״̬λ��')  
disp(['��Сֵ��',num2str(fym)]);  
disp(['����ȡֵ��',num2str(ym)]);  