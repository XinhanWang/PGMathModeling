%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
%% SA模拟退火: 求解函数y = x(1)*sin(pi*x(1))+x(2)*cos(4*pi*x(2))在x1属于[-3,3]，x2属于[4,5]内的最大值
clear;clc
x1 = -3:0.01:3;
x2 = 5:0.01:5;
% [x1,x2] = meshgrid(x1,x2);
y = x1 .* sin(pi*x1) + x2 .* cos(4*pi*x2);
% 参数初始化
narvs = 2; % 变量个数
T0 = 1000;   % 初始温度
T = T0; % 迭代中温度会发生改变，第一次迭代时温度就是T0
maxgen = 1000;  % 最大迭代次数
Lk = 300;  % 每个温度下的迭代次数
alfa = 0.95;  % 温度衰减系数
x_lb = [-3 4]; % x的下界
x_ub = [4 5]; % x的上界

%  随机生成一个初始解
x0 = zeros(1,narvs);
for i = 1: narvs
    x0(i) = x_lb(i) + (x_ub(i)-x_lb(i))* rand(1);    % 保证初始化粒子落在定义域内
end
y0 = fun(x0); % 计算当前解的函数值

% 定义一些保存中间过程的量，方便输出结果和画图
max_y = y0;     % 初始化找到的最佳的解对应的函数值为y0

% 模拟退火过程
for iter = 1 : maxgen  % 外循环, 指定最大迭代次数
    for i = 1 : Lk  %  内循环，在每个温度下开始迭代
        y = randn(1,narvs);  % 生成1行narvs列的N(0,1)随机数
        z = y / sqrt(sum(y.^2)); % 为了方便计算，进行标准化z
        x_new = x0 + z*T; % 跳到随机产生的x附近的x_new
        
        % 如果这个新解的位置超出了定义域，就对其进行调整
        for j = 1: narvs
            if x_new(j) < x_lb(j)
                r = rand(1);
                x_new(j) = r*x_lb(j)+(1-r)*x0(j);
            elseif x_new(j) > x_ub(j)
                r = rand(1);
                x_new(j) = r*x_ub(j)+(1-r)*x0(j);
            end
        end
        
        x1 = x_new;    % 将调整后的x_new赋值给新解x1
        y1 = fun(x1);  % 计算新解的函数值
        if y1 > y0    % 如果新解函数值大于当前解的函数值
            x0 = x1; % 更新当前解为新解
            y0 = y1;
        else
            p = exp(-(y0 - y1)/T); % 计算一个概率
            if rand(1) < p   % 生成一个随机数和这个概率比较，如果该随机数小于这个概率
                x0 = x1; % 更新当前解为新解
                y0 = y1;
            end
        end
        % 判断是否要更新找到的最佳的解
        if y0 > max_y  % 如果当前解更好，则对其进行更新
            max_y = y0;  % 更新最大的y
            best_x = x0;  % 更新找到的最好的x
        end
    end
end


disp('取最大值时的根是：'); disp(best_x)
disp('此时对应的最大值是：'); disp(max_y)