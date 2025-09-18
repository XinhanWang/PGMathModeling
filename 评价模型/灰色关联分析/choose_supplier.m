%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

%% 读取数据
clear;clc
X = xlsread('data.xlsx');

%% 正向化
disp('***************正在进行正向化...***************');
vec = input('请输入要正向化的向量组，请以数组的形式输入，如[1 2 3]表示1，2，3列需要正向化，不需要正向化请输入-1\n') %注意输入函数这里是单引号
if (vec ~= -1)
    for i = 1 : size(vec,2)
        flag = input(['第' num2str(vec(i)) '列是哪类数据(【1】:极小型 【2】：中间型 【3】：区间型)，请输入序号：\n']);
        if(flag == 1)%极小型
           X(:,vec(i)) = Min2Max(X(:,vec(i)));
        elseif (flag == 2) % 注意这里的else和if是连在一起的
            best = input('请输入中间型的最好值：\n');
            temp = X(:,vec(i));
            X(:,vec(i)) = Mid2Max(X(:,vec(i)), best);
        elseif (flag == 3)
            arr = input('请输入最佳区间，按照“[a,b]”的形式输入：\n');
            X(:,vec(i)) = Int2Max(X(:,vec(i)), arr(1), arr(2));
        end
    end
    disp('所有的数据均已完成正向化！')
end
%% 标准化
disp('***************正在进行标准化...***************');
[n,m] = size(X);
% 先检查有没有负数元素
isNeg = 0;
for i = 1 : n
    for j = 1 : m
        if(X(i,j) < 0)
            isNeg = 1;
            break;
        end
    end
end
if (isNeg == 0)
    squere_X = (X.*X);
    sum_X = sum(squere_X,1).^0.5; %按列求和,再开方
    stand_X = X./repmat(sum_X, n, 1);
else
    max_X = max(X,[],1); %按照列找出最大元素
    min_X = min(X,[],1); %按照列找出最小元素
    stand_X = X - repmat(min_X,n,1) ./ (repmat(max_X,n,1) - repmat(min_X,n,1));
end
disp('标准化完成！')

%% 灰色关联分析
res = stand_X;
gre_X = [res,max(res,[],2)]; %注意我们这里把x0放到了最后一列
[m,n] = size(gre_X);
gamma_X = zeros(m,n-1);
for i = 1 : n - 1
    gamma_X(:,i) = abs(gre_X(:,i) - gre_X(:,n));
end
a = min(min(gamma_X));
b = max(max(gamma_X));
roh = 0.5;
gamma_X = (a + roh*b) ./ (gamma_X + roh*b);
gre_res = sum(gamma_X) ./ m; 

%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
