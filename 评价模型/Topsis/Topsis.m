%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

%clear;clc;
%X = [89,1; 60,3; 74,2; 99,0]
%X=[99;100;98;97]
%X=[0.030;0.028;0;0.007]
%X=[99,0.030;100,0.028;98,0;97,0.007]
%X=[99,0.010;100,0.012;98,0.040;97,0.033]
%X = [35.2;35.8;36.5;37.2;38.0]
%X = [0.6;0.75;0.89;0.95]
%X = [180;175;170;185;190]
%X = [60;90;95;81;79]
X = xlsread('工作簿1.xlsx');
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
            arr = input('请输入最佳区间，按照“[a,b]”的形式输入：\n')
            X(:,vec(i)) = Int2Max(X(:,vec(i)), arr(1), arr(2));
        end
    end
    disp('所有的数据均已完成正向化！')
end
%% 标准化
disp('***************正在进行标准化...***************');
[n,m] = size(X)
% 先检查有没有负数元素
isNeg = 0;
for i = 1:n
    for j = 1 : m
        if(X(i,j) < 0)
            isNeg = 1;
            break;
        end
    end
end
if (isNeg == 0)
    squere_X = (X.*X)
    sum_X = sum(squere_X,1).^0.5 %按列求和,再开方
    stand_X = X./repmat(sum_X, n, 1)
else
    max_X = max(X,[],1); %按照列找出最大元素
    min_X = min(X,[],1); %按照列找出最小元素
    stand_X = X - repmat(min_X,n,1) ./ (repmat(max_X,n,1) - repmat(min_X,n,1));
end

%% （法一）用距离法打分
disp('***************正在用距离法打分...***************');
max_x = max(stand_X,[], 1) %按照列找出最大元素
min_x = min(stand_X,[], 1) %按照列找出最小元素

(stand_X - repmat(min_x,n,1)) ./ (max_x - min_x)


%% （法二）用优劣解打分
disp('***************正在用优劣解打分...***************');
tmp = ones(m);% 这个矩阵很有用，要掌握哦
w_j = tmp(:,1);
is_need_w = input('是否需要指定权值，如需要请输入1，否则请输入0:\n');
if (is_need_w == 1)
    w_j = input('请按列输入各指标的权值：(如[0.1;0.2;0.3;0.4])')
end
Z_plus = repmat(max_x,n,1);
Z_sub = repmat(min_x,n,1);
D_plus = sum(((stand_X - Z_plus).^2) * w_j, 2).^0.5 %注意是按行求和
D_sub = sum(((stand_X - Z_sub)).^2 * w_j, 2).^0.5

S = D_sub ./ (D_sub + D_plus)

%将结果归一化
res_topsis = S ./ sum(S)
%xlswrite('res_topsis.xlsx',res_topsis) %写入excel文档
%disp('已完成打分，请到当前目录下res_topsis.xlsx文件中取出结果！')


%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
