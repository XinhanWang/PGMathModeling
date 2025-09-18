%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

%% 区间型转极大型，传入参数为待正向化的向量，返回为正向化后的结果
function [res] = Int2Max(X, a, b)  
   M =  max(a - min(X), max(X) - b);
   for i = 1 : size(X)
       if(X(i) < a)
           X(i) = 1 - (a - X(i))/M;
       elseif (X(i) >= a && X(i) <= b)
           X(i) = 1;
       elseif (X(i) > b)
           X(i) = 1 - (X(i) - b)/M;
       end
   end
   res = X;
end

%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库