%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
function  result =  caculate_tsp(path, dist)
    n = length(path);
    result = 0; % 初始化该路径走的距离为0
    for i = 1:n-1 
        result = dist(path(i),path(i+1)) + result;  % 计算给定路径的代价
    end   
end