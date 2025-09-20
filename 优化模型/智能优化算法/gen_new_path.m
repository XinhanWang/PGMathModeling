%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
function path1 = gen_new_path(path0, dist)  
    % 用改良圈算法生成新路径
    L = length(path0);
    path1 = path0;% 初始圈    
    flag = 0;% 修改标志
    for i = 1 : L-2  % i是第一条弧的起点
        for j = i + 2 : L-1 % j是第二条弧的起点
            if( dist(path1(i),path1(j)) + dist(path1(i+1),path1(j+1)) < dist(path1(i),path1(i+1)) + dist(path1(j),path1(j+1)) )
                path1(i+1:j) = path1(j:-1:i+1);% 翻转中间的路径,j:-1:i+1表示从j到i+1，步长为-1
                flag = flag + 1;
            end
        end
    end
end
