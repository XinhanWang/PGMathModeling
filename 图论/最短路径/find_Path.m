%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

function [isconnect dist] = find_Path(C, D, j, i, dist) 
    if (j > 6 || i > 6)
        return;
    end
    if (j == i) %跳板城市与目标城市相等，即有路径
        isconnect  = 1;
        return;
    end
    isconnect = 0;
    for k = j + 1 : 6 %找下一个跳板城市
        if (k <= 6 && D(j,k) == 1)
            dist = dist + C(j,k);
            find_Path(C, D, k, i, dist);
        end
    end
end

%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

