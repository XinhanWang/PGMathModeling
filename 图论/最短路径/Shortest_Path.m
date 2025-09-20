%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

%% 用蒙特卡洛解决最短路径问题（了解即可，不用掌握） 
C = [0 50 999 40 25 10;
    50 0 15 20 999 25;
    999 15 0 10 20 999;
    40 20 10 0 10 25;
    25 999 20 10 0 55;
    10 25 999 25 55 0];
 s = [+Inf, +Inf, +Inf, +Inf, +Inf, +Inf];
 for k = 1 : 10000     
     D = randi(([0 1]),6);
     % 从c1开始
     for i = 1 : 6 %目标城市       
         isconnect = 0;
         dist = 0;
         for j = i + 1 : 6 %j表示跳板城市，用j枚举城市检查是否连通，因为是一个无向图，也就是有回路的，所以只需要考虑上三角即可
            if(D(1,j) == 1) %表示c1和跳板城市有连通
                %检查跳板城市能否到达目标城市，如果不能就找跳板的跳板城市
                dist = dist + C(1,j);
                [r_isconnect r_dist] = find_Path(C, D, j, i, dist);
                if (r_isconnect == 1)
                    isconnect = 1;
                    break;
                end
                %如果c1和跳板不连通，找下一个连通的城市
            end
         end
         if isconnect == 1 %记录当前模拟的路径距离
             s(i) = dist;
         end
     end
     
 end

%% Dijkstra
% 注意Matlab中的图节点要从1开始编号
s = [1 1 2 2 8 8 3 9 3 3 7  4  4 6];
t = [2 8 8 3 9 7 9 7 4 6 6  6  5 5];
w = [4 8 3 8 1 6 2 6 7 4 2 14 9 10];
G = graph(s,t,w);
plot(G, 'EdgeLabel', G.Edges.Weight, 'linewidth', 2) 
set( gca, 'XTick', [], 'YTick', [] ); %去掉坐标轴上的数字
[P,d] = shortestpath(G, 1, 4)  %计算从1到9的最短路径

%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

