%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库

x = 1:10;
y = log(x); %lnx
plot(x,y,'o') % 描点作图
hold on;
new_x = 0.01:0.1:10;
p = pchip(x,y,new_x); % 三次埃尔米特插值法
plot(new_x,p)
hold on;
p = spline(x,y,new_x);% 三次样条插值
plot(new_x,p,'b-')
legend('样本点', '三次埃尔米特插值', '三次样条插值', 'Location', 'SouthEast');

%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库