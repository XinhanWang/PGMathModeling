%% 注意：代码文件仅供参考，一定不要直接用于自己的数模论文中
%% 国赛对于论文的查重要求非常严格，代码雷同也算作抄袭
%% 全套课程购买地址：https://k.youshop10.com/NjNu80mX
%% 全套教材PPT请关注微信公众号：大师兄的知识库
function path1 = gen_new_path1(path0)
    % 用交换法生成新路径
    n = length(path0);
    c1 = randi([2 n-1],1);   % 生成2至n-1中的一个随机整数
    c2 = randi([2 n-1],1);   % 生成2至n-1中的一个随机整数
    path1 = path0;  % 将path0的值赋给path1
    path1(c1) = path0(c2);  %改变path1第c1个位置的元素为path0第c2个位置的元素
    path1(c2) = path0(c1);  %改变path1第c2个位置的元素为path0第c1个位置的元素
end