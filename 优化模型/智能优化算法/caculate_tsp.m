%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��
function  result =  caculate_tsp(path, dist)
    n = length(path);
    result = 0; % ��ʼ����·���ߵľ���Ϊ0
    for i = 1:n-1 
        result = dist(path(i),path(i+1)) + result;  % �������·���Ĵ���
    end   
end