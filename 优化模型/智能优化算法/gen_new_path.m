%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��
function path1 = gen_new_path(path0, dist)  
    % �ø���Ȧ�㷨������·��
    L = length(path0);
    path1 = path0;% ��ʼȦ    
    flag = 0;% �޸ı�־
    for i = 1 : L-2  % i�ǵ�һ���������
        for j = i + 2 : L-1 % j�ǵڶ����������
            if( dist(path1(i),path1(j)) + dist(path1(i+1),path1(j+1)) < dist(path1(i),path1(i+1)) + dist(path1(j),path1(j+1)) )
                path1(i+1:j) = path1(j:-1:i+1);% ��ת�м��·��,j:-1:i+1��ʾ��j��i+1������Ϊ-1
                flag = flag + 1;
            end
        end
    end
end
