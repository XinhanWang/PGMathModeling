%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��

function [isconnect dist] = find_Path(C, D, j, i, dist) 
    if (j > 6 || i > 6)
        return;
    end
    if (j == i) %���������Ŀ�������ȣ�����·��
        isconnect  = 1;
        return;
    end
    isconnect = 0;
    for k = j + 1 : 6 %����һ���������
        if (k <= 6 && D(j,k) == 1)
            dist = dist + C(j,k);
            find_Path(C, D, k, i, dist);
        end
    end
end

%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��

