%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��

%% ��ȡ����
clear;clc
X = xlsread('data.xlsx');

%% ����
disp('***************���ڽ�������...***************');
vec = input('������Ҫ���򻯵������飬�����������ʽ���룬��[1 2 3]��ʾ1��2��3����Ҫ���򻯣�����Ҫ����������-1\n') %ע�����뺯�������ǵ�����
if (vec ~= -1)
    for i = 1 : size(vec,2)
        flag = input(['��' num2str(vec(i)) '������������(��1��:��С�� ��2�����м��� ��3����������)����������ţ�\n']);
        if(flag == 1)%��С��
           X(:,vec(i)) = Min2Max(X(:,vec(i)));
        elseif (flag == 2) % ע�������else��if������һ���
            best = input('�������м��͵����ֵ��\n');
            temp = X(:,vec(i));
            X(:,vec(i)) = Mid2Max(X(:,vec(i)), best);
        elseif (flag == 3)
            arr = input('������������䣬���ա�[a,b]������ʽ���룺\n');
            X(:,vec(i)) = Int2Max(X(:,vec(i)), arr(1), arr(2));
        end
    end
    disp('���е����ݾ���������򻯣�')
end
%% ��׼��
disp('***************���ڽ��б�׼��...***************');
[n,m] = size(X);
% �ȼ����û�и���Ԫ��
isNeg = 0;
for i = 1 : n
    for j = 1 : m
        if(X(i,j) < 0)
            isNeg = 1;
            break;
        end
    end
end
if (isNeg == 0)
    squere_X = (X.*X);
    sum_X = sum(squere_X,1).^0.5; %�������,�ٿ���
    stand_X = X./repmat(sum_X, n, 1);
else
    max_X = max(X,[],1); %�������ҳ����Ԫ��
    min_X = min(X,[],1); %�������ҳ���СԪ��
    stand_X = X - repmat(min_X,n,1) ./ (repmat(max_X,n,1) - repmat(min_X,n,1));
end
disp('��׼����ɣ�')

%% ��ɫ��������
res = stand_X;
gre_X = [res,max(res,[],2)]; %ע�����������x0�ŵ������һ��
[m,n] = size(gre_X);
gamma_X = zeros(m,n-1);
for i = 1 : n - 1
    gamma_X(:,i) = abs(gre_X(:,i) - gre_X(:,n));
end
a = min(min(gamma_X));
b = max(max(gamma_X));
roh = 0.5;
gamma_X = (a + roh*b) ./ (gamma_X + roh*b);
gre_res = sum(gamma_X) ./ m; 

%% ע�⣺�����ļ������ο���һ����Ҫֱ�������Լ�����ģ������
%% �����������ĵĲ���Ҫ��ǳ��ϸ񣬴�����ͬҲ������Ϯ
%% ȫ�׿γ̹����ַ��https://k.youshop10.com/NjNu80mX
%% ȫ�׽̲�PPT���ע΢�Ź��ںţ���ʦ�ֵ�֪ʶ��
