function temp_1=load_process(x)

% The file 'test.csv' and the file 'train.csv' should be saved in the same folder of 'load_process.m'
% If you want to get image data in test.csv, you need to run the code "x =  csvread('test.csv',1,0);" first, 
% and run the function to get the image data. The input is x as previously get, out data is called temp_1.
% If you want to get image data in train.csv, the code should be changed into "x =  csvread('train.csv',1,1);",
% and then run the function.
% the label of training dataset could be got by "train =  csvread('train.csv',1,0; y=train(size(train,1),:)"

B=x';
a=size(x,1);
b=size(x,2);
B=B(:);
final=zeros(a,b);
temp=1;
for i=1:b
    for j=1:a
        final(j,i)=B(temp);
        temp=temp+1;
    end
end

temp_1=zeros(32,32,3,a);
for m=1:a
    temp=final(m,:);
    temp_2=temp(:);
    temp_3=1;
    for i=1:32
        for j=1:32
            for k=1:3
                temp_1(i,j,k,m)=temp_2(temp_3);
                temp_3=temp_3+1;
            end
        end
    end
end