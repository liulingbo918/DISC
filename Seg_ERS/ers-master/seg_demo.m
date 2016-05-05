
%close all;clear all;clc


DISC_Test_Image_List_Path = '../../DISC_Data/DISC_test.list';
DISC_Test_Image_Path = '../../DISC_Data/DISC_Input/';
DISC_Test_Image_Seg_Path = '../../DISC_Data/DISC_Input/';
DISC_Test_Image_List = textread(DISC_Test_Image_List_Path,'%s');
 
[filesize, ~] = size(DISC_Test_Image_List);

img_size = 64; 
nC = 200;  %nC is the target number of superpixels  
    
for index = 1:filesize
    index
    
    img_path = [DISC_Test_Image_Path, DISC_Test_Image_List{index}];
    imgage = imread([img_path, '.jpg']);
    img = imresize(imgage, [img_size, img_size]);
   
    % Our implementation can take both color and grey scale images.
    [a b] = size(size(img));
    if b == 3
        grey_img = double(rgb2gray(img));
    else 
         grey_img = double(img);
    end
    
    t = cputime;
    [labels] = mex_ers(double(img), nC);
    
    %printf(1,'Use %f sec.',cputime-t); 
    
    seg_min = min(min(labels(:)));
    seg_max = max(max(labels(:)));
    if seg_min ~=0 || seg_max ~=nC -1
        dlmwrite('error.txt',labels);
    end
    
    file_path = [DISC_Test_Image_Seg_Path, DISC_Test_Image_List{index}];
    dlmwrite([file_path,'_seg.list'], labels,' ');
end

exit;
