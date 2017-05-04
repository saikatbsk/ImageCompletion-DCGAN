clear; clc

data_dir = '/home/saikat/Downloads/abhishek bachchan _ Google Search/';
save_dir = '/home/saikat/Downloads/abhishek_bachchan';

input_images = dir(fullfile(data_dir, '*.jpg'));
input_images = {input_images.name};

faceDetector = vision.CascadeObjectDetector;

for i = 1:length(input_images)
    I = imread(char(fullfile(data_dir, input_images(i))));
    bboxes = step(faceDetector, I);

    for j = 1:size(bboxes, 1)
        face = imcrop(I, bboxes(j, :));
        if ((size(face, 1)*size(face, 1)) >= (96*96))
            fn = char(fullfile(save_dir, strcat(char(floor(25*rand(1, 24)+65)), '.jpg')));
            imwrite(face, fn);
        end
    end
end
