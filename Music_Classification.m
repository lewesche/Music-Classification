% Leif Wesche
% Music Classification

%% Load Genres
clear all
close all
clc

% untar('genres.tar.gz');
addpath('genres\classical');
addpath('genres\hiphop');
addpath('genres\country');

songs=100;
range=[1, 660000];
for i=[1:songs]
    if i >= 11
    dum=audioread(['genres\country\country.000', num2str(i-1), '.au'], range);    
    m1{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=audioread(['genres\classical\classical.000', num2str(i-1), '.au'], range);
    m2{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=audioread(['genres\hiphop\hiphop.000', num2str(i-1), '.au'], range);
    m3{i}=abs(spectrogram(dum(1:2:end,1)));
    end
    if i < 11
    dum=audioread(['genres\country\country.0000', num2str(i-1), '.au'], range);    
    m1{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=audioread(['genres\classical\classical.0000', num2str(i-1), '.au'], range);
    m2{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=audioread(['genres\hiphop\hiphop.0000', num2str(i-1), '.au'], range);
    m3{i}=abs(spectrogram(dum(1:2:end,1)));
    end
end

label=['Test 3: Genre Classification'];
label_short=['Test 3'];
legend={'Country', 'Classical', 'Hip Hop', 'Average'};

%% Load Nirvana, Mastadon, CHILLI PEPPERS

close all
clear all
clc

addpath('Music\N');
addpath('Music\M');
addpath('Music\P');

range=[200000, 400000];
songs=44;
for i=[1: songs]
    dum=(audioread(['M', num2str(i), '.mp3'], range)); 
    m1{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=(audioread(['N', num2str(i), '.mp3'], range)); 
    m2{i}=abs(spectrogram(dum(1:2:end,1))); 
    dum=(audioread(['P', num2str(i), '.mp3'], range)); 
    m3{i}=abs(spectrogram(dum(1:2:end,1))); 
end

label=['Test 2: Rock Band Classification'];
label_short=['Test 2'];
legend={'Mastadon', 'Nirvana', 'Red Hot Chili Peppers', 'Average'};

%% Load Gucci Mane, Chopin, Nirvana

clear all
close all
clc

addpath('Music\N');
addpath('Music\G');
addpath('Music\C');

range=[200000, 400000];
songs=67;
for i=[1: songs]
    dum=(audioread(['G', num2str(i), '.mp3'], range)); 
    m1{i}=abs(spectrogram(dum(1:2:end,1)));
    dum=(audioread(['N', num2str(i), '.mp3'], range)); 
    m2{i}=abs(spectrogram(dum(1:2:end,1))); 
    dum=(audioread(['C', num2str(i), '.mp3'], range)); 
    m3{i}=abs(spectrogram(dum(1:2:end,1))); 
end

label=['Test 1: Various Artists Classification'];
label_short=['Test 1'];
legend={'Gucci Mane', 'Nirvana', 'Chopin', 'Average'};

%% Arrange into vectors, SVD
clc
close all

music1=[]; music2=[]; music3=[];
for i=[1:songs]
    music1=[music1, m1{i}(:)];
    music2=[music2, m2{i}(:)];
    music3=[music3, m3{i}(:)];
end

X=[music1, music2, music3];
[U, S, V]=svd(X, 'econ');

% [Ur,Sr,Vr]=svd(reggae, 'econ');
% [Uc,Sc,Vc]=svd(classical, 'econ');
% [Uh,Sh,Vh]=svd(hiphop, 'econ');

%% Plot Singular Values
clc
close all

figure
bar(diag(S(1:end, 1:end))./sum(sum(S)), 'r')
title([label_short, ': Normalized Singular Value Spectrum']);
xlabel('Singular Value Number'); ylabel('Percentage of Spectrum');

figure
plot3(V(1:songs,2), V(1:songs,3), V(1:songs,4), 'ro', 'linewidth', 2)
hold on
plot3(V(songs+1:2*songs,2), V(songs+1:2*songs,3), V(songs+1:2*songs,4), 'bo', 'linewidth', 2)
hold on
plot3(V(2*songs+1:end,2), V(2*songs+1:end,3), V(2*songs+1:end,4), 'ko', 'linewidth', 2)
title([label_short, ': First Three Component Projections'])

%% Classify and Test songs
close all
clc

comps=[2:20];
train=round(0.85*songs);
test=round(0.15*songs);
trials=100;

Vm1=V(1:songs, comps);
Vm2=V(songs+1:2*songs, comps);
Vm3=V(2*songs+1:end, comps);
ctrain=[ones(train,1); 2*ones(train,1); 3*ones(train,1)]; 
svm_eval=[0,0,0];
nb_eval=[0,0,0];
tf_eval=[0,0,0];

for i=[1:trials]
    
q1=randperm(songs);
q2=randperm(songs);
q3=randperm(songs);

Xtrain=[Vm1(q1(1:train),:); Vm2(q2(1:train),:); Vm3(q3(1:train),:)];
Xtest=[Vm1(q1(train+1:end),:); Vm2(q2(train+1:end),:); Vm3(q3(train+1:end),:)];

svm=fitcecoc(Xtrain, ctrain);
pre_svm=predict(svm, Xtest);

nb=fitcnb(Xtrain, ctrain);
pre_nb=nb.predict(Xtest);

tf=fitctree(Xtrain,ctrain);
pre_tf=predict(tf, Xtest);

for j=[1:length(pre_svm)/3]
    if  pre_svm(j) == 1
        svm_eval(1)=svm_eval(1)+1;
    end
    if  pre_nb(j) == 1
        nb_eval(1)=nb_eval(1)+1;
    end
    if  pre_tf(j) == 1
        tf_eval(1)=tf_eval(1)+1;
    end
end
for j=[length(pre_svm)/3+1:2*length(pre_svm)/3]
    if  pre_svm(j) == 2
        svm_eval(2)=svm_eval(2)+1;
    end
    if  pre_nb(j) == 2
        nb_eval(2)=nb_eval(2)+1;
    end
    if  pre_tf(j) == 2
        tf_eval(2)=tf_eval(2)+1;
    end
end
for j=[2*length(pre_svm)/3+1:length(pre_svm)]
    if  pre_svm(j) == 3
        svm_eval(3)=svm_eval(3)+1;
    end
    if  pre_nb(j) == 3
        nb_eval(3)=nb_eval(3)+1;
    end
    if  pre_tf(j) == 3
        tf_eval(3)=tf_eval(3)+1;
    end
end

end

%figure
%bar(pre)

for i=[1:3]
svm_eval(i)=svm_eval(i)/(trials*test)*100;
nb_eval(i)=nb_eval(i)/(trials*test)*100;
tf_eval(i)=tf_eval(i)/(trials*test)*100;
end



figure
subplot(2,2,1)
bar([svm_eval(1), svm_eval(2), svm_eval(3), mean(svm_eval)], 'b')
set(gca,'xticklabel',legend); title([label_short, ': SVM Percent Accuracy']);
axis([0,5,0,100]); ylabel('Percent of Correct Clasification')
subplot(2,2,2)
bar([nb_eval(1), nb_eval(2), nb_eval(3), mean(nb_eval)], 'g')
set(gca,'xticklabel',legend); title([label_short, ': Naive Bayes Percent Accuracy']);
axis([0,5,0,100]); ylabel('Percent of Correct Clasification')
subplot(2,2,3)
bar([tf_eval(1), tf_eval(2), tf_eval(3), mean(tf_eval)], 'y')
set(gca,'xticklabel',legend); title([label_short ': Decision Tree Percent Accuracy']);
axis([0,5,0,100]); ylabel('Percent of Correct Clasification')
subplot(2,2,4)
bar([mean(svm_eval), mean(nb_eval), mean(tf_eval)], 'r')
set(gca,'xticklabel',{'SVM', 'Naive Bayes', 'Decision Tree'}); title([label_short, ': Method Comparison']);
axis([0,4,0,100]); ylabel('Percent of Correct Clasification')


