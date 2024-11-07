conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox python=3.9 pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia