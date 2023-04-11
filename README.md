# Disentangled and Recurrent Prompt Tunning (DRPT)


## Setup
```
conda create --name clip python=3.7
conda activate clip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/openai/CLIP.git
```
Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.
```
sh download_data.sh
```

If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`data/<dataset>` where `<datasets> = {'mit-states', 'ut-zappos', 'cgqa', 'clevr'}`.


## Training
```
python -u train.py --dataset <dataset>
```

### Evaluation
```
python -u test.py --dataset <dataset>
```
You can replace `--dataset` with `{mit-states, ut-zappos, cgqa, clevr}`.


## References
If you use this code, please cite
```
@article{lu2022decomposed,
  title={Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning},
  author={Lu, Xiaocheng and Liu, Ziming and Guo, Song and Guo, Jingcai},
  journal={arXiv preprint arXiv:2211.10681},
  year={2022}
}
```