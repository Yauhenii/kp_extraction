<div align="center">    

# Cryptocurrencies KPA

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/ICLR-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;  )


![CI testing](https://github.com/alronz/cryptocurrencies-kpa/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>

## Description

Key Point Analysis for cryptocurrencies white-papers and more.

## How to run

First, install dependencies

```bash
# clone project   
git clone https://github.com/alronz/cryptocurrencies-kpa

# install project   
cd cryptocurrencies-kpa
make create_environment
# activate the created env, then:
make requirements
 ```   

Next, navigate to any file and run it.

 ```bash
# module folder
cd project

# run 
python main.py    
```

### acknowledgement

The project was largely inspired by [lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers)
. It was changed to depend directly on lighting-pytorch to give more flexibility and to go beyond transformers.
However, many ideas were taken from lightning-transformers specially around the way the models are configured with
hydra.

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
