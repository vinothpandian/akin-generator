# Self-Attention GAN
Tensorflow-2.0 implementation of ["Self-Attention Generative Adversarial Networks."](https://arxiv.org/abs/1805.08318).
## Usage

* train

        python main.py --train

* generate semantic images
    
        python main.py
        
* generate json annotation file from semantic images
    
        python postProcessing.py
        
* generate wireframe images from json
    
        python prototypeGenerator.py --font_path complete_path_to_font_ttf_file
        


the results will show in `./results`.

## Results

![result](sa_result.jpg)

## Reference
[pytorch implementation](https://github.com/heykeetae/Self-Attention-GAN)

[Tensorflow-1.0 implementation](https://github.com/taki0112/Self-Attention-GAN-Tensorflow)

## Contributors
trainer.py contributed by [Kinna](https://github.com/KinnaChen)
