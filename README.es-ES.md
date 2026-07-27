<!-- <div align="center">
  <h1>Negative Token Merging: Image-based Adversarial Feature Guidance</h1>
  <p>Official Implementation for our paper: Negative Token Merging: Image-based Adversarial Feature Guidance
 </p>
</div>
<br> -->

<div align="center">
  
## Negative Token Merging: Image-based Adversarial Feature Guidance
  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://negtome.github.io/)

[[Paper](https://negtome.github.io/docs/negtome.pdf)] &emsp; [[Project Page](https://negtome.github.io/)] &emsp;  [[🤗 Huggingface Demo Flux ](https://a62ec09c8038aea4ee.gradio.live/)] [[🤗 Huggingface Demo SDXL](https://4e116ddf3ab78d0b07.gradio.live/)]
<!-- [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://negtome.github.io/)  -->
<br>
</div>


---


<!-- ## Negative Token Merging: Image-based Adversarial Feature Guidance -->

Implementación oficial de nuestro artículo: 
**Neg**ative **To**ken **Me**rging: Image-based Adversarial Feature Guidance


### ¿Qué es NegToMe?

El uso de un prompt negativo para evitar la generación de conceptos no deseados se ha convertido en un enfoque ampliamente adoptado. Sin embargo, capturar conceptos visuales complejos utilizando únicamente texto a menudo no es viable (por ejemplo, el niño en el parque en la figura de abajo) y puede ser insuficiente (por ejemplo, para eliminar personajes con derechos de autor).

Proponemos NegToMe, el cual sugiere realizar una guía adversarial directamente utilizando imágenes (en contraposición al uso exclusivo de texto). La idea clave es sencilla: incluso si describir los conceptos no deseados no es efectivo o viable solo con texto (por ejemplo: "niño en el parque" para la figura de abajo), podemos utilizar directamente las características visuales de una imagen de referencia para guiar adversarialmente el proceso de generación.


<div align="center">
  <img src="./docs/method-overview-v1.jpg" alt="Description of Image">
</div>

## Noticias y Actualizaciones
**[2024.12.4]** Lanzamiento inicial con Demo de Gradio.


## Ejemplos y Aplicaciones
Simplemente ajustando la imagen de referencia utilizada, NegToMe permite una gama de aplicaciones personalizadas.

#### Aumento de la Diversidad de Salida
> El uso de NegToMe a través de diferentes salidas mejora la diversidad de los resultados (al alejar las características visuales de cada imagen de las demás durante la difusión inversa)
<div align="center">
  <img src="./docs/diversity-sdxl-v2.jpg" alt="Description of Image">
</div>


#### Mitigación de Derechos de Autor (Copyright)
> Cuando se utiliza una base de datos de recuperación (RAG) con derechos de autor como referencia, NegToMe permite una mejor reducción de la similitud visual con las imágenes protegidas.
<div align="center">
  <img src="./docs/copyright-v1.jpg" alt="Description of Image">
</div>


#### Mejora de la Estética y los Detalles de la Salida
> El simple uso de una referencia borrosa o de mala calidad conduce a una mejora en la estética y los detalles de la salida sin requerir ningún ajuste fino (al alejarse de las características pobres o borrosas).
<div align="center">
  <img src="./docs/output-quality-v2.jpg" alt="Description of Image">
</div>


#### Guía de Estilo Adversarial
> El uso de NegToMe con respecto a una imagen de referencia de estilo ayuda a excluir ciertos elementos artísticos mientras se obtiene el contenido de salida deseado. 
<div align="center">
  <img src="./docs/style-guidance-v1.jpg" alt="Description of Image">
</div>



## TODO / Actualizaciones
- [x] Lanzamiento inicial del código
- [x] Implementación con Flux
- [x] Implementación con SDXL
- [x] Código fuente para la demo de gradio
- [ ] Guía Adversarial Enmascarada (Masked Adversarial Guidance)
- [ ] Demo de Gradio para baja VRAM
- [ ] Lanzamiento de Benchmark para Diversidad y Mitigación de Copyright


## Configuración
Para configurar nuestro entorno, por favor ejecute:

```
conda create -n negtome python=3.11 -y
conda activate negtome
pip install -r requirements.txt
```

<!-- ```
conda env create --name negtome --file=environment/environment.yml
``` -->

## Uso
NegToMe puede incorporarse en solo unas pocas líneas de código en la mayoría de los modelos de difusión state-of-the-art. Actualmente proporcionamos tres formas de usar NegToMe:


### Uso Directo del Pipeline de NegToMe
Primero, cargue el pipeline:
```python
from src.negtome.pipeline_negtome_flux import FluxNegToMePipeline
pipe = FluxNegToMePipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
```

La inferencia con y sin negtome puede ejecutarse de la siguiente manera:
```python
import torch
import time

negtome_args = {
    'use_negtome': False,
    'merging_alpha': 0.9,
    'merging_threshold': 0.65, 
    'merging_t_start': 1000, 
    'merging_t_end': 900,
    'num_joint_blocks': -1, # número de bloques transformadores conjuntos (flux) donde aplicar negtome
    'num_single_blocks': -1, # número de bloques transformadores individuales (flux) donde aplicar negtome
}

# prompt de entrada 
prompt = "a high resolution photo of a person"
print (f"using prompt: {prompt}")

# hiperparámetros
seed = 0 
num_inference_steps = 25
num_images_per_prompt = 4 # generar 4 imágenes en el lote
height = width = 768 

# generar tanto con como sin negtome
inference_times = []
for use_negtome in [False, True]:
    print(f"\nuse_negtome: {use_negtome}")
    generator = torch.Generator(pipe.device).manual_seed(seed)
    
    # Medir tiempo
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        use_negtome=use_negtome,
        negtome_args=negtome_args,
    ).images
    elapsed_time = time.time() - start_time
    inference_times.append(elapsed_time)
    
    print(f"use_negtome: {use_negtome}\nTime taken: {elapsed_time:.2f} seconds")
    display(display_alongside_batch(images, resize_dims=512))

# Calcular porcentaje de incremento
percentage_increase = ((inference_times[1] - inference_times[0]) / inference_times[0]) * 100
print(f"\nPercentage increase in inference time with negtome: {percentage_increase:.2f}%")
```


### Uso del Jupyter Notebook
Ejemplos de uso en ```notebooks/demo-negtome-flux.ipynb``` y ```notebooks/demo-negtome-sdxl.ipynb```

### Iniciar una demo local de gradio
Ejecute el siguiente comando:
```
python gradio_app_negtome.py
```



## Citación
Si encuentra nuestro trabajo útil, por favor considere citar:
```
@article{singh2024negtome,
  title={Negative Token Merging: Image-based Adversarial Feature Guidance},
  author={Singh, Jaskirat and Li, Lindsey and Shi, Weijia and Krishna, Ranjay and Choi, Yejin and 
    Wei, Pang and Gould, Stephen and Zheng, Liang and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv}, 
  url={https://arxiv.org/abs/2408},
  year={2024}
}
```

---
