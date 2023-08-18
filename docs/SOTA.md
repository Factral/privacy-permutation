# SOTA

papers:

## Learnable Imagen Encryption

<details>

  <summary>
    <h2>
      Detalles
    </h2>
  </summary>


se propone un esquema para mantener la privacidad en las imagenes, el proceso de privacidad es un pipeline de transformacion geoemtricas basadas en bloques, luego cuando se aplican esas transformaciones a esa imagen, la salida pasa por una red que tiene una particularidad en la primera capa y es que el tama;o del kernel y del pooling corresponde al mismo tama;o del bloque utilziado en las transformaciones geometricas, esto es con el fin de seguir operando con lo mismo pese a la transformacion

se entrena la red con las imagenes encriptadas, esto da un rendimiento menor al de las imagenes planas ademas de que necesita el preprocesamiento de las mismas.

se vende por el lado de la vigilancia y la necsidad de la privacidad en este campo

*"The key idea of this scheme is to encrypt images, so that human cannot understand images but the network can be train with encryped images"*

*"The trainer is often different from the data holder who has the training dataset"*

*"A convolutional layer with MxM-sized filter and MxM stride is put for the first layer"*
*"Block-wise Adaptation"*

tambien usan una convolucion de sub-pixel que alfinal es lo mismo que una deconvolucion solo que mas "eficiente" debido a un proceso de reordenamiento de los feature maps

</details>


## Block-wise Scrambled Image Recognition Using Adaptation Network

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  Este paper es una extension del paper Learnable Image Encryption (LE), se propone un esquema para esconder informacion perceptual de los huamnos mientras las maquinas pueden seguir accediendo a la informacion. Lo venden por el lado de los servicios de analisis de imagenes basado en la nube:

  *"Thease systems, however, can be further improved in terms of privacy issues. When a client transfers an image to a cloud server, a third party can view the image."*

  image:

  ![image](https://i.imgur.com/3kqymIu.png)

  Su pipeline propuesto queda de la siguiente manera:

  image => Image transformed => Adaptation Network => Classification Network => Label

  La transformacion se compone de reordamiento por bloques, luego los pixeles en cada bloque se reordenan, esto se hace con una key diferente para cada bloque, luego se integran de nuevo los bloques y esta imagen entra a la red adaptativa que se compone asi:

  Block wise sub-networks => Block integration => permutation network matrix -> reshape original resolution => classification network

  las block wise son una red por cada bloque, luego se integrqa cada bloque se pasa por la capa de permutation matrix para reordenar los bloques, luego un reshape para las dimensiones originales y luego la red del estado del arte

  El rendimiento obtenido es claramente menor respecto a la plain image.
    

  </details>


## Privacy-Preserving deep neural networks with pixel-based image encryption considering data augmentation in the encrypted domain

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>


  Se presenta un esquema que presevera la privcaidad para una red neural profunda, que permite no solo hacer inferencia sobre imagenes privadas si no tambien aplicar aumento de datos sobre las imagenes privada. **el fuerte esta en el aumento de datos**

  Tambien lo venden un poco por el lado de la seguridad en la nube:

  *"There are security issues when using deep learning in cloud environments to train and test data, such as data privacy, data leakage, and unauthorized data acess"*

  ![image](https://i.imgur.com/CyyV3BN.png)


  Definen dos tipos de privacy-preserving computing
  - Perceptual encryption-based (entra dentro del que ellos proponen)
  - Homorphic encryption (HE)

  La transformacion que proponnen se basa encada pixel y es la negative-positive transformation, proponen otra opcional que se basa en reordenar los canales RGB, el resultado de esta transformacion sera la imagen encriptada

  Luego a esta imagen encriptada que en teoria se envia a la nube se le hace el respectivo aumento de datos, solo permiten dos tipos
  - Horizontal/Vertical flip
  - Shifting

  Luego la imagen obtenida entra a la red de adaptacion para que asi las imagenes luego sean compatible con una dnn de clasificacion. Ellos proponen una adaptative netowrk que consiste de capas convolucionales de un kernel y stride de 1.

  Sus resultados obtenidos son menores a los de una imagen plana,ademas muestran que su rendimiento es mejor que los del estado del arte. Aunque como los otros metodos no estan adaptados al tipo de aumento de datos que se hizo y ademas de que parece que no usar la adaptation network de esos papers si no la que ellos propusieron

  </details>


## Pixel-Based Image Encryption Without Key Management for Privacy-Preserving Deep Neural Networks

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>


  Son los mismos autores del paper anterior, solo que este es un jounral, y en vez de enfocarse en la parte del aumento de datos se enfocan en que pueden cambiar la key usada, al cambiar la key la adaptative network sigue teniendo un buen rendimiento

  en eso se basa el paper, que si cambiar la key hay buen rendimiento, entonces la key no tiene que ser compartida y ademas se puede usar una diferente para la inferencia.

  bastante malo ese jorunal

  </details>


## Pixel-Based Image Encryption Without Key Management for Privacy-Preserving Deep Neural Networks


<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  paper malisimo. describe lo mismo que el resto y practicamente es una re-publicacion de otro paper que ya se explico aqui.
  todo se basa en un tipo de transformacion que depende de la key que se le ponga y por alguna razon que no explican el modelo sigue obteniendo un buen rendimiento

</details>



## Image to Perturbation: An Image Transformation Network for Generating Visually Protected Images for Privacy-Preserving Deep Neural Networks

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  En este jorunal se propone un framework en el que no se ncesita mantener llaves con las imagenes transformadas, y que funciona para una cnn en especifico con la key o la imagen plana

  lo venden tambien por el lado de la computacion en la nube y los software as service (Saas)

  Se entrena con dos redes, primero una u-net en la que entra la imagen plana y sale una imagen transformada, esta imagen transformada pasa a una cnn como una res-net, se calcula la loss de clasificacion se calcula la norma l2 de la imagen transformada con la imagen plana y se disminye la suma de las dos losses

  Lo dividen en dos, transformation network y classifcation model. Requieren de un tercero confiable para el respectivo entrenamiento.

  Consiguen resultados bastante parecidos a los de plain image, pero aun sigue teniendo un rendimiento menor.

  ![image](https://i.imgur.com/hk20iyg.png)

  Hacen pruebas de ataques con gans para recuperar la imagen a partir de la transformada, pero demuiestarn que no la pueden recuperar

</details>


## A GAN-Based Image Transformation Scheme for Privacy-Preserving Deep Neural Networks\


<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  Utilizan una GAN para generar las imagenes privadas, lo hacen disminuyendo tres losses, una loss perceptual, una loss de clasificacion y una loss de reconstruccion por cada pixel que seria la l2

  la GAN que se utiliza es una cycleGAN

  hacen prueas para diferentes tipos de ataques y de reconstruccion de las imagenes a partir de las privadas, demuestran que los otros metodos son sensibles a este tipo de ataques , y que con el de ellos consiguen la menor metrica en SSIM

</details>


## Access Control Using Spatially Invariant Permutation of Feature Maps for Semantic Segmentation Models


<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  Proponenn un metodo para controlar el acesso a una red en especifico, esto lo hacen permutado los mapas de caracteristicas selecionnados en una red, estas permutaciones son a lo largo de los canales y no modifican la relacion espacial de cada feature map, por tanto son invariant permutations, estos se permutan con una llave secreta, que sera la entrada para los usuarios autorizados, los usuarios no autorizados no tendran acceso a la llave y por tanto no podran acceder a la red o si ponen una llave incorrecta obtendran un performance muy malo

  *"Considering the expenses neccesary for the expertise, money, and time taken to train a CNN model, a model should be regarded as a kind of intellectual property"*

  *"it is crucial to investigate mechanism for protecting DNN models from unauthorized access and misuse"*

  Evaluan el rendimiento en modelos de segmentacion, ya que en clasificacion ya se habian propuesto muchos modelos en el SOTA, pero en segmentacion ninguno


  ![image](https://i.imgur.com/9eiiGVO.png)

  ese es todo el paper.s

</details>


# Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret-Key

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  Se basa en poner una marca de agua modelos de deep learning utiliszando transformaciones con una llave secreta.

  *"production-level trained DNN models have great bussines alue, and the need to protect models from copyright infrigment is an urgent issue"*

  Todo esta en el entrenamiento, se entrena con imagenes normales y imagenes transformadas para asi maximizar el rendimiento de clasificacion en los dos tipos de imagenes, para que luego un inspector pueda a partir de la key aplicarla a una imagen y verificar quien es el ownership del modelo

  es parte de la investigacion en IP protection pero hace focus solo en owenership verification y deja de lado el access control

</details>


## Privacy-Preserving Image Classification Using ConvMixer with Adaptative Permutation Matrix

<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  Basicamente lo mismo de los papers de arriba, hacen una transformacion por bloques solo que en vez de usar una red adaptativa, ponen una convmixer con una capa de pseudo permutation antes de la entrada a la convmixer. Por alguna raozn la convmixer tiene un muy buen comportamiento al reemplazar la adaptative network


</details>


## Privacy-Preserving Image Classification Using Vision Transformer 


<details>

  <summary>
    <h2>
      Detalles
    </h2>  
    </summary>

  En este enfoque usan un transformer como red de clasificacion, se aprovechan de los patches que se hacen en el transformer para hacer una transformacion por bloques y asi tener una imagen privada

  los resultados que obtienen son bastante buenos.


</details>