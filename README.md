# **YOLO con modulos de atención** #
## **Diseñado por:**
* *Alba Maria Ramirez Marquinez*
    * **Código:** 2216260
* *Milton Guarin Arias* 
    * **Código:** 2210702
* *Carlos Arbey Mejia Martinez*
    * **Código:** 2210549
* *Andres Felipe Guerra Vargas* 
    * **Código:** 2211058

## **Contenido** ##
En este repositorio se podrá encontrar la documentación, fuentes y resultados llevados a cabo en la comparación de los modelos de la arquitectura YOLO v3 utilizando diferentes modelos de atención aplicados a nuestro DataSet.

*   **Generación DataSet:** Para la generación de nuestro DataSet nos apoyamos en la herramienta **LabelImg** tomada del repositorio Git [LabelImg](https://github.com/tzutalin/labelImg). Se utilizó la aplicación Anaconda para realizar una instalación de entorno python, a este entorno se le instalo lo siguiente para la correcta utilización del programa LabelImg:

    ```bash
    conda install pyqt=5
    conda install -c anaconda lxml
    pyrcc5 -o libs/resources.py resources.qrc
    ```
    Para iniciar el programa LabelImg:
    ```bash
    python labelImg.py
    ```
    <p align="center">
    <img src="img/example.png" alt="" style="height: 400px; width:600px;"/>
    </P>

    En la toma de las imagenes se crearon dos archivos bash Windows de apoyo para la transformación de los nombres de las imagenes **Change_name.ps1** y **Change_name_replace.ps1**, el proposito de estos script es el siguiente:

    *   **Change_name.ps1:** Cambiar el nombre de las imagenes descargas a las nomenclaturas definidas por el equipo. La estructura de los nombres de la imagenes será, **[Inicial_Ingeniero]\_img\_[id].[extension_imagen]**
    *   **Change_name_replace.ps1:** Ajustar los nombres de los archivos .txt con la información de los bounding boxes de las imagenes a que correspondan al mismo nombre de la imagen generando así su emparejamiento.
    <br>
    <br>
*   **Redimensionamiento:** Dado que las imagenes que se buscaron en internet presentan diferentes dimensiones, se creo el colab **Utilidades/Images_transform.ipynb** para redimensionarlas todas a un tamaño predefinido de 416x416, a continuación se puede visualizar un ejemplo del redimensionado que se genero sobre las imagenes.
    <p align="center">
    <img src="img/img_redim.png" alt="" style="height: 400px; width:600px;"/>
    </P>
    
    Las imagenes originales se podrán en contrar en el repositorio [Repositorio Google Drive](https://drive.google.com/drive/folders/1XLkhu0QKoeiVU00qahQfJsPmESpb9Y3l?usp=sharing) en la carpeta **images_ori** y las imagenes redimensionadas se podrán encontrar en la carpeta **images**.

*   **Entrenamiento:** Para llevar a cabo el entrenamiento de YOLOv3 con nuestra clase **Helmet** se creó el Colab **YOLO_tiny_se.ipynb**, el cual presenta los llamados necesarios a la librería de YOLOv3 tomado del repositorio GitHub [PyTorch-YOLOv3](https://github.com/promach/PyTorch-YOLOv3)

## **Links alternos al repositorio:** ##
* [Repositorio Google Drive](https://drive.google.com/drive/folders/1XLkhu0QKoeiVU00qahQfJsPmESpb9Y3l?usp=sharing)

## **Referencias:** ##
* [LabelImg](https://github.com/tzutalin/labelImg)
* [PyTorch-YOLOv3](https://github.com/promach/PyTorch-YOLOv3)
