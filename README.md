# Predicción de pesos de peces

Todos los pescadores alguna vez han tenido el problema de tener huincha para medir los peces, pero no balanza. Este proyecto es un prototipo de una posible solución.

Este repositorio permite entrenar un modelo de regresión avanzado para **predecir el peso de los peces a partir de medidas anatomicas**, y permite servirlo en una API REST. De este modo todos los pescadores del mundo pueden disfrutar de tan increible innovación.

## Instalación:

Instalar la imagen de docker y mapear algun puerto al puerto 5001. Por ejemplo:

```bash
sudo docker run -p 4000:5001
```
monta el servidor en el puerto 4000.

![image](./img/docker.png)


## Usar el servidor:

La información del modelo se puede leer en el path /model-info

![image](./img/test_get.png)

Para realizar predicciones se debe pasar una lista de diccionarios con las siguientes claves:
Species, Length1, Length2, Length3, Height, Width

![image](./img/test_post.png)

## CI/CD:

El repositorio incluye integración continua básica usando GitHub Actions. 

![image](./img/actions.png)

